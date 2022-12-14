import os

# Vision transformer model
from zsp.method import vision_transformer_flexible as vits
# Correspondence functions
from zsp.method.correspondence_functions import (
    find_correspondences_batch, 
    find_correspondences_original,
    find_correspondences_batch_with_knn
)
# Utils
from zsp.method.zero_shot_pose_utils import (
    rank_target_images_by_global_sim, 
    batch_intersection_over_union,
    normalize_cyclical_dists, 
    scale_points_from_patch, 
    RigidBodyUmeyama,
)
from PIL import Image

import torch
from torchvision import transforms
import numpy as np

from zsp.method.dense_descriptor_utils import (
    _log_bin, gaussian_blurring, extract_saliency_maps)
from zsp.utils.project_utils import get_results_length

# from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.transforms import Rotate, Translate, Scale
from skimage.measure import ransac

class DescriptorExtractor():
    def __init__(
        self,
        patch_size=8,
        feat_layer=9,
        high_res=False,
        binning='none',
        image_size=224,
        n_target=5,
        saliency_map_thresh=0.1,
        num_correspondences=50,
        kmeans=False,
        best_frame_mode="corresponding_feats_similarity",
    ):
        # if do_log_bin and do_gaussian_blur:
        #     print('Warning: both gaussian blur and log bin flags set to True')
        self.patch_size = patch_size
        self.feat_layer = feat_layer
        self.high_res = high_res
        self.binning = binning
        # self.do_log_bin = do_log_bin
        # self.do_gaussian_blur = do_gaussian_blur
        self.image_size = image_size
        self.n_target = n_target
        self.saliency_map_thresh = saliency_map_thresh
        self.num_correspondences = num_correspondences
        self.best_frame_mode = best_frame_mode

        if self.patch_size == 16:
            self.model_name = 'vit_base'
            self.stride = 8
            self.num_patches = 14
            self.padding = 5
        elif self.patch_size == 8:
            self.model_name = 'vit_small'
            self.stride = 4
            self.num_patches = 28
            self.padding = 2
        else:
            raise ValueError('ViT models only supported with patch sizes 8 or 16')

        if self.high_res: 
            self.num_patches *= 2

        if kmeans:
            self.correspondence_mode = 'batch_knn'       # ('batch', 'batch_knn', 'original')
        else:
            self.correspondence_mode = 'batch'       # ('batch', 'batch_knn', 'original')
        self.batched_correspond = True
        # Image processing
        self.image_norm_mean = (0.485, 0.456, 0.406)
        self.image_norm_std = (0.229, 0.224, 0.225)

        # Initialise None model - this gets loaded by call to load_model
        self.model = None


    def load_model(self, pretrain_path, device):
        model = vits.__dict__[self.model_name](patch_size=self.patch_size)
        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        if self.high_res: 
            model.patch_embed.proj.stride = (self.stride, self.stride)
            model.num_patches = self.num_patches ** 2
            model.patch_embed.patch_size = self.stride
            model.patch_embed.proj.padding = self.padding
        self.model = model


    def extract_features_and_attn(self, all_images):
        """
        A definition of relevant dimensions {all_b, nh, t, d}:
            image_size: Side length of input images (assumed square)
            all_b: The first dimension size of the input tensor - not necessarily
                the same as "batch size" in high-level script, as we assume that
                reference and target images are all flattened-then-concatenated
                along the batch dimension. With e.g. a batch size of 2, and 5 target
                images, 1 reference image; all_b = 2 * (5+1) = 12
            h: number of heads in ViT, e.g. 6
            t: number of items in ViT keys/values/tokens, e.g. 785 (= 28*28 + 1)
            d: feature dim in ViT, e.g. 64

        Args:
            all_images (torch.Tensor): shape (all_b, 3, image_size, image_size)
        Returns:
            features (torch.Tensor): shape (all_b, nh, t, d) e.g. (12, 6, 785, 64)
            attn (torch.Tensor): shape (all_b, nh, t, t) e.g. (12, 6, 785, 785)
            output_cls_tokens (torch.Tensor): shape (all_b, nh*d) e.g. (12, 384)
        """
        MAX_BATCH_SIZE = 50
        all_images_batch_size = all_images.size(0)
        c, img_h, img_w = all_images.shape[-3:]
        all_images = all_images.view(-1, c, img_h, img_w)

        with torch.no_grad():
            torch.cuda.empty_cache()

            if all_images_batch_size <= MAX_BATCH_SIZE:
                # import ipdb; ipdb.set_trace()
                data = self.model.get_specific_tokens(all_images, layers_to_return=(9, 11))
                features = data[self.feat_layer]['k']
                attn = data[11]['attn']
                output_cls_tokens = data[11]['t'][:, 0, :]

            # Process in chunks to avoid CUDA out-of-memory
            else:
                num_chunks = np.ceil(all_images_batch_size / MAX_BATCH_SIZE).astype('int')
                data_chunks = []
                for i, ims_ in enumerate(all_images.chunk(num_chunks)):
                    data_chunks.append(self.model.get_specific_tokens(ims_, layers_to_return=(9, 11)))

                features = torch.cat([d[self.feat_layer]['k'] for d in data_chunks], dim=0)
                attn = torch.cat([d[11]['attn'] for d in data_chunks], dim=0)
                output_cls_tokens = torch.cat([d[11]['t'][:, 0, :] for d in data_chunks], dim=0)

        return features, attn, output_cls_tokens

    def create_reshape_descriptors(self, features, attn, batch_size, device):
        """
        Relevant dimensions are defined as for extract_features_and_attn method above
        
        3 new dimension params here are:
            B: This is the batch size used in the dataloader/calling script
            n_tgt: This is equal to self.n_target
            feat_dim: This is the dimensionality of the descriptors - while related
                to the ViT feature dimension, it may have undergone further binning
                procedures that will increase its dimension, or dimensionality reduction
                approaches to *decrease* the dimension

        Args:
            features (torch.Tensor): shape (all_b, nh, t, d) e.g. (12, 6, 785, 64)
            attn (torch.Tensor): shape (all_b, nh, t, t) e.g. (12, 6, 785, 785)
            output_cls_tokens (torch.Tensor): shape (all_b, nh*d) e.g. (12, 384)

        Returns:
            features (torch.Tensor): shape Bx(n_tgt+1)x1x(t-1)xfeat_dim, this is
                a descriptor tensor, rather than raw features from the ViT. 
            attn (torch.Tensor): shape Bx(n_tgt+1)xhxtxt, this is the spatial
                self-attention maps
        """
        all_b, h, t, d = features.size()
        # Remove cls output (first 'patch') from features
        features = features[:, :, 1:, :]  # (all_b) x h x (t-1) x d  e.g. (12, 6, 784, 64)
        # Roll multiple ViT heads into a single feature, re-add head dimension
        features = features.permute(0, 2, 1, 3).reshape(all_b, t-1, h * d)[:, None, :, :]  # all_b x 1 x (t-1) x (d*h)
        if self.binning == 'log':
            features = _log_bin(features, device=device) # all_b x 1 x (t - 1) x (17*d*h), with 17 from the log binning
        elif self.binning == 'gaussian':
            features = gaussian_blurring(features.squeeze(1), kernel_size=7, sigma=2)
            features = features[:, None, :, :]
        elif self.binning == 'none':
            pass
        else:
            raise ValueError(f"{self.binning} is not a valid choice for the 'binning' parameter of the DescriptorExtractor")

        # Reshape back to batched view
        _, _, _, feat_dim_after_binning = features.size()
        features = features.view(batch_size, -1, 1, t-1, feat_dim_after_binning) # Bx(n_tgt+1)x1x(t-1)xfeat_dim

        attn = attn.view(batch_size, -1, h, t, t)       # B x (n_tgt+1) x h x t x t

        # Ensure descriptors & attn are on the correct device
        features = features.to(device)
        attn = attn.to(device)
        return features, attn

    def split_ref_target(self, features, attn):
        """
        Reshapes, repeats and splits features and attention into ref/tgt

        Specifically, this function splits out the reference and target descriptors/attn,
        repeats the reference image n_tgt times, and flattens the n_tgt dimension
        into the batch dimension.

        Dimensions as for extract_features_and_attn, create_reshape_descriptors
        Args:
            features (torch.Tensor): shape Bx(n_tgt+1)x1x(t-1)xfeat_dim, this is
                a descriptor tensor, rather than raw features from the ViT. 
            attn (torch.Tensor): shape Bx(n_tgt+1)xhxtxt, this is the spatial
                self-attention maps

        Returns:
            ref_feats (torch.Tensor): shape (B*n_tgt)x1x(t-1)xfeat_dim, this is
                a descriptor tensor, repeated n_tgt times to match target_feats. 
            target_feats (torch.Tensor): shape (B*n_tgt)x1x(t-1)xfeat_dim, this is
                the tensor of descriptors for the target images
            ref_attn (torch.Tensor): shape (B*n_tgt)xhxtxt, the reference im's spatial
                self-attention map, repeated n_tgt times to match target_feats
            target_attn (torch.Tensor): shape (B*n_tgt)xhxtxt, the spatial
                self-attention maps for the target images    
        """
        batch_size, _, h, t, t = attn.size()
        feat_dim_after_binning = features.size()[-1]
        
        # Split descriptors, attn back to reference image & target images
        ref_feats, target_feats = features.split((1, self.n_target), dim=1)
        ref_attn, target_attn = attn.split((1, self.n_target), dim=1)

        ref_feats = ref_feats.repeat(1, self.n_target, 1, 1, 1)
        ref_attn = ref_attn.repeat(1, self.n_target, 1, 1, 1)

        # Flatten first 2 dims again:
        ref_feats = ref_feats.view(batch_size * self.n_target, 1, t - 1, feat_dim_after_binning)
        ref_attn = ref_attn.view(batch_size * self.n_target, h, t, t)

        target_feats = target_feats.reshape(batch_size * self.n_target, 1, t - 1, feat_dim_after_binning)
        target_attn = target_attn.reshape(batch_size * self.n_target, h, t, t)
        return ref_feats, target_feats, ref_attn, target_attn


    def get_correspondences(self, ref_feats, target_feats, ref_attn, target_attn, device):
        """
        Args:
            ref_feats (torch.Tensor): shape (B*n_tgt)x1x(t-1)xfeat_dim, this is
                a descriptor tensor, repeated n_tgt times to match target_feats. 
            target_feats (torch.Tensor): shape (B*n_tgt)x1x(t-1)xfeat_dim, this is
                the tensor of descriptors for the target images
            ref_attn (torch.Tensor): shape (B*n_tgt)xhxtxt, the reference im's spatial
                self-attention map, repeated n_tgt times to match target_feats
            target_attn (torch.Tensor): shape (B*n_tgt)xhxtxt, the spatial
                self-attention maps for the target images 

        Returns:
            selected_points_image_2 (torch.Tensor): Shape (Bxn_tgt)xself.num_correspondencesx2, 
                this is a tensor giving the 
            selected_points_image_1 (torch.Tensor):
            cyclical_dists (torch.Tensor):
            sim_selected_12 (torch.Tensor):
            
        """

        # Note flipped way in which features and attention maps are passed to find correspondence function
        if self.correspondence_mode == 'batch':

            selected_points_image_2, selected_points_image_1, cyclical_dists, sim_selected_12 = find_correspondences_batch(
                descriptors1=target_feats,
                descriptors2=ref_feats,
                attn1=target_attn,
                attn2=ref_attn,
                device=device,
                num_pairs=self.num_correspondences)

        elif self.correspondence_mode == 'original':
            raise NotImplementedError("'original' (non-batched) correspondence mode not currently supported")
            selected_points_image_1_batch = []
            selected_points_image_2_batch = []
            sim_selected_12_batch = []
            for i in range(len(target_feats)):
                selected_points_image_2, selected_points_image_1, sim_selected_12 = find_correspondences_original(
                    descriptors1=target_feats[i:i+1],
                    descriptors2=ref_feats[i:i+1],
                    attn1=target_attn[i:i+1],
                    attn2=ref_attn[i:i+1],
                    device=device,
                    num_pairs=self.num_correspondences)
                selected_points_image_1_batch.append(selected_points_image_1)
                selected_points_image_2_batch.append(selected_points_image_2)
                sim_selected_12_batch.append(sim_selected_12)
            selected_points_image_1 = torch.stack(selected_points_image_1_batch, dim=0)
            selected_points_image_2 = torch.stack(selected_points_image_2_batch, dim=0)
            sim_selected_12 = torch.stack(sim_selected_12_batch, dim=0)

        elif self.correspondence_mode == 'batch_knn':
            if self.high_res:
                num_pairs_for_topk = 8 * self.num_correspondences
            else:
                num_pairs_for_topk = 2 * self.num_correspondences

            selected_points_image_2, selected_points_image_1, cyclical_dists, sim_selected_12 = \
                find_correspondences_batch_with_knn(
                descriptors1=target_feats,
                descriptors2=ref_feats,
                attn1=target_attn,
                attn2=ref_attn,
                device=device,
                num_pairs_for_topk=num_pairs_for_topk)

        return (selected_points_image_2, selected_points_image_1, cyclical_dists, sim_selected_12)

    def find_closest_match(self, attn, output_cls_tokens, sim_selected_12, batch_size):
        batch_size, _, h, t, t = attn.size()
        # N is the height or width of the feature map
        N = int(np.sqrt(t - 1))

        ref_attn, target_attn = attn.split((1, self.n_target), dim=1)
        ref_attn = ref_attn.repeat(1, self.n_target, 1, 1, 1)
        # Flatten first 2 dims again:
        ref_attn = ref_attn.view(batch_size * self.n_target, h, t, t)
        target_attn = target_attn.reshape(batch_size * self.n_target, h, t, t)

        if self.best_frame_mode == 'global_similarity':
            output_cls_tokens = output_cls_tokens.view(batch_size, self.n_target + 1, -1)
            ref_global_feats, target_global_feats = output_cls_tokens.split((1, self.n_target), dim=1)
            similarities = rank_target_images_by_global_sim(ref_global_feats, target_global_feats)
            best_idxs = similarities.argmax(dim=-1)
            best_idxs = best_idxs.squeeze()
        elif self.best_frame_mode == 'ref_to_target_saliency_map_iou':
            # TODO: Can make the below quicker by calling on just one copy of ref_attn, *then* 'repeat' 
            saliency_map_ref = extract_saliency_maps(ref_attn) # (B*n_tgt)x(t-1), with (t-1)=(N**2)
            saliency_map_target = extract_saliency_maps(target_attn) # (B*n_tgt)x(t-1), with (t-1)=(N**2)
            # Reshape saliency maps to (B*n_tgt)xNxN (N is height and width of feature map)
            saliency_map_ref, saliency_map_target = (x.view(batch_size * self.n_target, N, N)
                                                    for x in (saliency_map_ref, saliency_map_target))
            # Compute IoUs
            similarities, intersection_map = batch_intersection_over_union(
                saliency_map_ref, saliency_map_target, threshold=self.saliency_map_thresh)
            similarities = similarities.view(batch_size, self.n_target)
            best_idxs = similarities.argmax(dim=-1)
        elif self.best_frame_mode == 'corresponding_feats_similarity':
            sim_selected_12 = sim_selected_12.view(batch_size, self.n_target, self.num_correspondences)
            similarities = sim_selected_12.sum(dim=-1)
            best_idxs = similarities.argmax(dim=-1)
        elif self.best_frame_mode == 'cylical_dists_to_saliency_map_iou':
            saliency_map_target = extract_saliency_maps(target_attn)
            saliency_map_target = saliency_map_target.view(batch_size * self.n_target, N, N)

            cyclical_dists = normalize_cyclical_dists(cyclical_dists)
            similarities, intersection_map = batch_intersection_over_union(
                cyclical_dists, saliency_map_target, threshold=self.saliency_map_thresh)
            similarities = similarities.view(batch_size, self.n_target)
            best_idxs = similarities.argmax(dim=-1)
        else:
            raise ValueError(f'Method of picking best frame not implemented: {self.best_frame_mode}')
        return similarities, best_idxs


    def scale_patch_to_pix(self, points1, points2, N):
        """
        Args:
            points1 (torch.Tensor): shape num_correspondencesx2, the *patch* coordinates
                of correspondence points in image 1 (the reference image)
            points2 (torch.Tensor): shape num_correspondencesx2, the *patch* coordinates
                of correspondence points in image 2 (the best target image)
            N (int): N is the height or width of the feature map
        """
        if self.batched_correspond:
            points1_rescaled, points2_rescaled = (scale_points_from_patch(
                p, vit_image_size=self.image_size, num_patches=N) for p in (points1, points2))
        else: # earlier descriptor extractor functions for ViT features scaled before return
            points1_rescaled, points2_rescaled = (points1, points2)
        return points1_rescaled, points2_rescaled


    def get_transform(self):
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_norm_mean, std=self.image_norm_std)
        ])
        return image_transform


    def denorm_torch_to_pil(self, image):
        image = image * torch.Tensor(self.image_norm_std)[:, None, None]
        image = image + torch.Tensor(self.image_norm_mean)[:, None, None]
        return Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))


class ZeroShotPoseMethod():
    def __init__(
        self,
        num_samples_per_class=100,
        batched_correspond=True,
        num_plot_examples_per_batch=1,
        saliency_map_thresh=0.1,
        ransac_thresh=0.2,
        n_target=5,
        num_correspondences=50,
        take_best_view=False,    # if True, simply use the best view as the pose estimate
        ransac_min_samples=4,
        ransac_max_trials=10000,
    ):
        self.num_samples_per_class = num_samples_per_class
        self.batched_correspond = batched_correspond
        self.num_plot_examples_per_batch = num_plot_examples_per_batch
        self.saliency_map_thresh = saliency_map_thresh
        self.ransac_thresh = ransac_thresh
        self.n_target = n_target
        self.num_correspondences = num_correspondences
        self.take_best_view = take_best_view
        self.ransac_min_samples = ransac_min_samples
        self.ransac_max_trials = ransac_max_trials
        
    def make_log_dirs(self, log_dir, category):
        cat_log_dir = os.path.join(log_dir, category)
        if os.path.exists(cat_log_dir):
            results_file = os.path.join(cat_log_dir, f"results.txt")
            num_run = get_results_length(results_file)
            if num_run >= self.num_samples_per_class:
                print(f"CATEGORY {category} ALREADY RUN -- SKIPPING!")
                return 0
            else:
                print(F"CATEGORY {category} RUN FOR {num_run} ITEMS - REMOVING, RE-RUNNING")
                if os.path.exists(results_file):
                    os.remove(results_file)
        fig_dir = os.path.join(cat_log_dir, 'plots')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        return cat_log_dir, fig_dir

    def solve_umeyama_ransac(self, world_corr1, world_corr2):
        rbt_model, inliers = ransac(data=(world_corr1, world_corr2),
                            model_class=RigidBodyUmeyama,
                            min_samples=4,
                            residual_threshold=self.ransac_thresh,
                            # max_trials=10000)
                            max_trials=1000)
        # print(f"{sum(inliers)} inliers from {self.num_correspondences} points")
        # if rbt_model.lam == 1:
        #     print("UMEYAMA RETURNED IDENTITY AS FALLBACK")
        R = rbt_model.T[:3, :3] / rbt_model.lam
        t = rbt_model.T[:3, 3:]
        scale = rbt_model.lam

        R_ = Rotate(torch.Tensor(R.T).unsqueeze(0))
        T_ = Translate(torch.tensor(t.T))
        S_ = Scale(scale)
        # trans21 = get_world_to_view_transform(torch.Tensor(R.T).unsqueeze(0), torch.tensor(t.T))
        trans21 = S_.compose(R_.compose(T_))
        return trans21