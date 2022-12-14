# Co3D
from zsp.datasets.co3d_dataset import Co3dDataset
# from zsp.datasets.co3d_dataset_fix import Co3dDataset

# ML
from torch.utils.data import Dataset
import torch
import numpy as np
import random

# I/O
import os
import json
from PIL import Image

# Typing
from typing import List

co3d_root = ''
label_dir = ''

class Co3DPoseDataset(Dataset):

    def __init__(self, dataset_root=co3d_root,
                 categories: List[str] = ['bicycle'],
                 num_samples_per_class: int = 5,
                 target_frames_sampling_mode: str = 'uniform',
                 num_frames_in_target_seq: int = 5,
                 label_dir: str = label_dir,
                 determ_eval_root: str = None,
                 image_transform=None):

        # Get all candidate sequences for a class
        candidate_seqs = {}
        for cat in categories:

            candidate_seqs[cat] = []
            cat_dir = os.path.join(label_dir, cat)

            for f in os.listdir(cat_dir):
                candidate_seqs[cat].append(f.rstrip('.json'))

        # Dict of datasets, one for each class
        self.classes = categories
        self.all_datasets = {
            cls: self.get_base_dataset(dataset_root, cls)
            for cls in self.classes
        }

        # Stores which sequences contain which Frame IDs
        self.labelled_seqs_to_frames = {cls: self.construct_labelled_seq_dict(cls, candidate_seqs[cls])
                              for cls in self.classes}

        # Stores all the sequences which have labels
        self.labelled_seqs = candidate_seqs

        # Computes and stores samples of (reference_frame, [target_frame_0...target_frame_N]) pairs
        # Stored in format
        # sample = {
        #         'class': class,
        #         'reference_seq_name': root_seq_name,
        #         'reference_frame_id': root_frame_id,
        #         'target_seq_name': target_seq_name,
        #         'all_target_seq_id': target_frame_ids
        #         }
        if determ_eval_root:
            if len(categories) != 1:
                raise ValueError('Can only use determ_eval_path when a single category is being loaded')
            determ_eval_path = os.path.join(determ_eval_root, f'{categories[0]}.pt')
            if os.path.exists(determ_eval_path):
                self.samples = torch.load(determ_eval_path)
                if len(self.samples) < num_samples_per_class:
                    raise ValueError(
                        f'Only {len(self.samples)} samples in the determ set, but asked for {num_samples_per_class}')
                self.samples = self.samples[:num_samples_per_class]
                print(f"Successfully loaded determ eval set for category {categories[0]}, kept {len(self.samples)} samples")
            else:
                self.samples = self.sample_instances(num_samples_per_class=num_samples_per_class,
                                                    num_frames_in_target_seq=num_frames_in_target_seq,
                                                    target_frame_sampling_mode=target_frames_sampling_mode)
        else:
            self.samples = self.sample_instances(num_samples_per_class=num_samples_per_class,
                                                num_frames_in_target_seq=num_frames_in_target_seq,
                                                target_frame_sampling_mode=target_frames_sampling_mode)
        # Load ground truth coordinate frames
        self.seq_world_coords = self.load_seq_world_coords(label_dir)

        # Transform on image data
        self.image_transform = image_transform

    def get_base_dataset(self, dataset_root, cls):
        return Co3dDataset(frame_annotations_file=os.path.join(dataset_root, cls, 'frame_annotations.jgz'),
                                sequence_annotations_file=os.path.join(dataset_root, cls, 'sequence_annotations.jgz'),
                                # subset_lists_file=os.path.join(dataset_root, cls, 'set_lists.json'),
                                dataset_root=dataset_root,
                                box_crop=True,
                                box_crop_context=0.1,
                                image_height=None,      # Doesn't resize
                                image_width=None,       # Doesn't resize
                                load_point_clouds=True
                                )
        
    def sample_instances(self, num_samples_per_class=1,
                    num_frames_in_target_seq=5, target_frame_sampling_mode='uniform'):

        """
        For all categories, compare 'num_samples_per_class' sequences
        Choose sequence as the root and randomly sample a FrameID
        Sample 'num_frames_in_target_seq' frame IDs from the other frame
        Two sampling modes for the target frames:
            * 'uniform': Randomly sample a first index then uniformly sample the rest
            * 'random': Randomly sample all target frames
        Store all of this in a list of dicts
        """

        def custom_range(n, end, start=0):
            return list(range(start, n)) + list(range(n+1, end))

        np.random.seed(0)
        random.seed(0)

        all_samples = []

        for cls in self.classes:

            # First, sample which pairs of sequences we are going to compare
            labelled_seqs = self.labelled_seqs[cls]
            labelled_seqs = sorted(labelled_seqs)
            num_seqs_in_cls = len(labelled_seqs)
            seq_pairs = []

            root_seq_id = 0
            num_samples_in_this_class_so_far = 0
            while num_samples_in_this_class_so_far < num_samples_per_class:

                root_seq_name = labelled_seqs[root_seq_id]
                target_seq_id = random.choice(custom_range(root_seq_id, num_seqs_in_cls, start=0))
                target_seq_name = labelled_seqs[target_seq_id]
                seq_pairs.append((root_seq_name, target_seq_name))

                num_samples_in_this_class_so_far += 1
                root_seq_id += 1
                root_seq_id = root_seq_id % num_seqs_in_cls

            # Now sample frame IDs from each pair of sequences
            # One frame from the root, and 'num_frames_in_target_seq' in the target
            for root_seq_name, target_seq_name in seq_pairs:

                # Sample frame from root sequence
                root_frame_number = random.choice(
                    range(len(self.labelled_seqs_to_frames[cls][root_seq_name]))
                )

                root_frame_id = self.labelled_seqs_to_frames[cls][root_seq_name][root_frame_number]

                # Sample frames from target sequence
                # Uniform sampling:
                len_current_target_sequence = len(self.labelled_seqs_to_frames[cls][target_seq_name])
                if target_frame_sampling_mode == 'uniform':
                    target_frame_number = random.choice(
                        range(len(self.labelled_seqs_to_frames[cls][target_seq_name]))
                    )
                    target_frame_ids = [
                        self.labelled_seqs_to_frames[cls][target_seq_name][target_frame_number]
                    ]
                    while len(target_frame_ids) < num_frames_in_target_seq:

                        target_frame_number = (target_frame_number + len_current_target_sequence // num_frames_in_target_seq) % len_current_target_sequence
                        target_frame_ids.append(
                            self.labelled_seqs_to_frames[cls][target_seq_name][target_frame_number]
                        )

                # Random sampling:
                elif target_frame_sampling_mode == 'random':

                    target_frame_numbers = random.sample(range(len_current_target_sequence),
                                                         k=num_frames_in_target_seq)
                    target_frame_ids = [
                        self.labelled_seqs_to_frames[cls][target_seq_name][i]
                        for i in target_frame_numbers
                    ]

                else:

                    raise ValueError

                # Construct sample:
                sample = {
                    'class': cls,
                    'reference_seq_name': root_seq_name,
                    'reference_frame_id': root_frame_id,
                    'target_seq_name': target_seq_name,
                    'all_target_id': target_frame_ids
                }

                all_samples.append(sample)

        return all_samples

    def construct_labelled_seq_dict(self, cls, labelled_seq_names):

        ds = self.all_datasets[cls]

        seqs_to_frames = dict([(seq_name, {}) for seq_name in labelled_seq_names])
        for i, frame_ann in enumerate(ds.frame_annots):
            frame_ann = frame_ann['frame_annotation']
            if frame_ann.sequence_name in labelled_seq_names:
                seqs_to_frames[frame_ann.sequence_name][frame_ann.frame_number] = i

        return seqs_to_frames


    def load_seq_world_coords(self, label_dir):

        cat_labelled_dict = {}
        for cat in os.listdir(label_dir):
            cat_labelled_dict[cat] = {}
            cat_dir = os.path.join(label_dir, cat)

            for f in os.listdir(cat_dir):

                label_path = os.path.join(cat_dir, f)
                with open(label_path, 'r') as json_file:
                    seq_trans = json.load(json_file)

                cat_labelled_dict[cat][seq_trans['seq']] = np.array(seq_trans['trans'])

        return cat_labelled_dict


    def process_list_of_frames(self, frames):

        """
        Take a list of frames and return list of:
            Images (PIL or Tensor)
            Scalings
            Depth maps
            Cameras
        """
        images = [co3d_rgb_to_pil(f.image_rgb) for f in frames]

        if self.image_transform is not None:

            images = [self.image_transform(im) for im in images]
            scalings = [
                np.array(f.image_rgb.shape[1:]) / np.array(im.shape[1:]) for im, f in zip(images, frames)
            ]

        else:

            scalings = [
                torch.Tensor([1, 1]) for _ in frames
            ]

        # Get depth maps
        depth_map = [f.depth_map for f in frames]

        # Get cameras
        cameras = [f.camera for f in frames]

        # Pointclouds (just one needed per seq - same for every frame!)
        pcd = frames[0].sequence_point_cloud

        return images, depth_map, scalings, cameras, pcd


    def __getitem__(self, item):

        sample = self.samples[item]
        cls = sample['class']
        ref_seq_name = sample['reference_seq_name']
        target_seq_name = sample['target_seq_name']

        ref_frame_id = sample['reference_frame_id']
        all_target_id = sample['all_target_id']

        base_dataset = self.all_datasets[cls]

        # Get frames
        ref_frame = base_dataset[ref_frame_id]
        all_target_frames = [base_dataset[idx] for idx in all_target_id]

        # Get world coordinate transforms for each seq
        ref_transform = self.seq_world_coords[cls][ref_seq_name]
        target_transform = self.seq_world_coords[cls][target_seq_name]

        # Process ref frame
        ref_image, ref_depth_map, ref_scaling, ref_camera, ref_pcd = (x[0] for x in self.process_list_of_frames([ref_frame]))

        # Process target frames
        all_target_images, all_target_depth_maps, all_target_scalings, \
        all_target_cameras, target_pcd = self.process_list_of_frames(all_target_frames)

        return ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
               all_target_images, target_transform, all_target_depth_maps, \
                all_target_scalings, all_target_cameras, ref_pcd, target_pcd, item


    def __len__(self):
        return len(self.samples)


class ICPCo3DPoseDataset(Co3DPoseDataset):

    def __init__(self, *args, **kwargs):

        super(ICPCo3DPoseDataset, self).__init__(*args, **kwargs)

    def get_base_dataset(self, dataset_root, cls):
        return Co3dDataset(frame_annotations_file=os.path.join(dataset_root, cls, 'frame_annotations.jgz'),
                                sequence_annotations_file=os.path.join(dataset_root, cls, 'sequence_annotations.jgz'),
                                subset_lists_file=os.path.join(dataset_root, cls, 'set_lists.json'),
                                dataset_root=dataset_root,
                                box_crop=True,
                                box_crop_context=0.1,
                                image_height=224,      # Doesn't resize
                                image_width=224,       # Doesn't resize
                                load_point_clouds=True
                                )

    def process_list_of_frames(self, frames):

        """
        Take a list of frames and return list of:
            Images (PIL or Tensor)
            Scalings
            Depth maps
            Cameras
        """
        images = [co3d_rgb_to_pil(f.image_rgb) for f in frames]

        if self.image_transform is not None:

            images = [self.image_transform(im) for im in images]
            scalings = [
                np.array(f.image_rgb.shape[1:]) / np.array(im.shape[1:]) for im, f in zip(images, frames)
            ]

        else:

            scalings = [
                torch.Tensor([1, 1]) for _ in frames
            ]

        # Get depth maps
        depth_map = [f.depth_map for f in frames]

        # Get cameras
        cameras = [f.camera for f in frames]

        # Pointclouds (just one needed per seq - same for every frame!)
        pcd = frames[0].sequence_point_cloud

        # Foreground probability maps
        fg_probs = [f.fg_probability for f in frames]

        # Original image_rgb - for unprojecting single/few images to pointclouds
        image_rgbs = [f.image_rgb for f in frames]

        return images, depth_map, scalings, cameras, pcd, fg_probs, image_rgbs


    def __getitem__(self, item):

        sample = self.samples[item]
        cls = sample['class']
        ref_seq_name = sample['reference_seq_name']
        target_seq_name = sample['target_seq_name']

        ref_frame_id = sample['reference_frame_id']
        all_target_id = sample['all_target_id']

        base_dataset = self.all_datasets[cls]

        # Get frames
        ref_frame = base_dataset[ref_frame_id]
        all_target_frames = [base_dataset[idx] for idx in all_target_id]

        # Get world coordinate transforms for each seq
        ref_transform = self.seq_world_coords[cls][ref_seq_name]
        target_transform = self.seq_world_coords[cls][target_seq_name]

        # Process ref frame
        ref_image, ref_depth_map, ref_scaling, ref_camera, ref_pcd, ref_fgprob, ref_image_rgb = (
            x[0] for x in self.process_list_of_frames([ref_frame])
        )

        # Process target frames
        all_target_images, all_target_depth_maps, all_target_scalings, all_target_cameras, \
        target_pcd, all_target_fgprobs, all_target_image_rgb = self.process_list_of_frames(
            all_target_frames
        )

        # Now also return additional cameras for the 0th frame from the reference and target sequences
        base_dataset = self.all_datasets[cls]
        ref_sequence_zero_id = self.labelled_seqs_to_frames[cls][ref_seq_name][0]
        target_sequence_zero_id = self.labelled_seqs_to_frames[cls][target_seq_name][0]

        ref_sequence_zero_camera = base_dataset[ref_sequence_zero_id].camera
        target_sequence_zero_camera = base_dataset[target_sequence_zero_id].camera

        return ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
               all_target_images, target_transform, all_target_depth_maps, \
               all_target_scalings, all_target_cameras, ref_pcd, target_pcd, \
               ref_fgprob, all_target_fgprobs, ref_sequence_zero_camera, target_sequence_zero_camera, \
               ref_image_rgb, all_target_image_rgb, item


class PlottingCo3DPoseDataset(ICPCo3DPoseDataset):

    def __init__(self, *args, **kwargs):

        super(PlottingCo3DPoseDataset, self).__init__(*args, **kwargs)

    def get_base_dataset(self, dataset_root, cls):
        return Co3dDataset(frame_annotations_file=os.path.join(dataset_root, cls, 'frame_annotations.jgz'),
                                sequence_annotations_file=os.path.join(dataset_root, cls, 'sequence_annotations.jgz'),
                                subset_lists_file=os.path.join(dataset_root, cls, 'set_lists.json'),
                                dataset_root=dataset_root,
                                box_crop=True,
                                box_crop_context=0.1,
                                image_height=None,      # Doesn't resize
                                image_width=None,       # Doesn't resize
                                load_point_clouds=True
                                )


def co3d_rgb_to_pil(image_rgb):
    return Image.fromarray((image_rgb.permute(1, 2, 0) * 255).numpy().astype(np.uint8))


def co3d_pose_dataset_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera, \
    all_target_images, target_transform, all_target_depth_maps, \
    all_target_scalings, all_target_cameras, ref_pcd, target_pcd, item = zip(*batch)

    ref_image = torch.stack(ref_image)
    all_target_images = torch.stack([torch.stack(x) for x in all_target_images])
    ref_transform = torch.as_tensor(np.stack(ref_transform))
    target_transform = torch.as_tensor(np.stack(target_transform))

    idx_into_dataset = torch.as_tensor(item)

    ref_meta_data = {
        'depth_maps': ref_depth_map,
        'scalings': torch.as_tensor(np.stack(ref_scaling)),
        'cameras': ref_camera,
        'pcd': ref_pcd
    }

    target_meta_data = {
        'depth_maps': all_target_depth_maps,
        'scalings': torch.as_tensor(np.stack(all_target_scalings)),
        'cameras': all_target_cameras,
        'pcd': target_pcd
    }

    return (ref_image, all_target_images), (ref_transform, target_transform), (ref_meta_data, target_meta_data), idx_into_dataset



def co3d_icp_pose_dataset_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
    all_target_images, target_transform, all_target_depth_maps, \
    all_target_scalings, all_target_cameras, ref_pcd, target_pcd, \
    ref_fgprob, all_target_fgprobs, ref_sequence_zero_camera, target_sequence_zero_camera, \
    ref_image_rgb, all_target_image_rgb, item = zip(*batch)

    # Stack images
    ref_image = torch.stack(ref_image)
    all_target_images = torch.stack([torch.stack(x) for x in all_target_images])
    # Stack fullsize, original images
    ref_image_rgb = torch.stack(ref_image_rgb)
    all_target_image_rgb = torch.stack([torch.stack(x) for x in all_target_image_rgb])
    # Stack transforms
    ref_transform = torch.as_tensor(np.stack(ref_transform))
    target_transform = torch.as_tensor(np.stack(target_transform))
    # Stack depth_map
    ref_depth_map = torch.stack(ref_depth_map)
    all_target_depth_maps = torch.stack([torch.stack(x) for x in all_target_depth_maps])
    # Stack fg_probability
    ref_fgprob = torch.stack(ref_fgprob)
    all_target_fgprobs = torch.stack([torch.stack(x) for x in all_target_fgprobs])

    idx_into_dataset = torch.as_tensor(item)

    ref_meta_data = {
        'image_rgb': ref_image_rgb,
        'depth_maps': ref_depth_map,
        'fg_probability': ref_fgprob,
        'scalings': torch.as_tensor(np.stack(ref_scaling)),
        'cameras': ref_camera,
        'pcd': ref_pcd,
        'zero_camera': ref_sequence_zero_camera
    }

    target_meta_data = {
        'image_rgb': all_target_image_rgb,
        'depth_maps': all_target_depth_maps,
        'fg_probability': all_target_fgprobs,
        'scalings': torch.as_tensor(np.stack(all_target_scalings)),
        'cameras': all_target_cameras,
        'pcd': target_pcd,
        'zero_camera': target_sequence_zero_camera
    }

    return (ref_image, all_target_images), (ref_transform, target_transform), (ref_meta_data, target_meta_data),\
           idx_into_dataset

def co3d_plotting_pose_dataset_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
    all_target_images, target_transform, all_target_depth_maps, \
    all_target_scalings, all_target_cameras, ref_pcd, target_pcd, \
    ref_fgprob, all_target_fgprobs, ref_sequence_zero_camera, target_sequence_zero_camera, \
    ref_image_rgb, all_target_image_rgb, item = zip(*batch)

    # Stack images
    ref_image = torch.stack(ref_image)
    all_target_images = torch.stack([torch.stack(x) for x in all_target_images])
    # Stack transforms
    ref_transform = torch.as_tensor(np.stack(ref_transform))
    target_transform = torch.as_tensor(np.stack(target_transform))
    # NB: don't stack depth_map, fg_probability, image_rgb as these are not
    # all the same size in the PlottingCo3DPoseDataset

    idx_into_dataset = torch.as_tensor(item)

    ref_meta_data = {
        'image_rgb': ref_image_rgb,
        'depth_maps': ref_depth_map,
        'fg_probability': ref_fgprob,
        'scalings': torch.as_tensor(np.stack(ref_scaling)),
        'cameras': ref_camera,
        'pcd': ref_pcd,
        'zero_camera': ref_sequence_zero_camera
    }

    target_meta_data = {
        'image_rgb': all_target_image_rgb,
        'depth_maps': all_target_depth_maps,
        'fg_probability': all_target_fgprobs,
        'scalings': torch.as_tensor(np.stack(all_target_scalings)),
        'cameras': all_target_cameras,
        'pcd': target_pcd,
        'zero_camera': target_sequence_zero_camera
    }

    return (ref_image, all_target_images), (ref_transform, target_transform), (ref_meta_data, target_meta_data),\
           idx_into_dataset

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    # Image processing
    image_norm_mean = (0.485, 0.456, 0.406)
    image_norm_std = (0.229, 0.224, 0.225)
    image_size = 224    # Image size
    image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_norm_mean, std=image_norm_std)
        ])
    categories = ["backpack", "bicycle", "book", "car", "chair", "hairdryer", "handbag",
              "hydrant", "keyboard", "laptop", "motorcycle", "mouse", "remote", 
              "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck"]
    all_samples = {}
    num_samples_per_class = 100
    num_frames_in_target_seq = 50
    determ_eval_root = f'/home/bras3856/Code/Pose/zero-shot-pose/data/determ_eval/200samples_{num_frames_in_target_seq}tgt'
    os.makedirs(determ_eval_root, exist_ok=True)
    for category in categories:
        print(category)
        determ_eval_path = os.path.join(determ_eval_root, f'{category}.pt')
        dataset = Co3DPoseDataset(dataset_root='/data/engs-robot-learning/bras3856/CO3D',
                            categories=[category],
                            num_samples_per_class=num_samples_per_class,
                            target_frames_sampling_mode='uniform',
                            num_frames_in_target_seq=num_frames_in_target_seq,
                            label_dir='/home/bras3856/Code/Pose/zero-shot-pose/data/class_labels',
                            determ_eval_root = determ_eval_root,
                            image_transform=image_transform)
        
        # batch_size = 8
        # dataloader = DataLoader(dataset, num_workers=8,
        #                 batch_size=batch_size, collate_fn=co3d_pose_dataset_collate, shuffle=False)

        # for batch in tqdm(dataloader):
        #     pass
            
        torch.save(dataset.samples, determ_eval_path)
