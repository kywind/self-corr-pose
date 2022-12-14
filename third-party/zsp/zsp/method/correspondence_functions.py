import torch
import numpy as np
from sklearn.cluster import KMeans
from zsp.method.dense_descriptor_utils import extract_saliency_maps, _to_cartesian, chunk_cosine_sim

import torch.nn.functional as F

import time
# -----------------
# BATCHED VERSION OF CORRESPONDENCE FUNCTION
# Uses cyclical distances rather than mutual nearest neighbours
# -----------------
def find_correspondences_batch(descriptors1, descriptors2, attn1, attn2,
                               num_pairs: int = 10, thresh: float = 0.05, device : torch.device = torch.device('cpu')):
    """
    Finding point correspondences between two images.
    Legend: B: batch, T: total tokens (num_patches ** 2 + 1), D: Descriptor dim per head, H: Num attention heads

    Method: Compute similarity between all pairs of pixel descriptors
            Find nearest neighbours from Image1 --> Image2, and Image2 --> Image1
            Use nearest neighbours to define a cycle from Image1 --> Image2 --> Image1
            Take points in Image1 (and corresponding points in Image2) which have smallest 'cycle distance'
            Also, filter examples which aren't part of the foreground in both images, as determined by ViT attention maps

    :param descriptors1: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param descriptors2: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param attn1: ViT attention maps from final layer of shape B x H x T x T
    :param attn2: ViT attention maps from final layer of shape B x H x T x T
    :param num_pairs: number of outputted corresponding pairs.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    """
    # extracting descriptors for each image
    B, _, t_m_1, d_h = descriptors1.size()

    # Hard code
    num_patches1, load_size1 = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1))), 224
    inf_idx = int(t_m_1)

    # -----------------
    # EXTRACT SALIENCE MAPS
    # -----------------
    saliency_map1 = extract_saliency_maps(attn1)        # B x T - 1
    saliency_map2 = extract_saliency_maps(attn2)
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # -----------------
    # COMPUTE SIMILARITIES
    # calculate similarity between image1 and image2 descriptors
    # -----------------
    t1 = time.time()
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # -----------------
    # COMPUTE MUTUAL NEAREST NEIGHBOURS
    # -----------------
    sim_1, nn_1 = torch.max(similarities, dim=-1, keepdim=False)  # nn_1 - indices of block2 closest to block1. B x T - 1
    sim_2, nn_2 = torch.max(similarities, dim=-2, keepdim=False)  # nn_2 - indices of block1 closest to block2. B x T - 1
    nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]

    # Map nn_2 points which are not highlighed by fg_mask to 0
    nn_2[~fg_mask2] = 0     # TODO: Note, this assumes top left pixel is never a point of interest
    cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)  # Intuitively, nn_2[nn_1]

    # -----------------
    # COMPUTE SIMILARITIES
    # Find distance between cyclical point and original point in Image1
    # -----------------
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1])[None, :].repeat(B, 1)
    cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches1).to(device)
    image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches1).to(device)

    # Find which points are mapped to 0, artificially map them to a high value
    # TODO: tom: why the subtraction?
    zero_mask = (cyclical_idxs_ij - torch.Tensor([0, 0])[None, None, :].to(device)) == 0

    # TODO: tom: is the inf_idx value correct?
    cyclical_idxs_ij[zero_mask] = inf_idx

    # Find negative of distance between cyclical point and original point
    # View to make sure PairwiseDistance behaviour is consistent across torch versions
    b, hw, ij_dim = cyclical_idxs_ij.size()
    cyclical_dists = -torch.nn.PairwiseDistance(p=2)(cyclical_idxs_ij.view(-1, ij_dim), image_idxs_ij.view(-1, ij_dim))
    cyclical_dists = cyclical_dists.view(b, hw)

    # TODO: tom: why to normalize?
    cyclical_dists_norm = cyclical_dists - cyclical_dists.min(1, keepdim=True)[0]        # Normalize to [0, 1]
    cyclical_dists_norm /= cyclical_dists_norm.max(1, keepdim=True)[0]

    # -----------------
    # Further mask pixel locations in Image1 which are not highlighted by FG mask
    # -----------------
    cyclical_dists_norm *= fg_mask1.float()

    # -----------------
    # Find the TopK points in Image1 and their correspondences in Image2
    # -----------------
    sorted_vals, selected_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)
    selected_points_image_1 = selected_points_image_1[:, :num_pairs]

    # Get corresponding points in image 2
    selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)

    # -----------------
    # Compute the distances of the selected points
    # -----------------
    sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=selected_points_image_1.to(device))

    # Convert to cartesian coordinates
    selected_points_image_1, selected_points_image_2 = (_to_cartesian(inds, shape=num_patches1) for inds in
                                                        (selected_points_image_1, selected_points_image_2))

    cyclical_dists = cyclical_dists.reshape(-1, num_patches1[0], num_patches1[1])

    return selected_points_image_1, selected_points_image_2, cyclical_dists, sim_selected_12


#-----------------
# CORRESPONDENCE FUNCTION FROM: https://github.com/ShirAmir/dino-vit-features
# -----------------
def find_correspondences_original(descriptors1, descriptors2, attn1, attn2,
                         num_pairs: int = 10, thresh: float = 0.05, patch_size: int = 16,
                         stride: int = 8, device : torch.device = torch.device('cpu')):
    """
    finding point correspondences between two images.
    Legend: B: batch, T: total tokens (num_patches ** 2 + 1), D: Descriptor dim per head, H: Num attention heads
    :param descriptors1: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param descriptors2: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param attn1: ViT attention maps from final layer of shape B x H x T x T
    :param attn2: ViT attention maps from final layer of shape B x H x T x T
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """
    # extracting descriptors for each image
    B, _, t_m_1, d_h = descriptors1.size()

    # Hard code
    num_patches1, load_size1 = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1))), 224
    num_patches2, load_size2 = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1))), 224

    # extracting saliency maps for each image
    saliency_map1 = extract_saliency_maps(attn1)[0]
    saliency_map2 = extract_saliency_maps(attn2)[0]

    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length

    if len(normalized) == 0:
        return [], []

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    sim_selected_12 = sim_1[img1_indices_to_show]

    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * stride + stride + patch_size // 2
        y1_show = (int(y1) - 1) * stride + stride + patch_size // 2
        x2_show = (int(x2) - 1) * stride + stride + patch_size // 2
        y2_show = (int(y2) - 1) * stride + stride + patch_size // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))
    points1 = torch.Tensor(points1)
    points2 = torch.Tensor(points2)

    return points1, points2, sim_selected_12


# -----------------
# BATCHED VERSION OF CORRESPONDENCE FUNCTION WITH KNN
# Same as 'find_correspondences_batch' but uses KNN to find well distributed features
# -----------------
def find_correspondences_batch_with_knn(descriptors1, descriptors2, attn1, attn2,
                               num_pairs_for_topk: int = 10, thresh: float = 0.05, high_res=False,
                                        device : torch.device = torch.device('cpu')):
    """
    Finding point correspondences between two images.
    Legend: B: batch, T: total tokens (num_patches ** 2 + 1), D: Descriptor dim per head, H: Num attention heads

    Method: Compute similarity between all pairs of pixel descriptors
            Find nearest neighbours from Image1 --> Image2, and Image2 --> Image1
            Use nearest neighbours to define a cycle from Image1 --> Image2 --> Image1
            Take points in Image1 (and corresponding points in Image2) which have smallest 'cycle distance'
            Also, filter examples which aren't part of the foreground in both images, as determined by ViT attention maps

    :param descriptors1: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param descriptors2: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param attn1: ViT attention maps from final layer of shape B x H x T x T
    :param attn2: ViT attention maps from final layer of shape B x H x T x T
    :param num_pairs: number of outputted corresponding pairs.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    """
    # extracting descriptors for each image
    B, _, t_m_1, d_h = descriptors1.size()

    # Hard code
    num_patches1, load_size1 = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1))), 224
    inf_idx = int(t_m_1)

    # -----------------
    # EXTRACT SALIENCE MAPS
    # -----------------
    saliency_map1 = extract_saliency_maps(attn1)        # B x T - 1
    saliency_map2 = extract_saliency_maps(attn2)

    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # -----------------
    # COMPUTE SIMILARITIES
    # calculate similarity between image1 and image2 descriptors
    # -----------------
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # -----------------
    # COMPUTE MUTUAL NEAREST NEIGHBOURS
    # -----------------
    sim_1, nn_1 = torch.max(similarities, dim=-1, keepdim=False)  # nn_1 - indices of block2 closest to block1. B x T - 1
    sim_2, nn_2 = torch.max(similarities, dim=-2, keepdim=False)  # nn_2 - indices of block1 closest to block2. B x T - 1
    nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]

    # Map nn_2 points which are not highlighed by fg_mask to 0
    nn_2[~fg_mask2] = 0     # TODO: Note, this assumes top left pixel is never a point of interest
    cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)  # Intuitively, nn_2[nn_1]

    # -----------------
    # COMPUTE SIMILARITIES
    # Find distance between cyclical point and original point in Image1
    # -----------------
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1])[None, :].repeat(B, 1)
    cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches1).to(device)
    image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches1).to(device)

    # Find which points are mapped to 0, artificially map them to a high value
    zero_mask = (cyclical_idxs_ij - torch.Tensor([0, 0])[None, None, :].to(device)) == 0
    cyclical_idxs_ij[zero_mask] = inf_idx

    # Find negative of distance between cyclical point and original point
    # View to make sure PairwiseDistance behaviour is consistent across torch versions
    b, hw, ij_dim = cyclical_idxs_ij.size()
    cyclical_dists = -torch.nn.PairwiseDistance(p=2)(cyclical_idxs_ij.view(-1, ij_dim), image_idxs_ij.view(-1, ij_dim))
    cyclical_dists = cyclical_dists.view(b, hw)

    cyclical_dists_norm = cyclical_dists - cyclical_dists.min(1, keepdim=True)[0]        # Normalize to [0, 1]
    cyclical_dists_norm /= cyclical_dists_norm.max(1, keepdim=True)[0]

    # -----------------
    # Further mask pixel locations in Image1 which are not highlighted by FG mask
    # -----------------
    cyclical_dists_norm *= fg_mask1.float()

    # -----------------
    # Find the TopK points in Image1 and their correspondences in Image2
    # -----------------
    sorted_vals, topk_candidate_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)
    topk_candidate_points_image_1 = topk_candidate_points_image_1[:, :num_pairs_for_topk]

    # -----------------
    # Now do K-Means clustering on the descriptors in image 1 to choose well distributed features
    # -----------------
    if high_res:
        num_pairs_to_return = num_pairs_for_topk // 8
    else:
        num_pairs_to_return = num_pairs_for_topk // 2
    selected_points_image_1 = []
    for b in range(B):

        idxs_b = topk_candidate_points_image_1[b]
        feats_b = descriptors1[b][0, :, :][idxs_b]      # num_pairs_for_topk x D * H
        feats_b = F.normalize(feats_b, dim=-1).cpu().numpy()
        salience_b = saliency_map1[b][idxs_b]           # num_pairs_for_topk

        kmeans = KMeans(n_clusters=num_pairs_to_return, random_state=0).fit(feats_b)
        kmeans_labels = torch.as_tensor(kmeans.labels_).to(device)

        final_idxs_chosen_from_image_1_b = []
        for k in range(num_pairs_to_return):

            locations_in_cluster_k = torch.where(kmeans_labels == k)[0]
            saliencies_at_k = salience_b[locations_in_cluster_k]
            point_chosen_from_cluster_k = saliencies_at_k.argmax()
            final_idxs_chosen_from_image_1_b.append(idxs_b[locations_in_cluster_k][point_chosen_from_cluster_k])

        final_idxs_chosen_from_image_1_b = torch.stack(final_idxs_chosen_from_image_1_b)
        selected_points_image_1.append(final_idxs_chosen_from_image_1_b)

    selected_points_image_1 = torch.stack(selected_points_image_1)

    # Get corresponding points in image 2
    selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)

    # -----------------
    # Compute the distances of the selected points
    # -----------------
    sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=selected_points_image_1.to(device))

    # Convert to cartesian coordinates
    selected_points_image_1, selected_points_image_2 = (_to_cartesian(inds, shape=num_patches1) for inds in
                                                        (selected_points_image_1, selected_points_image_2))

    cyclical_dists = cyclical_dists.reshape(-1, num_patches1[0], num_patches1[1])

    return selected_points_image_1, selected_points_image_2, cyclical_dists, sim_selected_12