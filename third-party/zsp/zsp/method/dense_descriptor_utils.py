"""
Several functions in this file are either taken from or inspired by the
code from Shir Amir at https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py,
which is an implementation for the paper "Deep ViT Features as Dense Visual Descriptors"
(Amir et al 2021).
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple

import torchvision

def _to_cartesian(coords: torch.Tensor, shape: Tuple):

    """
    Takes raveled coordinates and returns them in a cartesian coordinate frame
    coords: B x D
    shape: tuple of cartesian dimensions
    return: B x D x 2
    """

    i, j = (torch.from_numpy(inds) for inds in np.unravel_index(coords.cpu(), shape=shape))
    return torch.stack([i, j], dim=-1)


def draw_correspondences(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                         image1: Image.Image, image2: Image.Image) -> Tuple[plt.Figure, plt.Figure]:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    fig2, ax2 = plt.subplots()
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 8, 1
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    return fig1, fig2


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def extract_saliency_maps(attn_maps: torch.Tensor, head_idxs=[0, 2, 4, 5]) -> torch.Tensor:
    """
    extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
    in of the CLS token. All values are then normalized to range between 0 and 1.
    :param attn_maps: attention maps of the last layer. B x H x T x T
    :return: a tensor of saliency maps. has shape Bxt-1
    """
    curr_feats = attn_maps      #B x h x t x t
    cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #B x (t-1)
    temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0][:, None], cls_attn_map.max(dim=1)[0][:, None]
    cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
    return cls_attn_maps


def gaussian_blurring(features, kernel_size=9, sigma=3):

    """
    Assume features in shape B x N x D
    """

    b, n_patches, d = features.size()
    n_patches_h = np.sqrt(n_patches).astype('int')

    blur_kernel = torchvision.transforms.GaussianBlur(kernel_size, sigma=(sigma, sigma))

    features = features.reshape(b, n_patches_h, n_patches_h, d).permute(0, 3, 1, 2)
    features = blur_kernel(features)
    features = features.permute(0, 2, 3, 1).reshape(b, n_patches_h * n_patches_h, d)

    return features


def _log_bin(x: torch.Tensor, hierarchy: int = 2, device: str = 'cpu') -> torch.Tensor:
    """
    create a log-binned descriptor.
    :param x: tensor of features. Has shape Bx1xtx(dxh).
    :param hierarchy: how many bin hierarchies to use.
    """
    B = x.shape[0]
    if x.shape[1] != 1:
        raise ValueError('log_bin function now expects features reshaped to Bx1xtx(dxh), not Bxhxtxd')
    num_patches = int(np.sqrt(x.shape[2]))
    num_bins = 1 + 8 * hierarchy

    # bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
    bin_x = x.squeeze(1)  # Bx(t-1)x(dxh)
    bin_x = bin_x.permute(0, 2, 1) # Bx(dxh)x(t-1)
    bin_x = bin_x.reshape(B, bin_x.shape[1], num_patches, num_patches)
    # Bx(dxh)xnum_patches[0]xnum_patches[1]
    sub_desc_dim = bin_x.shape[1]

    avg_pools = []
    # compute bins of all sizes for all spatial locations.
    for k in range(0, hierarchy):
        # avg pooling with kernel 3**kx3**k
        win_size = 3 ** k
        avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
        avg_pools.append(avg_pool(bin_x))

    bin_x = torch.zeros((B, sub_desc_dim * num_bins, num_patches, num_patches)).to(device)
    for y in range(num_patches):
        for x in range(num_patches):
            part_idx = 0
            # fill all bins for a spatial location (y, x)
            for k in range(0, hierarchy):
                kernel_size = 3 ** k
                for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                    for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                        if i == y and j == x and k != 0:
                            continue
                        if 0 <= i < num_patches and 0 <= j < num_patches:
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                     :, :, i, j]
                        else:  # handle padding in a more delicate way than zero padding
                            temp_i = max(0, min(i, num_patches - 1))
                            temp_j = max(0, min(j, num_patches - 1))
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                     :, :, temp_i,
                                                                                                     temp_j]
                        part_idx += 1
    bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
    # Bx1x(t-1)x(dxh)
    return bin_x
