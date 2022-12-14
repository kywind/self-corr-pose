import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines


image_norm_mean = (0.485, 0.456, 0.406)
image_norm_std = (0.229, 0.224, 0.225)
image_size = 224    # Image size

def denorm_torch_to_pil(image):
    image = image * torch.Tensor(image_norm_std)[:, None, None]
    image = image + torch.Tensor(image_norm_mean)[:, None, None]
    return Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))


def fig_to_pil(fig):
    tmp_file = tempfile.SpooledTemporaryFile(max_size=10*1024*1024) # 10MB 
    fig.savefig(tmp_file, bbox_inches='tight', pad_inches=0)
    image = Image.open(tmp_file)
    image.load()
    tmp_file.close()
    return image


def fig_to_array(fig):
    return np.array(fig_to_pil(fig))


def arrange(images):
    rows = []
    for row in images:
        rows += [np.concatenate(row, axis=1)]
    image = np.concatenate(rows, axis=0)
    return image


def co3d_rgb_to_pil(image_rgb):
    return Image.fromarray((image_rgb.permute(1,2,0)*255).numpy().astype(np.uint8))



def tile_ims_horizontal_highlight_best(ims, gap_px=20, highlight_idx=None):
    cumul_offsets = [0]
    for im in ims:
        cumul_offsets.append(cumul_offsets[-1]+im.width+gap_px)
    max_h = max([im.height for im in ims])
    dst = Image.new('RGB', (cumul_offsets[-1], max_h), (255, 255, 255))
    for i, im in enumerate(ims):
        dst.paste(im, (cumul_offsets[i], (max_h - im.height) // 2))
        
        if i == highlight_idx:
            img1 = ImageDraw.Draw(dst)  
            # shape is defined as [(x1,y1), (x2, y2)]
            shape = [(cumul_offsets[i],(max_h - im.height) // 2), 
                     (cumul_offsets[i]+im.width, max_h-(max_h - im.height) // 2)]
            img1.rectangle(shape, fill = None, outline ="green", width=6)

    return dst


def plot_pcd_and_ims(im1, im2, pcd1, pcd2, P_im, error, axs, pcd_frame=1):
    # Sort the resulting points by their (NDC) depth, from high to low depth
    _, indices = torch.sort(P_im[:,2], 0, descending=False)
    P_im = P_im[indices]
    if pcd_frame == 1:
        colors = pcd1.features_list()[0][indices].numpy()
    elif pcd_frame == 2:
        colors = pcd2.features_list()[0][indices].numpy()
    N_subsample = 100000
    if len(P_im) > N_subsample:
        subsample = np.random.choice(np.arange(len(P_im)), N_subsample, replace=False)
        colors = colors[subsample]
        P_im = P_im[subsample]

    fig, ax = plt.subplots(1,3, figsize=(10,3))
    ax[0].imshow(im1)
    ax[0].set_title('Original image1')
    ax[0].axis('off')
    ax[1].imshow(im2)
    ax[1].set_title('Original image2')
    ax[1].axis('off')
    ax[2].scatter(P_im[:, 0], -P_im[:, 1],
                c = colors[:], s=3)
    ax[2].axis('equal')
    # ax[2].axis('off')
    ax[2].set_title(f'im1 pcd in im2 pose - {error} deg error')


def plot_pcd(P_im, pcd, ax, N_subsample=100000, s=3):
    # Sort the resulting points by their (NDC) depth, from high to low depth
    _, indices = torch.sort(P_im[:,2], 0, descending=False)
    P_im = P_im[indices]
    colors = pcd.features_list()[0][indices].numpy()
    if len(P_im) > N_subsample:
        subsample = np.random.choice(np.arange(len(P_im)), N_subsample, replace=False)
        colors = colors[subsample]
        P_im = P_im[subsample]
    ax.scatter(P_im[:, 0], -P_im[:, 1],
                c = colors[:], s=s)
    ax.axis('equal')
    return ax


def get_concat_h_cut_center(im1, im2, gap_px=20):
    dst = Image.new('RGB', (im1.width + im2.width + gap_px, min(im1.height, im2.height)), (255, 255, 255))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + gap_px, (im1.height - im2.height) // 2))
    return dst


def draw_correspondences_lines(points1, points2, image1, image2, ax):
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :param ax: a matplotlib axis object
    :return: the matplotlib axis.
    """
    gap = 20

    im = np.array(get_concat_h_cut_center(image1, image2, gap))
    ax.imshow(im)
    ax.axis('off')
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                       "maroon", "black", "white", "chocolate", "gray", "blueviolet"]*(1+num_points//15))
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 6, 2
    points2 += np.array([0, gap+image1.width])
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax.add_patch(circ1_1)
        ax.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax.add_patch(circ2_1)
        ax.add_patch(circ2_2)
        l = mlines.Line2D([x1,x2], [y1,y2], c=color, linewidth=0.75)
        ax.add_line(l)
        ax.plot(x1, y1, x2, y2, linestyle='-', c='w')
    return ax

