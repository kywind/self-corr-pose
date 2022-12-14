import numpy as np
import torch
import torch.nn.functional as F

from zsp.utils.depthproc import cv2_depth_fillzero, NDCGridRaysampler
from zsp.method.rigid_body import least_squares_solution, umeyama

from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.transforms import so3_relative_angle
# from pytorch3d.transforms.transform3d import _check_valid_rotation_matrix

# ----------------
# NORMALIZED CYCLICAL DISTANCES
# ----------------
def normalize_cyclical_dists(dists):

    # Assume dists are in the format B x H x W
    b, h, w = dists.size()
    dists = dists.view(b, h * w)

    # Normalize to [0, 1]
    dists -= dists.min(dim=-1)[0][:, None]
    dists /= dists.max(dim=-1)[0][:, None]

    # Hack to find the mininimum non-negligible value
    dists -= 0.05
    dists[dists < 0] = 3

    # Re-Normalize to [0, 1]
    dists -= dists.min(dim=-1)[0][:, None]
    dists[dists > 1] = 0
    dists /= dists.max(dim=-1)[0][:, None]

    dists = dists.view(b, h, w)

    return dists


# ----------------
# FIND CLOSEST IMAGE BY GLOBAL FEATURE SIMILARITY
# ----------------
def rank_target_images_by_global_sim(ref_global_feats, target_global_feats):
    """
    Similarity from source image to target images with ViT global feature
    """
    b, n_tgt, d = target_global_feats.size()

    ref_global_feats = F.normalize(ref_global_feats, dim=-1)
    target_global_feats = F.normalize(target_global_feats, dim=-1)

    sim = torch.matmul(ref_global_feats.view(b, 1, d), target_global_feats.permute(0, 2, 1))

    return sim


# ----------------
# INTERSECTION OVER UNION
# ----------------
def batch_intersection_over_union(tensor_a, tensor_b, threshold=None):

    """
    a is B x H x W
    b is B x H x W
    """

    intersection = (tensor_a * tensor_b).sum(dim=(-1, -2))
    union = (tensor_a + tensor_b).sum(dim=(-1, -2))
    iou = 2 * (intersection / union)

    if threshold is None:

        return iou  # Return shape (B,)

    else:

        tensor_a[tensor_a < threshold] = 0
        tensor_b[tensor_b < threshold] = 0

        tensor_a[tensor_a >= threshold] = 1
        tensor_b[tensor_b >= threshold] = 1

        intersection_map = tensor_a * tensor_b

        return iou, intersection_map


# ----------------
# SCALE BACK POINTS
# ----------------
def scale_points_from_patch(points, vit_image_size=224, num_patches=28):
    points = (points + 0.5) / num_patches * vit_image_size
    
    return points


def scale_points_to_orig(points, image_scaling):
    points *= image_scaling

    return points.int().long()


# ---------------
# GET STRUCTURED POINT CLOUDS
# ---------------
def get_structured_pcd(frame, inpaint=True, world_coordinates=False):

    """Takes a frame from CO3D dataset
    Frame is not a frame object, but a dict with keys 'shape', 'camera', 'depth_map'


    Pointcloud returned is in world-coordinates, with shape (H, W, 3),
    with the 3 dimensions encoding X, Y and Z world coordinates.
    Optionally infills the depth map from the CO3D dataset, which tends
    to be ~50% zeros (the equivalent of NaNs in the .png encoding). This
    leads to fewer NaNs in the unprojected structured pointcloud.

    world_coordinates passed to cam.unproject_points: if it is false, the
    points are unprojected to the *camera* frame, else to the world frame
    """
    H, W = frame['shape']
    camera = frame['camera']
    depth_map = frame['depth_map']

    # --- 1. Make a grid of the coordinates in Pytorch3D NDC space ---
    gridsampler = NDCGridRaysampler(image_width=W, image_height=H, n_pts_per_ray=1, min_depth=0, max_depth=1)
    xy_grid = gridsampler._xy_grid

    # --- 2. In paint depth map ---
    if inpaint:
        depth_proc = torch.Tensor(cv2_depth_fillzero(depth_map.squeeze().numpy()))
    else:
        depth_proc = depth_map.squeeze()

    # --- 3. Stack the xy coords with depth, which is in NDC/screen format (no difference in this case, I think)
    xy_grid_ndc = torch.cat((xy_grid,
                             depth_proc.unsqueeze(-1)), dim=-1)
    xy_grid_ndc = xy_grid_ndc.view(-1, 3)

    # --- 4. Unproject ---
    unproj = camera.unproject_points(xy_grid_ndc, world_coordinates)    # (H*W, 3)

    # --- 5. Convert back to image shape, remove homogeneous dimension ---
    structured_pcd = unproj.view(H, W, 3)

    return structured_pcd


# ---------------
# RIGID BODY TRANSFORM
# ---------------
class RigidBodyTransform():

    def estimate(self, world_corr1, world_corr2):
        self.R, self.t, self.lam = least_squares_solution(world_corr1.T, world_corr2.T)

    def residuals(self, world_corr1, world_corr2):
        # E(x2_i) = λ*R*x1_i+t, {i = 1, ..., I}
        world_corr2_est = self.transform(world_corr1)
        res = torch.nn.PairwiseDistance(p=2)(torch.Tensor(world_corr2_est),
                                             torch.Tensor(world_corr2))
        return res.numpy()

    def transform(self, world_corr1):
        return (self.lam * self.R @ world_corr1.T + self.t).T


class RigidBodyUmeyama():
    
    def estimate(self, world_corr1, world_corr2):
        self.T, self.lam = umeyama(world_corr1, world_corr2)
        
    def residuals(self, world_corr1, world_corr2):
        world_corr2_est = self.transform(world_corr1)
        res = torch.nn.PairwiseDistance(p=2)(torch.Tensor(world_corr2_est),
                                             torch.Tensor(world_corr2))
        return res.numpy()
    
    def transform(self, world_corr1):
        w1_homo = np.vstack((world_corr1.T, np.ones((1, (len(world_corr1))))))
        transformed = self.T @ w1_homo
        return (transformed[:3, :]).T

# ---------------
# EVAL UTILS
# ---------------
def rotation_err(R_pred, R_gt):
    """compute rotation error for viewpoint estimation"""
    # compute the angle distance between rotation matrix in degrees
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err


def rotation_acc(R_err, th=30.):
    return 100. * torch.mean((R_err <= th).float())


def trans21_error(trans21, trans1_gt, trans2_gt, cam1, cam2):
    """Returns geodesic rotation error (degrees)

    Args:
        trans21: The camera-frame transform between im1 and im2
        trans1_gt: The world-frame transform between the pointcloud of im1
            and the pointcloud of im2
        trans2_gt: The world-frame transform between the pointcloud of im1
            and the pointcloud of im2
        cam1 (pytorch3d.renderer.cameras.PerspectiveCameras): the camera for
            im1, including the viewpoint (extrinsics) and intrinsics
        cam2 (pytorch3d.renderer.cameras.PerspectiveCameras): the camera for
            im2, including the viewpoint (extrinsics) and intrinsics
    """
    trans1 = get_world_to_view_transform(trans1_gt[:3, :3].t().unsqueeze(0),
                                         trans1_gt[:3, 3:].t())
    trans2 = get_world_to_view_transform(trans2_gt[:3, :3].t().unsqueeze(0),
                                         trans2_gt[:3, 3:].t())

    trans_gt = trans1.inverse().compose(trans2)
    w2v_cam1 = cam1.get_world_to_view_transform()
    w2v_cam2 = cam2.get_world_to_view_transform()

    trans21_gt = w2v_cam1.inverse().compose(trans_gt).compose(w2v_cam2)
    R21_pred = trans21.get_matrix().permute(0, 2, 1)[:, :3, :3]
    R21_gt = trans21_gt.get_matrix().permute(0, 2, 1)[:, :3, :3]
    umeyama_scale = torch.linalg.norm(R21_pred[:,0,:], 2, dim=-1)
    R21_pred /= umeyama_scale[:, None, None]
    # print(f"Scaled R21_pred by {umeyama_scale}, now testing R21_pred for valid rotation matrix")
    # _check_valid_rotation_matrix(R21_pred)
    # print("testing R21_gt for valid rotation matrix")
    # _check_valid_rotation_matrix(R21_gt)

    # if so3_relative_angle(R21_pred, R21_gt) * 180 / np.pi < 25:
    #     t21_pred = trans21.get_matrix().permute(0, 2, 1)[:, :3, 3:]
    #     t21_gt = trans21_gt.get_matrix().permute(0, 2, 1)[:, :3, 3:]
    #     import pdb
    #     pdb.set_trace()
    #     print(torch.nn.PairwiseDistance(p=2)(t21_pred.squeeze(), t21_gt.squeeze()))

    return so3_relative_angle(R21_pred, R21_gt) * 180 / np.pi


def trans_gt_error(trans21w_hat, trans1_gt, trans2_gt):
    """Returns geodesic rotation error (degrees)

    Args:
        trans21w_hat: The estimated world-frame transform between pcd1 and pcd2
        trans1_gt: The world-frame transform between the pointcloud of im1
            and the pointcloud of im2
        trans2_gt: The world-frame transform between the pointcloud of im1
            and the pointcloud of im2
    """
    
    trans1_gt = get_world_to_view_transform(trans1_gt[:3, :3].t().unsqueeze(0),
                                         trans1_gt[:3, 3:].t())
    trans2_gt = get_world_to_view_transform(trans2_gt[:3, :3].t().unsqueeze(0),
                                         trans2_gt[:3, 3:].t())

    trans21w_gt = trans1_gt.inverse().compose(trans2_gt)

    R21_pred = trans21w_hat.get_matrix().permute(0, 2, 1)[:, :3, :3]
    R21_gt = trans21w_gt.get_matrix().permute(0, 2, 1)[:, :3, :3]

    umeyama_scale = torch.linalg.norm(R21_pred[:,0,:], 2, dim=-1)
    R21_pred /= umeyama_scale[:, None, None]

    return so3_relative_angle(R21_pred, R21_gt) * 180 / np.pi
