import numpy as np
import torch
from pytorch3d.ops.points_alignment import iterative_closest_point
from pytorch3d.ops import utils as oputil
from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.structures.pointclouds import Pointclouds

from zsp.utils.pcd_proc import apply_transform_to_pcd
from zsp.datasets.co3d_tools.camera_utils import concatenate_cameras
from zsp.datasets.co3d_tools.point_cloud_utils import get_rgbd_point_cloud


def subsample_random_pcd(pcd, K=4000):
    X1, numpoints1 = oputil.convert_pointclouds_to_tensor(pcd)
    col1 = pcd.features_list()[0]
    perm = torch.randperm(X1.size(1))
    if len(perm) <= K:
        idx = perm
    else:
        idx = perm[:K]
    X1_sub = X1[:, idx, :]
    col1_sub = col1[idx][None, :, :]
    pcd1_sub = Pointclouds(X1_sub, col1_sub)
    return pcd1_sub


def subsample_and_ICP(pcd_ref, pcd_query, K=4000):
    pcd1_sub = subsample_random_pcd(pcd_ref, K=K)
    pcd2_sub = subsample_random_pcd(pcd_query, K=K)

    ICPSolution = iterative_closest_point(pcd1_sub, pcd2_sub, 
                                          estimate_scale=True,
                                          max_iterations=500,
                                          verbose=False
                                         )
    R_icp = ICPSolution.RTs.R # (B, 3, 3)
    t_icp = ICPSolution.RTs.T # (B, 3)
    T21w_icp = get_world_to_view_transform(R_icp.permute(0,2,1), t_icp)
    return T21w_icp


def transform_query_pcds(query_cameras, query_camera0, query_pcd):
    """Transforms pcds to make the view in cam0 match original view in original cam
    """
    # Initialise the list of transforms: 
    query_pcds_transforms = []
    T_c0 = query_camera0.get_world_to_view_transform()
    for cam1 in query_cameras:
        T_c1 = cam1.get_world_to_view_transform()
        # Rotation to apply to pcd 
        T10w_hat = T_c0.compose(T_c1.inverse())
        query_pcds_transforms.append(T10w_hat)

    query_pcds_transformed = []
    for T10w_hat in query_pcds_transforms:
        query_pcd_cam_i = apply_transform_to_pcd(
            query_pcd.clone(),
            T10w_hat
        )
        query_pcds_transformed.append(query_pcd_cam_i)
    
    return query_pcds_transforms, query_pcds_transformed


def transform_ref_pcd(ref_camera, ref_camera0, ref_pcd):
    ref_pcd_transforms, ref_pcd_transformed = transform_query_pcds(
        [ref_camera], ref_camera0, ref_pcd)

    return ref_pcd_transforms[0], ref_pcd_transformed[0]


def icp_ref2query(ref_camera, query_cameras, 
                  ref_camera0, query_camera0,
                  ref_pcd, query_pcd,
                  best_view_idx=None):
    """ Returns ICP as the average estimate over mu

    """
    print("Calling ICP")
    # Make a copy of the query pcd for each query image, transformed such that its image
    # in query_camera0 is the original image as seen through the original camera. 
    if not best_view_idx:
        best_view_idx = 0
    query_pcd_transform, query_pcd_transformed = transform_query_pcds(
        [query_cameras[best_view_idx]], query_camera0, query_pcd)
    query_pcd_transform = query_pcd_transform[0]
    query_pcd_transformed = query_pcd_transformed[0]
    # Make a copy of the reference pcd transformed such that its image
    # in ref_camera0 is the original image as seen through the original camera. 
    ref_pcd_transform, ref_pcd_transformed = transform_ref_pcd(
        ref_camera, ref_camera0, ref_pcd)

    Tref2query_icp = subsample_and_ICP(ref_pcd_transformed, query_pcd_transformed)
    # Transform the predicted transform so it can be compared to the ground-truth
    # (that is, undo the effects of transforming the ref and query pcds before ICP)
    T_hat_GT = query_pcd_transform.compose(Tref2query_icp).compose(ref_pcd_transform.inverse())
    return T_hat_GT


def icp_ref2query_bestframe(ref_camera, query_cameras, 
                            ref_camera0, query_camera0,
                            ref_pcd, query_pcd):
    return
