"""Functions for transforming pointclouds"""

from pytorch3d.structures.pointclouds import Pointclouds
from typing import Union
from pytorch3d.ops import utils as oputil
import torch
from pytorch3d.transforms.transform3d import Transform3d

def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Copied from "https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html
    
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X

def apply_similarity_transform(
    X: Union[torch.Tensor, Pointclouds], R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Copied from "https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html
    
    Wraps _apply_similarity_transform, handles Tensor/Pointcloud conversions
    """
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)

    # apply the init transform to the input point cloud
    Xt = _apply_similarity_transform(Xt, R, T, s)
    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt[0])  # type: ignore
        
    return Xt

def apply_transform_to_pcd(
    X: Pointclouds, tran: Transform3d
) -> torch.Tensor:
    """
    Copied from "https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html
    
    Wraps a transform, handles Tensor/Pointcloud conversions
    """
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)

    # apply the transform to the input point cloud
    Xt = tran.transform_points(Xt)
    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore
    return Xt

