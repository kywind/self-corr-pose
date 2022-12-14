from __future__ import absolute_import, division, print_function

import numpy as np
import torch

def reflect_cam_pose(cam_pose):
    batch_dims = cam_pose.dim()-1
    cam_pose = cam_pose * torch.tensor([1, -1, 1, 1, 1, -1, -1],
                                        dtype=cam_pose.dtype,
                                        device=cam_pose.device).view((1,)*batch_dims + (-1,))
    return cam_pose

def reflect_cam_pose_z(cam_pose):
    batch_dims = cam_pose.dim()-1
    axis = torch.tensor([[0, 1, 0]], dtype=cam_pose.dtype, device=cam_pose.device)
    angle = torch.tensor([np.pi], dtype=cam_pose.dtype, device=cam_pose.device)
    rot180 = axisangle2quat(axis, angle).view((1,)*batch_dims + (-1,))
    quat = hamilton_product(rot180, cam_pose[..., 3:7])
    quat = quat * torch.tensor([1, 1, -1, -1],
                                    dtype=quat.dtype,
                                    device=quat.device).view((1,)*batch_dims + (-1,))
    cam_pose = torch.cat((cam_pose[...,:3], quat), dim=-1)
    return cam_pose