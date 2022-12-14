import torch
import numpy as np
from model.util.quaternion import quat_product

def azElRot_to_quat(azElRot):
    """
    azElRot: ...,az el ro
    """
    _axis = torch.eye(3, dtype=azElRot.dtype, device=azElRot.device)
    num_dims = azElRot.dim()-1
    _axis = _axis.view((1,)*num_dims+(3,3))
    azz = azElRot[..., 0]
    ell = azElRot[..., 1]
    rot = azElRot[..., 2]
    quat_azz = axisAngle_to_quat(_axis[...,1], azz) # ...,4
    quat_ell = axisAngle_to_quat(_axis[...,0], ell) # ...,4
    quat_rot = axisAngle_to_quat(_axis[...,2], rot) # ...,4
    quat = quat_product(quat_ell, quat_azz)
    quat = quat_product(quat_rot, quat)
    return quat

def quat_to_axisAngle(quat):
    """
    quat: B x 4: [quaternions]
    returns quaternion axis, angle
    """
    cos = quat[..., 0]
    sin = quat[..., 1:].norm(dim=-1)
    axis = quat[..., 1:]/sin[..., None]
    angle = 2*cos.clamp(-1+1e-6,1-1e-6).acos()
    return axis, angle

def axisAngle_to_quat(axis, angle):
    """
    axis: B x 3: [axis]
    angle: B: [angle]
    returns quaternion: B x 4
    """
    axis = torch.nn.functional.normalize(axis,dim=-1)
    angle = angle.unsqueeze(-1)/2
    quat = torch.cat([angle.cos(), angle.sin()*axis], dim=-1)
    return quat

def xyz_to_uv(verts):
    """
    X : N,3
    Returns UV: N,2 normalized to [-1, 1]
    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    """
    eps = 1e-4
    rad = torch.norm(verts, dim=-1).clamp(min=eps)
    theta = torch.acos((verts[..., 1] / rad).clamp(min=-1+eps,max=1-eps))    # Inclination: Angle with +Z [0,pi]
    phi = torch.atan2(verts[..., 2], verts[..., 0])  # Azimuth: Angle with +X [-pi,pi]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    uv = torch.stack([uu, vv], dim=-1)
    return uv

def uv_to_xyz(uv, rad=1):
    '''
    Takes a uv coordinate between [-1,1] and returns a 3d point on the sphere.
    uv -- > [......, 2] shape

    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    '''
    phi = np.pi * uv[...,0]
    theta = np.pi * (uv[...,1] + 1) / 2

    if type(uv) == torch.Tensor:
        x = torch.sin(theta) * torch.cos(phi)
        z = torch.sin(theta) * torch.sin(phi)
        y = torch.cos(theta)
        points3d = torch.stack([x,y,z], dim=-1)
    else:
        x = np.sin(theta) * np.cos(phi)
        z = np.sin(theta) * np.sin(phi)
        y = np.cos(theta)
        points3d = np.stack([x,y,z], axis=-1)
    return points3d * rad


