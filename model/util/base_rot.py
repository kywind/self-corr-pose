from absl import flags
import torch
import kornia
import numpy as np
from model.util.conversion import axisAngle_to_quat, quat_product


flags.DEFINE_list('base_rot', [1,0,0,0,1,0,0,0,1], 'scale offset, 3 by 3 matrix')

def get_base_rot(opts):
    # align the axes between our mesh prior and ground truth prior
    br = [float(x) for x in opts.base_rot]
    base_angles = torch.tensor([[br[0], br[1], br[2]],
                                [br[3], br[4], br[5]], 
                                [br[6], br[7], br[8]]], dtype=torch.float32)[None]
    base_angles = base_angles.cuda()
    return base_angles

def get_base_quaternions(num_pose_az=8, num_pose_el=1, initial_quat_bias_deg=45., elevation_bias=0, azimuth_bias=0):
    _axis = torch.eye(3).float()

    # Quaternion base bias
    xxx_base = [1.,0.,0.]
    aaa_base = initial_quat_bias_deg
    axis_base = torch.tensor(xxx_base).float()
    angle_base = torch.tensor(aaa_base).float() / 180. * np.pi
    qq_base = axisAngle_to_quat(axis_base, angle_base) # 4

    # Quaternion multipose bias
    azz = torch.as_tensor(np.linspace(0,2*np.pi,num=num_pose_az,endpoint=False)).float() + azimuth_bias * np.pi/180
    ell = torch.as_tensor(np.linspace(-np.pi/2,np.pi/2,num=(num_pose_el+1),endpoint=False)[1:]).float() + elevation_bias * np.pi/180
    quat_azz = axisAngle_to_quat(_axis[1], azz) # num_pose_az,4
    quat_ell = axisAngle_to_quat(_axis[0], ell) # num_pose_el,4
    quat_el_az = quat_product(quat_ell[None,:,:], quat_azz[:,None,:]) # num_pose_az,num_pose_el,4
    quat_el_az = quat_el_az.view(-1,4)                  # num_pose_az*num_pose_el,4
    _quat = quat_product(quat_el_az, qq_base[None,...]).float()

    return _quat

