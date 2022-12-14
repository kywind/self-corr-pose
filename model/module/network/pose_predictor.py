from absl import flags
import torch
import torch.nn as nn
import kornia

from model.module.network.net_blocks import fc_stack
from model.util.base_rot import get_base_quaternions 
from model.util.quaternion import QuaternionCoeffOrder


flags.DEFINE_bool('use_scale', False, 'use scale estimation')
flags.DEFINE_list('rotation_offset', [0,0,0,0,0,0], 'a small offset to stablize rotation in early stage of training')
flags.DEFINE_float('depth_offset', 10., 'offset of depth estimation')

flags.DEFINE_float('initial_quat_bias_deg', 0, 'Rotation bias in deg. 90 for head-view, 45 for breast-view')
flags.DEFINE_float('baseQuat_elevationBias', 0, 'Increase elevation by this angle')
flags.DEFINE_float('baseQuat_azimuthBias', 0, 'Increase azimuth by this angle')
flags.DEFINE_integer('num_multipose_az', 1, 'Number of camera pose hypothesis bins (along azimuth)')
flags.DEFINE_integer('num_multipose_el', 1, 'Number of camera pose hypothesis bins (along elevation)')


class PosePredictor(nn.Module):
    def __init__(self, opts, nc_input):
        super(PosePredictor, self).__init__()
        self.opts = opts
        self.offset = opts.depth_offset
        self.use_scale = opts.use_scale
        self.symmetry = opts.symmetry_idx

        self.base_rots = self.init_rot()
        self.n_hypo = self.base_rots.shape[0]
        assert self.n_hypo == 1
        self.nc_input = nc_input

        nc_hidden = 128
        self.rot_pred_layer = nn.Sequential(
            fc_stack(self.nc_input, nc_hidden, 3, use_bn=False),
            nn.Linear(nc_hidden, 6*self.n_hypo)
        )
        self.trans_pred_layer = nn.Linear(self.nc_input, 3*self.n_hypo)
        if self.use_scale: self.scale_pred_layer = nn.Linear(self.nc_input, 3*self.n_hypo)
        
        r_off = [float(r) for r in opts.rotation_offset]
        self.x_offset = nn.Parameter(torch.tensor([r_off[:3]]), requires_grad=False)
        self.y_offset = nn.Parameter(torch.tensor([r_off[3:]]), requires_grad=False)
    
    def init_rot(self):
        self.n_hypo = self.opts.num_multipose_az * self.opts.num_multipose_el
        base_quats = get_base_quaternions(num_pose_az=self.opts.num_multipose_az,
                                          num_pose_el=self.opts.num_multipose_el,
                                          initial_quat_bias_deg=self.opts.initial_quat_bias_deg,
                                          elevation_bias=self.opts.baseQuat_elevationBias,
                                          azimuth_bias=self.opts.baseQuat_azimuthBias).cuda()
        return kornia.geometry.quaternion_to_rotation_matrix(base_quats, \
                            order=QuaternionCoeffOrder.WXYZ)  # (n_hypo, 3, 3)

    def forward(self, feat):
        bsz = feat.shape[0]
        rot = self.rot_pred_layer.forward(feat).reshape(bsz*self.n_hypo, 6)
        trans = self.trans_pred_layer.forward(feat).reshape(bsz*self.n_hypo, 3)

        rot = rot.reshape(-1, 6)
        x = rot[:, :3]
        y = rot[:, 3:6]
        
        x += self.x_offset
        y += self.y_offset

        x = torch.nn.functional.normalize(x)
        z = torch.cross(x, y)
        z = torch.nn.functional.normalize(z)
        y = torch.cross(z, x)
        y = torch.nn.functional.normalize(y)
        rot = torch.stack((x,y,z), 2)

        trans[:, :2] = trans[:, :2] * 0.1
        trans[:, 2] = trans[:, 2] + self.offset

        if self.use_scale:
            scale = self.scale_pred_layer.forward(feat).reshape(bsz*self.n_hypo, 3)
            scale = scale * 0.1 + 1.
        else:
            scale = torch.ones((bsz*self.n_hypo, 3)).cuda()
        return rot, trans, scale



