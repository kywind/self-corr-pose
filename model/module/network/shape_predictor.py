from absl import flags
import torch
import torch.nn as nn

from nerf import models, run_network
from nerf.nerf_helpers import positional_encoding


flags.DEFINE_bool('no_deform', False, 'do not predict deformation')
flags.DEFINE_float('deform_ratio', 1., 'deform ratio')

class ShapePredictor(nn.Module):
    def __init__(self, opts):
        super(ShapePredictor, self).__init__()
        self.shapenerf = models.CondNeRFModel(
            num_layers=2,
            num_encoding_fn_xyz=0,
            num_encoding_fn_dir=0,
            include_input_xyz=True,
            include_input_dir=False,
            use_viewdirs=False,
            out_channel=3,
            codesize=opts.codedim
        )
        self.iters = 0
        self.no_deform = opts.no_deform
        self.deform_ratio = opts.deform_ratio
    
    def forward(self, mean_v, shape_code):
        if self.no_deform:
            return mean_v
        else:
            shape_delta = run_network(
                self.shapenerf, 
                mean_v.detach(), 
                None, 
                131072, 
                None, 
                None,
                code=shape_code
            )[:,:,:-1]
            shape_delta -= shape_delta.mean(1, keepdims=True)
            pred_v = mean_v + shape_delta * self.deform_ratio
            return pred_v