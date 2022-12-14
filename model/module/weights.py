from absl import flags
import numpy as np


flags.DEFINE_float('mask_wt', 0.1, 'weight of loss')
flags.DEFINE_float('tex_wt', 0.05, 'weight of loss')
flags.DEFINE_float('depth_wt', 0.05, 'weight of loss')
flags.DEFINE_float('match_wt', 0.01, 'weight of loss')
flags.DEFINE_float('imatch_wt', 0.02, 'weight of loss')
flags.DEFINE_float('triangle_wt', 0.001, 'weight of loss')
flags.DEFINE_float('pullfar_wt', 0.001, 'weight of loss')
flags.DEFINE_float('deform_wt', 0.05, 'weight of loss')
flags.DEFINE_float('symmetry_wt', 1., 'weight of loss')
flags.DEFINE_float('camera_wt', 0.005, 'weight of loss')
flags.DEFINE_float('cycle_loss_wt', 0.2, '')
flags.DEFINE_float('cycle_loss_pretrain_wt', 0.05, '')
flags.DEFINE_float('decay_ratio', 1., '')



def reg_decay(curr_steps, max_steps, min_wt, max_wt, mode='linear'):
    if curr_steps > max_steps: current = min_wt
    elif mode == 'log':
        current = np.exp(curr_steps / float(max_steps) * (np.log(min_wt) - np.log(max_wt))) * max_wt 
    elif mode == 'linear':
        current = curr_steps / float(max_steps) * (min_wt - max_wt) + max_wt
    else:
        raise NotImplementedError
    return current


class Weights:
    def __init__(self, opts):
        self.opts = opts
        self.total_iters = opts.total_iters

        ## basic
        self.mask_wt = opts.mask_wt
        self.depth_wt = opts.depth_wt
        self.tex_wt = opts.tex_wt
        self.match_wt = opts.match_wt
        self.imatch_wt = opts.imatch_wt

        ## regularization
        self.triangle_wt = opts.triangle_wt
        self.pullfar_wt = opts.pullfar_wt
        self.deform_wt = opts.deform_wt
        self.symmetry_wt = opts.symmetry_wt
        self.camera_wt  = opts.camera_wt

        self.cycle_loss_wt = opts.cycle_loss_wt
        self.cycle_loss_pt_wt = opts.cycle_loss_pretrain_wt


    def schedule(self, iter):
        ## decreasing
        self.triangle_wt = reg_decay(iter, self.total_iters, self.opts.decay_ratio * self.opts.triangle_wt, self.opts.triangle_wt)
        self.symmetry_wt = reg_decay(iter, self.total_iters, self.opts.decay_ratio * self.opts.symmetry_wt, self.opts.symmetry_wt)
        self.cycle_loss_wt = reg_decay(iter, self.total_iters, self.opts.decay_ratio * self.opts.cycle_loss_wt, self.opts.cycle_loss_wt)
        self.cycle_loss_pt_wt = reg_decay(iter, self.total_iters, self.opts.decay_ratio * self.opts.cycle_loss_pretrain_wt, self.opts.cycle_loss_pretrain_wt)

        ## increasing
        self.match_wt = reg_decay(iter, self.total_iters, self.opts.match_wt, self.opts.decay_ratio * self.opts.match_wt)
        self.imatch_wt = reg_decay(iter, self.total_iters, self.opts.imatch_wt, self.opts.decay_ratio * self.opts.imatch_wt)

