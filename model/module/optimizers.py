from absl import flags
import torch


class Optimizers:
    def __init__(self, opts, model):
        self.opts = opts
        self.model = model
        self.total_steps = opts.total_iters * opts.ngpu

        vert_params = []
        cam_params = []
        shape_params = []
        feat_params = []
        backbone_params = []
        
        for name, p in self.model.named_parameters():
            if 'mean_v' in name: 
                print('found vert_params: {}, shape: {}, requires_grad: {}'.format(name, list(p.shape), p.requires_grad))
                vert_params.append(p)
            elif 'pose_predictor' in name:
                print('found cam_params: {}, shape: {}, requires_grad: {}'.format(name, list(p.shape), p.requires_grad))
                cam_params.append(p)
            elif 'shape_predictor' in name or 'shape_code_predictor' in name:
                print('found shape_params: {}, shape: {}, requires_grad: {}'.format(name, list(p.shape), p.requires_grad))
                shape_params.append(p)
            elif 'featnet' in name:
                print('found feat_params: {}, shape: {}, requires_grad: {}'.format(name, list(p.shape), p.requires_grad))
                feat_params.append(p)
            elif 'backbone' in name:
                print('found backbone_params: {}, shape: {}, requires_grad: {}'.format(name, list(p.shape), p.requires_grad))
                backbone_params.append(p)
            elif 'pretrain_corr_net' in name:
                continue
            else:
                print('*found unknown params: {}, shape: {}, requires_grad: {}'.format(name, list(p.shape), p.requires_grad))

        self.optimizer = torch.optim.AdamW(
            [
                {'params': vert_params},
                {'params': cam_params},
                {'params': shape_params},
                {'params': feat_params},
                {'params': backbone_params},
            ],
            lr = self.opts.learning_rate,
            betas = (0.9, 0.999),
            weight_decay = 1e-4
        )
        vert_lr = self.opts.vert_lr_ratio * self.opts.learning_rate
        cam_lr = self.opts.cam_lr_ratio * self.opts.learning_rate
        shape_lr = 1 * self.opts.learning_rate
        feat_lr = 1 * self.opts.learning_rate
        backbone_lr = 1 * self.opts.learning_rate
        pct_start = 0.05
        div_factor = 25
        final_div_factor = 25

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            [
                vert_lr,
                cam_lr,
                shape_lr,
                feat_lr,
                backbone_lr,
            ],
            total_steps = self.total_steps,
            pct_start = pct_start, 
            cycle_momentum = False, 
            anneal_strategy = 'cos',
            final_div_factor = final_div_factor, 
            div_factor = div_factor
        )
    

    def step(self, iter):
        self.optimizer.step()  # update parameters
        self.scheduler.step()


    def zero_grad(self):
        self.optimizer.zero_grad()

