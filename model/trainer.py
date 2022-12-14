from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import sys
sys.path.insert(0,'third-party')

import time
import pdb
import numpy as np
from absl import flags
import time
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import data_loader
from model.model import MeshNet
from model.module.optimizers import Optimizers


def add_image(log, tag, img, step, scale=True):
    if len(img.shape) == 2:
        formats = 'HW'
    else:
        if len(img.shape) == 4:
            img = img[0]
        if img.shape[0] == 3:
            formats = 'CHW'
        else:
            formats = 'HWC'
    if scale:
        img = (img - img.min()) / (img.max() - img.min())
    log.add_image(tag, img, step, dataformats=formats)


def add_scalar(log, loss_dict, step):
    for key in loss_dict.keys():
        log.add_scalar(key, loss_dict[key], step)


class Trainer:

    def __init__(self, opts):
        self.opts = opts
        # torch.autograd.set_detect_anomaly(True)
        if opts.local_rank <= 0:
            self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
            os.makedirs(self.save_dir, exist_ok=True)
            log_file = os.path.join(self.save_dir, 'config.txt')
            with open(log_file, 'w') as f:
                f.write(flags.FLAGS.flags_into_string())
            self.log = SummaryWriter('%s/%s' % (opts.checkpoint_dir, opts.name), comment=opts.name)

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname == 'BatchNorm2d':
            for name, p in m.named_parameters():
                p.requires_grad = False

    def define_model(self):
        self.model = MeshNet(self.opts)
        if self.opts.model_path != '':
            self.model.load_network(model_path=self.opts.model_path, iter=0)
        self.model.apply(self.set_bn_eval)
        # ddp
        if self.opts.local_rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            device = torch.device('cuda:{}'.format(self.opts.local_rank))
            self.model = self.model.to(device)
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.opts.local_rank],
                output_device=self.opts.local_rank,
                find_unused_parameters=True,
            )
            self.model = self.ddp.module
        else:
            self.model = self.model.cuda()


    def batch_reshape(self, batch):
        img = batch['img'].type(torch.FloatTensor).cuda()  # bsz, 3, 256, 256
        mask = batch['mask'].cuda().squeeze(1)  # bsz, 256, 256
        if self.opts.use_depth:
            depth = batch['depth'].cuda().squeeze(1)  # bsz, 256, 256
        else: depth = None
        if self.opts.use_occ:
            occ = batch['occ'].cuda().squeeze(1)  # bsz, 256, 256
        else: occ = None
        center = batch['center']  # bsz, 2
        length = batch['length']  # bsz, 2
        indices = batch['idx'].cuda()  # bsz
        foc = batch['foc'].cuda()  # bsz, 2
        foc_crop = batch['foc_crop'].cuda()  # bsz, 2
        pp = batch['pp'].cuda()  # bsz, 2
        pp_crop = batch['pp_crop'].cuda()  # bsz, 2
        gt = None  # not used in training

        ## change pp and foc to ndc space coordinates
        pp_crop = pp_crop / (self.opts.img_size / 2.) - 1.
        foc_crop = foc_crop / (self.opts.img_size / 2.)
        return img, mask, depth, occ, center, length, foc, foc_crop, pp, pp_crop, indices, gt


    def train(self):
        self.define_model()  # defines self.model, self.symmetric
        self.dataloader, self.dataset = data_loader(self.opts)
        self.optim = Optimizers(self.opts, self.model)

        time1 = time.time()
        self.model.train()

        for i, batch in enumerate(self.dataloader):
            if (i+1) % self.opts.batch_log_interval == 0:
                time2 = time.time()
                print('batch {}, batch size {}, mean per iter time:{}'.format(i+1, batch['img'].shape[0], (time2 - time1) / self.opts.batch_log_interval))
                time1 = time.time()
            self.model.iters = i
            self.optim.zero_grad()
            data = self.batch_reshape(batch)
            total_loss, aux_output = self.model(data)
            total_loss.mean().backward()
            grad = self.collect_grad()
            self.optim.step(i)
            self.write_log(data, total_loss, grad, aux_output, i)

            if (i+1) % self.opts.save_freq == 0: 
                self.save(i+1)
                print('saving the model at iters {:d}.'.format(i+1))


    def collect_grad(self):
        shapenerf_grad = []
        pose_predictor_grad = []
        grad_meanv_norm = 0
        for name, p in self.model.named_parameters():
            if 'mean_v' in name and p.grad is not None:
                torch.nn.utils.clip_grad_norm_(p, 1.)
                grad_meanv_norm = p.grad.view(-1).norm(2,-1)
            elif 'shapenerf' in name and p.grad is not None:
                shapenerf_grad.append(p)
            elif 'pose_predictor' in name and p.grad is not None:
                pose_predictor_grad.append(p)
            if (not p.grad is None) and (torch.isnan(p.grad).sum() > 0):
                print('bad gradient')
                # import ipdb; ipdb.set_trace()
                self.optim.zero_grad()
        grad_shapenerf_norm = torch.nn.utils.clip_grad_norm_(shapenerf_grad, 1)
        grad_pose_predictor_norm = torch.nn.utils.clip_grad_norm_(pose_predictor_grad, 0.1)
        return grad_meanv_norm, grad_shapenerf_norm, grad_pose_predictor_norm

    
    def write_log(self, data, total_loss, grad, aux_output, step):
        if self.opts.local_rank <= 0:
            grad_meanv_norm, grad_shapenerf_norm, grad_pose_predictor_norm = grad
            loss_dict = {}
            loss_dict['total_loss/total_loss'] = total_loss.mean()
            loss_dict['norms/grad_meanv_norm'] = grad_meanv_norm
            loss_dict['norms/grad_shapenerf_norm'] = grad_shapenerf_norm
            loss_dict['norms/grad_pose_predictor_norm'] = grad_pose_predictor_norm

            if 'mask_loss' in aux_output.keys(): loss_dict['render_loss/mask_loss'] = aux_output['mask_loss'].mean()
            if 'cam_loss' in aux_output.keys(): loss_dict['regularization/cam_loss'] = aux_output['cam_loss'].mean()
            if 'match_loss' in aux_output.keys(): loss_dict['correspondence/match_loss'] = aux_output['match_loss'].mean()
            if 'imatch_loss' in aux_output.keys(): loss_dict['correspondence/imatch_loss'] = aux_output['imatch_loss'].mean()
            if 'cycle_loss' in aux_output.keys(): loss_dict['correspondence/cycle_loss'] = aux_output['cycle_loss']
            if 'cycle_loss_pretrain' in aux_output.keys(): loss_dict['correspondence/cycle_loss_pretrain'] = aux_output['cycle_loss_pretrain']
            if 'texture_loss' in aux_output.keys(): loss_dict['render_loss/texture_loss'] = aux_output['texture_loss'].mean()
            if 'depth_loss' in aux_output.keys(): loss_dict['render_loss/depth_loss'] = aux_output['depth_loss'].mean()
            if 'triangle_loss' in aux_output.keys(): loss_dict['regularization/triangle_loss'] = aux_output['triangle_loss']
            if 'deform_loss' in aux_output.keys(): loss_dict['regularization/deform_loss'] = aux_output['deform_loss']
            if 'symmetry_loss' in aux_output.keys(): loss_dict['regularization/symmetry_loss'] = aux_output['symmetry_loss']
            add_scalar(self.log, loss_dict, step)

            if (step+1) % self.opts.vis_freq == 0:
                vis_id = (step+1) // self.opts.vis_freq - 1

                img, mask, depth, *_ = data
                add_image(self.log, 'vis/img', img[0].permute(1,2,0).detach().cpu().numpy(), vis_id, scale=False)
                add_image(self.log, 'vis/mask', mask[0,:,:,None].repeat(1,1,3).detach().cpu().numpy(), vis_id, scale=False)
                add_image(self.log, 'vis/depth_render', aux_output['depth_render_vis'][:,:,None].repeat(1,1,3).detach().cpu().numpy(), vis_id, scale=True)
                add_image(self.log, 'vis/depth_mean_v_render', aux_output['mean_v_render_vis'][:,:,None].repeat(1,1,3).detach().cpu().numpy(), vis_id, scale=True)
                if self.opts.use_depth:
                    add_image(self.log, 'vis/depth_gt', depth[0,:,:,None].repeat(1,1,3).detach().cpu().numpy(), vis_id, scale=True)
                    add_image(self.log, 'vis/depth_diff_render', aux_output['depth_diff_render_vis'].permute(1,2,0).detach().cpu().numpy(), vis_id, scale=True)

                add_image(self.log, 'vis/cycle_match', aux_output['cycle_match_vis'], vis_id, scale=True)
                add_image(self.log, 'vis/cycle_match_gt', aux_output['cycle_match_gt_vis'], vis_id, scale=True)
                add_image(self.log, 'vis/pt_src', aux_output['pt_src_vis'], vis_id, scale=True)
                add_image(self.log, 'vis/pt_tgt', aux_output['pt_tgt_vis'], vis_id, scale=True)
                add_image(self.log, 'vis/pt_img_src', aux_output['pt_img_src_vis'], vis_id, scale=True)
                add_image(self.log, 'vis/pt_img_tgt', aux_output['pt_img_tgt_vis'], vis_id, scale=True)

    def save(self, prefix):
        if self.opts.local_rank <= 0:
            save_filename = 'pred_net_{}.pth'.format(prefix)
            save_path = os.path.join(self.save_dir, save_filename)
            save_dict = self.model.state_dict()
            save_dict['mesh.faces'] = self.model.mesh.faces.cpu()
            torch.save(save_dict, save_path)




