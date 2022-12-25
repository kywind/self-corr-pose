from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import trimesh

from model.module.weights import Weights
from model.module.mesh import CanonicalMesh
from model.module.encoder import Encoder
from model.module.correspondence import Correspondence
from model.module.pretrained_corr import PretrainedCorrespondence
from model.module.renderer import Renderer

from model.util.loss_utils import *


flags.DEFINE_bool('feat_shape', False, '')
flags.DEFINE_bool('flatten_loss', False, 'whether to use flatten loss')
flags.DEFINE_bool('camera_loss', False, 'whether to use camera movement loss')
flags.DEFINE_bool('depth_loss_chamfer', False, '')
flags.DEFINE_bool('use_depth', False, 'use depth info')
flags.DEFINE_bool('surface_texture', False, 'use surface texture')
flags.DEFINE_float('vert_lr_ratio', 0.1, 'vert learning rate ratio')
flags.DEFINE_float('cam_lr_ratio', 0.1, 'camera learning rate ratio')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('n_tex_sample', 6, 'number of texture sampling')
flags.DEFINE_integer('nz_feat', 128, 'Encoded feature size')
flags.DEFINE_integer('codedim', 16, 'Encoded feature size')
flags.DEFINE_integer('n_corr_feat', 16, 'Encoded feature size')


class MeshNet(nn.Module):

    def __init__(self, opts):
        super(MeshNet, self).__init__()
        self.opts = opts

        self.mesh = CanonicalMesh(opts)
        self.weights = Weights(opts)
        self.encoder = Encoder(opts)
        self.corr_net = Correspondence(opts)
        self.pretrain_corr_net = PretrainedCorrespondence(self.opts, self.mesh, pretrained=True)
        self.renderer = Renderer(opts, self.mesh)
        self.iters = 0

        self.triangle_loss_fn = LaplacianLoss(self.mesh.mean_v, self.mesh.faces, average=True)
        if self.opts.flatten_loss: 
            self.flatten_loss_fn = FlattenLoss(self.mesh.faces, average=True)


    def forward(self, data):
        opts = self.opts
        self.weights.schedule(self.iters)

        img, mask, depth, occ, center, length, foc, foc_crop, pp, pp_crop, indices, gt = data
        bsz = img.shape[0]

        mean_v = self.mesh.mean_v[None].repeat(bsz,1,1)
        faces = self.mesh.faces[None].repeat(bsz,1,1)

        img_feat, mesh_feat, pred_v, rotation, translation, scale = self.encoder(img, mean_v, pp_crop, foc_crop)

        pointcorr, match, imatch, match_conf = self.corr_net.match(img_feat, mesh_feat, mask, pred_v)
        tex = self.mesh.get_texture(pred_v, faces, imatch, img)
        
        if not opts.train:
            return pred_v, faces, tex, imatch, match, match_conf, rotation, translation, scale, pointcorr
        
        mask_render, tex_render, depth_render, match_gt, imatch_gt, tex_mask, depth_mask, match_mask, depth_weight = \
                self.renderer.render_all(pred_v, faces, tex, foc_crop, pp_crop, rotation, translation, scale)

        if opts.use_occ:
            mask_loss_sub = compute_mask_loss_with_occ(img, mask, mask_render, occ)
            texture_loss_sub = compute_texture_loss_with_occ(img, mask, tex_render, tex_mask, occ)
        else:
            mask_loss_sub = compute_mask_loss(img, mask, mask_render)
            texture_loss_sub = compute_texture_loss(img, mask, tex_render, tex_mask)
        if opts.use_depth:
            if opts.depth_loss_chamfer:
                depth_loss_sub, depth_diff = compute_depth_loss_chamfer(pred_v, faces, depth, depth_render, depth_mask, mask, \
                            pp_crop, foc_crop, rotation, translation)
            else:
                depth_loss_sub, depth_diff = compute_depth_loss(depth, depth_render, depth_mask, mask)

        match_loss_sub = compute_match_loss(match, match_gt, match_mask, mask)
        imatch_loss_sub = compute_imatch_loss(imatch, imatch_gt, depth_weight)
        
        mask_loss = self.weights.mask_wt * mask_loss_sub.mean(0)
        match_loss = self.weights.match_wt * match_loss_sub.mean(0)
        texture_loss = self.weights.tex_wt * texture_loss_sub.mean(0)
        imatch_loss = self.weights.imatch_wt * imatch_loss_sub.mean(0)
        if opts.use_depth:
            depth_loss = self.weights.depth_wt * depth_loss_sub.mean(0)

        ## other losses (symmetry, smoothness, camera, etc.)
        symmetry_loss = self.weights.symmetry_wt * self.mesh.compute_symmetry_loss(pred_v, faces)

        triangle_loss = self.weights.triangle_wt * self.triangle_loss_fn(pred_v) * pred_v.shape[1] / 64.
        if opts.flatten_loss:
            triangle_loss += self.weights.triangle_wt * self.flatten_loss_fn(pred_v) * 0.1 * np.sqrt(pred_v.shape[1] / 64.)

        pullfar_loss = self.weights.pullfar_wt * F.relu(1 - translation[:, :, -1]).mean()

        deform_loss = self.weights.deform_wt * F.smooth_l1_loss(pred_v, mean_v, reduction='mean')

        cycle_loss_pt, pt_pts_src, pt_pts_tgt, pt_match, pt_mask, pt_img_src, pt_img_tgt = \
                self.pretrain_corr_net.compute_cycle_loss(img, mask, depth_weight, pointcorr)
        cycle_loss_pt *= self.weights.cycle_loss_pt_wt

        cycle_loss, cycle_match, cycle_match_gt, cycle_pt_mask = \
            self.corr_net.compute_rotation_cycle_loss(img, mask, img_feat, self.encoder)
        cycle_loss *= self.weights.cycle_loss_wt

        if opts.camera_loss:
            rotation_2 = rotation.detach().clone().reshape(-1, opts.repeat, 3, 3)
            rotation_2 = torch.cat((rotation_2[:, 1:], rotation_2[:, :1]), dim=1).reshape(bsz, 3, 3)
            cam_loss = self.weights.camera_wt * compute_camera_loss(rotation, rotation_2).mean()
    
        total_loss = mask_loss + symmetry_loss + triangle_loss + deform_loss + pullfar_loss + \
            texture_loss + match_loss + imatch_loss + cycle_loss_pt + cycle_loss
        if opts.use_depth:
            total_loss += depth_loss
        if opts.camera_loss:
            total_loss += cam_loss

        aux_output = {
            'total_loss': total_loss, 
            'mask_loss': mask_loss,
            'triangle_loss': triangle_loss,
            'deform_loss': deform_loss,
            'pullfar_loss': pullfar_loss,
            'symmetry_loss': symmetry_loss,
            'match_loss': match_loss,
            'texture_loss': texture_loss,
            'imatch_loss': imatch_loss,
            'cycle_loss_pretrain': cycle_loss_pt,
            'cycle_loss': cycle_loss,
        }
        if opts.use_depth: 
            aux_output['depth_loss'] = depth_loss
        if opts.camera_loss:
            aux_output['cam_loss'] = cam_loss

        ## visualize
        if (self.iters+1) % opts.vis_freq == 0:
            bsz = pred_v.shape[0]
            
            with torch.no_grad():
                pred_v_vis = pred_v[0].cpu().numpy()
                xmin, xmax = pred_v_vis[:, 0].min(), pred_v_vis[:, 0].max()
                ymin, ymax = pred_v_vis[:, 1].min(), pred_v_vis[:, 1].max()
                zmin, zmax = pred_v_vis[:, 2].min(), pred_v_vis[:, 2].max()

                pred_v_vis[:, 0] = (pred_v_vis[:, 0] - xmin) / (xmax - xmin)
                pred_v_vis[:, 1] = (pred_v_vis[:, 1] - ymin) / (ymax - ymin)
                pred_v_vis[:, 2] = (pred_v_vis[:, 2] - zmin) / (zmax - zmin)

                img_vis = img[0].permute(1,2,0).flip(-1).cpu().numpy() * 255
                match_vis = match[0].permute(1,2,0).cpu().numpy()
                match_gt_vis = match_gt[0].permute(1,2,0).cpu().numpy()

                match_gt_vis[:, :, 0] = (match_gt_vis[:, :, 0] - xmin) / (xmax - xmin)
                match_gt_vis[:, :, 1] = (match_gt_vis[:, :, 1] - ymin) / (ymax - ymin)
                match_gt_vis[:, :, 2] = (match_gt_vis[:, :, 2] - zmin) / (zmax - zmin)
                match_gt_vis *= 255.

                match_vis[:, :, 0] = (match_vis[:, :, 0] - xmin) / (xmax - xmin)
                match_vis[:, :, 1] = (match_vis[:, :, 1] - ymin) / (ymax - ymin)
                match_vis[:, :, 2] = (match_vis[:, :, 2] - zmin) / (zmax - zmin)
                match_vis *= 255.
                
                depthw_vis = img_vis.copy()
                imatch_vis = np.ones((256, 256, 3)) * 255.
                imatch_gt_vis = np.ones((256, 256, 3)) * 255.

                for pi, (point, point_gt) in enumerate(zip(imatch[0].T, imatch_gt[0].T)):  
                    # for imatch_gt (projected vertex locations), (x,y), x: right, y: up, center: 0,0
                    point = point.detach().cpu().numpy()
                    point = (point+1) * 128
                    point_gt = point_gt.detach().cpu().numpy()
                    point_gt = (point_gt+1) * 128
                    color = (int(pred_v_vis[pi, 2] * 255), \
                             int(pred_v_vis[pi, 1] * 255), \
                             int(pred_v_vis[pi, 0] * 255))
                    color_den = int(depth_weight[0, pi] * 255)
                    if color_den < 128:
                        continue
                    cv2.circle(imatch_vis, (int(point[0]), int(point[1])), 3, color, -1)
                    cv2.circle(imatch_gt_vis, (int(point_gt[0]), int(point_gt[1])), 3, color, -1)
                    cv2.circle(depthw_vis, (int(point_gt[0]), int(point_gt[1])), 3, (color_den,color_den,color_den), -1)

                imatch_vis = imatch_vis[:,:,::-1]
                imatch_gt_vis = imatch_gt_vis[:,:,::-1]
                depthw_vis = depthw_vis[:,:,::-1]

                if opts.use_depth:
                    depth_diff_render_vis = depth_diff[0][None].repeat(3, 1, 1)
                    depth_diff_render_vis[0][depth_diff_render_vis[0] > 0] = 0  # red: depth_diff < 0, estimated z is too small
                    depth_diff_render_vis[0] *= -1
                    depth_diff_render_vis[1][depth_diff_render_vis[1] < 0] = 0  # green: depth_diff > 0, estimated z is too large
                    depth_diff_render_vis[2] = 0
                else:
                    depth_diff_render_vis = None

                texture_render_vis = tex_render[0].detach()
                depth_render_vis = depth_render[0].detach()
                depth_mask_vis = depth_mask[0].detach()
                try: depth_render_vis[depth_mask_vis == 0] = depth_render_vis[depth_mask_vis > 0].max()
                except: print('depth_mask_vis empty!')# pdb.set_trace()

                mean_v_render_vis = self.renderer.render_mean_mesh(\
                            foc_crop[0:1], pp_crop[0:1], rotation[0:1], translation[0:1])
                mean_v_mask = mean_v_render_vis[0, 3]
                mean_v_render_vis = mean_v_render_vis[0, 2]
                try: mean_v_render_vis[mean_v_mask == 0] = mean_v_render_vis[mean_v_mask > 0].max()
                except: print('mean_v_mask empty!')# pdb.set_trace()

                mesh_save = trimesh.Trimesh(pred_v[0].detach().cpu().numpy(), 
                                            faces[0].detach().cpu().numpy(), 
                                            process=False, 
                                            vertex_colors=None)
                mean_mesh = trimesh.Trimesh(self.mesh.mean_v.detach().cpu().numpy(), 
                                            self.mesh.faces.detach().cpu().numpy(), 
                                            process=False, 
                                            vertex_colors=None)
                mean_mesh.export('{}/{}/{}-iter-mean-mesh.obj'.format(opts.checkpoint_dir, opts.name, self.iters+1), file_type='obj')

            with torch.no_grad():
                
                cycle_grid = self.corr_net.meshgrid.reshape(2, self.corr_net.hf, self.corr_net.wf)[None].repeat(bsz,1,1,1)
                cycle_grid = F.interpolate(cycle_grid, (self.corr_net.hf//2, self.corr_net.wf//2), mode='nearest')
                cycle_grid = cycle_grid.reshape(bsz,2,-1)
                cycle_match_vis = np.ones((256, 256, 3)) * 255.
                cycle_match_gt_vis = np.ones((256, 256, 3)) * 255.

                for pi, (point, point_gt) in enumerate(zip(cycle_match[0].T, cycle_match_gt[0].T)):  
                    # for imatch_gt (projected vertex locations), (x,y), x: right, y: up, center: 0,0
                    point = point.detach().cpu().numpy()
                    point = (point+1) * 128
                    point_gt = point_gt.detach().cpu().numpy()
                    point_gt = (point_gt+1) * 128
                    color = (0, \
                             int(cycle_grid[0, 1, pi] * 127 + 128), \
                             int(cycle_grid[0, 0, pi] * 127 + 128))
                    color_den = int(cycle_pt_mask[0, pi] * 255)
                    if color_den < 128:
                        continue
                    cv2.circle(cycle_match_vis, (int(point[0]), int(point[1])), 3, color, -1)
                    cv2.circle(cycle_match_gt_vis, (int(point_gt[0]), int(point_gt[1])), 3, color, -1)

                pt_img_src_vis = pt_img_src[0].permute(1,2,0).cpu().numpy() * 255
                pt_img_tgt_vis = pt_img_tgt[0].permute(1,2,0).cpu().numpy() * 255

                pt_src_vis = np.ones((256, 256, 3)) * 255.
                pt_tgt_vis = np.ones((256, 256, 3)) * 255.
                pt_pred_vis = np.ones((256, 256, 3)) * 255.
                pt_src_vis = 0.7 * pt_src_vis + 0.3 * pt_img_src_vis
                pt_tgt_vis = 0.7 * pt_tgt_vis + 0.3 * pt_img_tgt_vis
            
                for pi, (point_src, point_tgt, point_pred) in enumerate(zip(pt_pts_src[0].T, pt_pts_tgt[0].T, pt_match[0].T)):
                    # for imatch_gt (projected vertex locations), (x,y), x: right, y: up, center: 0,0
                    point_tgt = point_tgt.detach().cpu().numpy()
                    point_tgt = (point_tgt+1) * 128
                    point_src = point_src.detach().cpu().numpy()
                    point_src = (point_src+1) * 128
                    point_pred = point_pred.detach().cpu().numpy()
                    point_pred = (point_pred+1) * 128
                    color = (int(pt_pts_tgt[0, 0, pi] * 127 + 128), \
                             int(pt_pts_tgt[0, 1, pi] * 127 + 128), 0)
                    color_den = int(pt_mask[0, pi] * 255)
                    if color_den < 128:
                        continue
                    cv2.circle(pt_src_vis, (int(point_src[0]), int(point_src[1])), 3, color, -1)
                    cv2.circle(pt_tgt_vis, (int(point_tgt[0]), int(point_tgt[1])), 3, color, -1)
                    cv2.circle(pt_pred_vis, (int(point_pred[0]), int(point_pred[1])), 3, color, -1)
                


            aux_output['depth_render_vis'] = depth_render_vis
            aux_output['mean_v_render_vis'] = mean_v_render_vis

            aux_output['match_vis'] = match_vis
            aux_output['match_gt_vis'] = match_gt_vis
            aux_output['texture_render_vis'] = texture_render_vis
            aux_output['imatch_vis'] = imatch_vis
            aux_output['imatch_gt_vis'] = imatch_gt_vis
            aux_output['depthw_vis'] = depthw_vis
            if opts.use_depth: 
                aux_output['depth_diff_render_vis'] = depth_diff_render_vis

            aux_output['cycle_match_vis']  = cycle_match_vis
            aux_output['cycle_match_gt_vis']  = cycle_match_gt_vis
            aux_output['pt_src_vis'] = pt_src_vis
            aux_output['pt_tgt_vis'] = pt_tgt_vis
            aux_output['pt_pred_vis'] = pt_pred_vis
            aux_output['pt_img_src_vis'] = pt_img_src_vis
            aux_output['pt_img_tgt_vis'] = pt_img_tgt_vis

        return total_loss, aux_output



    def load_network(self, model_path, iter=0):
        print('loading {}..'.format(model_path))
        states = torch.load(model_path)

        keys = list(states.keys())
        for name in keys:
            if name.endswith('mean_v'):
                self.num_verts = states[name].shape[-2]
            if name.endswith('faces'):
                self.num_faces = states[name].shape[-2]
            if 'symm_rots' in name or 'triangle_loss_fn' in name or 'flatten_loss_fn' in name:
                states.pop(name)
                continue

        self.load_state_dict(states, strict=False)
        print('network load complete')
    
