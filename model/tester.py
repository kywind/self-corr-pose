

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import os
import sys
sys.path.insert(0,'third-party')

import time
import pdb
import numpy as np
from absl import flags
import cv2
import trimesh
import time
from tqdm import tqdm
import copy
import kornia

import soft_renderer as sr

from data.dataloader import test_loader
from model.model import MeshNet
from model.util.loss_utils import pinhole_cam
from model.util.umeyama import estimateSimilarityTransform

from model.util.eval_utils import map_kp,  draw_kp, get_best_iou, get_best_deg_cm, draw_bboxes, draw_bboxes_3d
from model.util.base_rot import get_base_rot
from objectron.dataset import box


flags.DEFINE_bool('eval', False, 'whether to evaluate the prediction (only for test)')
flags.DEFINE_bool('eval_nocs', False, 'evaluate in NOCS style (3diou, deg-cm error)')
flags.DEFINE_bool('eval_cub', False, 'evaluate in CUB style (2diou, rotation error, keypoint transfer)')

flags.DEFINE_bool('vis_pred', False, 'whether to visualize prediction')
flags.DEFINE_bool('visualize_mesh', False, '')
flags.DEFINE_bool('visualize_conf', False, '')
flags.DEFINE_bool('visualize_match', False, '')
flags.DEFINE_bool('visualize_imatch', False, '')
flags.DEFINE_bool('visualize_gt', False, '')
flags.DEFINE_bool('visualize_bbox', False, '')
flags.DEFINE_bool('visualize_depth', False, '')
flags.DEFINE_bool('visualize_tex', False, '')
flags.DEFINE_bool('visualize_mask', False, '')
flags.DEFINE_bool('match_with_bbox', False, '')


class Tester:

    def __init__(self, opts):
        self.opts = opts
        # torch.autograd.set_detect_anomaly(True)
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        os.makedirs(self.save_dir, exist_ok=True)

        if opts.local_rank <= 0:
            log_file = os.path.join(self.save_dir, 'config-test.txt')
            with open(log_file, 'w') as f:
                f.write(flags.FLAGS.flags_into_string())
            if opts.vis_pred: os.makedirs(opts.vis_path, exist_ok=True)


    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname == 'BatchNorm2d':
            for name, p in m.named_parameters():
                p.requires_grad = False
                # print(name)
                # print(p.__class__.__name__)


    def define_model(self):
        self.model = MeshNet(self.opts)
        assert(self.opts.model_path != '')
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
        if 'rot_gt' in batch.keys():
            rot_gt = batch['rot_gt'].cuda()
            trans_gt = batch['trans_gt'].cuda()
            scale_gt = batch['scale_gt'].cuda()
            gt = (rot_gt, trans_gt, scale_gt)
        else:
            gt = None
        
        ## change pp and foc to ndc space coordinates
        pp_crop = pp_crop / (self.opts.img_size / 2.) - 1.
        foc_crop = foc_crop / (self.opts.img_size / 2.)
        return img, mask, depth, occ, center, length, foc, foc_crop, pp, pp_crop, indices, gt


    def test(self):  # predict when the model is loaded
        self.define_model()  # defines self.model, self.symmetric

        if self.opts.surface_texture: self.texture_type = 'surface'
        else: self.texture_type = 'vertex'

        h = self.opts.img_size
        w = self.opts.img_size
        meshgrid = torch.Tensor(np.array(np.meshgrid(range(w), range(h)))).cuda().reshape(2, -1) + 0.5 # 2,h*w
        meshgrid = meshgrid / (w / 2) - 1
        meshgrid = meshgrid.reshape(-1)  # 2*h*w
        self.meshgrid = meshgrid

        hf = self.opts.corr_h
        wf = self.opts.corr_w
        meshgrid_corr = torch.Tensor(np.array(np.meshgrid(range(wf), range(hf)))).cuda().reshape(2, -1) + 0.5 # 2,h*w
        meshgrid_corr = meshgrid_corr / (wf / 2) - 1
        meshgrid_corr = meshgrid_corr.reshape(-1)  # 2*h*w
        self.meshgrid_corr = meshgrid_corr

        self.renderer_hard = sr.SoftRenderer(image_size=self.opts.img_size, sigma_val=1e-12, gamma_val=1e-4,
            camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
            light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)    

        self.base_rot = get_base_rot(self.opts)
        if self.opts.eval_nocs:
            self.iou_thresh = [0.25, 0.5]
            self.iou_result = []
            self.deg_cm_thresh = [[5, 2], [5, 5], [10, 2], [10, 5]]
            self.deg_cm_result = []
        elif self.opts.eval_cub:
            self.iou_thresh = [0.25, 0.5]
            self.iou_result = []
            self.kps_thresh = [0.1, 0.2]  # alpha
            self.kps_result = []

        self.model.eval()
        total_steps = 0
        self.dataloader, self.dataset = test_loader(self.opts)

        with torch.no_grad():
            total_steps = 0
            for i, batch in tqdm(enumerate(self.dataloader)):
                self.model.iters = i
                self.model.total_steps = total_steps

                if self.opts.eval: gt = self.parse_gt(batch)
                else: gt = None
                video_id = batch['idx']
                frame_id = batch['frame_idx']

                print(f'testing batch {i}/{len(self.dataloader)}')

                data = self.batch_reshape(batch)
                pred = self.model(data)

                pred_fit = self.pose_fitting(data, pred)
                if self.opts.eval: self.eval(video_id, frame_id, data, pred, pred_fit, gt)
                if self.opts.vis_pred: self.visualize(i, video_id, frame_id, data, pred, pred_fit, gt)
                total_steps += 1
        
        if self.opts.eval: 
            if self.opts.eval_nocs:
                iou_result = np.array(self.iou_result) * 1.0
                deg_cm_result = np.array(self.deg_cm_result) * 1.0
                print('iou@25:', iou_result[:, 0].sum() / iou_result.shape[0])
                print('iou@50:', iou_result[:, 1].sum() / iou_result.shape[0])
                print(  '5deg*2cm:', deg_cm_result[:, 0].sum() / deg_cm_result.shape[0])
                print(  '5deg*5cm:', deg_cm_result[:, 1].sum() / deg_cm_result.shape[0])
                print( '10deg*2cm:', deg_cm_result[:, 2].sum() / deg_cm_result.shape[0])
                print( '10deg*5cm:', deg_cm_result[:, 3].sum() / deg_cm_result.shape[0])
            elif self.opts.eval_cub:
                iou_result = np.array(self.iou_result) * 1.0
                kps_result = np.array(self.kps_result) * 1.0
                print(  'mIoU:', iou_result.mean())
                print('kp@0.1:', kps_result[:, 0].sum() / kps_result.shape[0]) 
                print('kp@0.2:', kps_result[:, 1].sum() / kps_result.shape[0]) 


    def parse_gt(self, batch):
        if self.opts.dataset_name == 'cub':
            sfm_pose = batch['sfm_pose']
            kp = batch['kp']
            gt = (sfm_pose, kp)
        else:
            rotation = batch['rotation']
            translation = batch['translation']
            scale = batch['scale']
            gt = (rotation, translation, scale)
        return gt


    def eval(self, video_id, frame_id, batch, pred, pred_fit, gt):
        if self.opts.eval_nocs: self.eval_nocs(video_id, frame_id, pred_fit, gt)
        elif self.opts.eval_cub: self.eval_cub(video_id, frame_id, batch, pred, pred_fit, gt)
        else: raise NotImplementedError

    
    def eval_cub(self, video_id, frame_id, batch, pred, pred_fit, gt):
        img, mask, depth, occ, center, length, foc, foc_crop, pp, pp_crop, indices, *_ = batch
        pred_v, faces, tex, imatch, match, match_conf, rotation, translation, scale, pointcorr = pred
        bbox_pred, verts, rotation_fit, translation_fit = pred_fit
        sfm_pose, kp = gt
        bsz = mask.shape[0]
        
        ## render mask
        self.renderer_hard.rasterizer.background_color = [0, 0, 0]
        mask_render = self.render(self.renderer_hard, pred_v, faces, None, foc_crop, pp_crop, rotation_fit, translation_fit, \
                            rotation_detach=False, translation_detach=False, render_mask=True, texture_type='vertex')[:, 2]

        ## compute iou
        intersection = mask * mask_render
        union = mask + mask_render - intersection
        iou = intersection.sum((1,2)) / union.sum((1,2))

        for i in range(bsz):
            iou_i = iou[i].item()
            self.iou_result.append(iou_i)

        ## eval kp transfer
        ## process gt
        kp = kp.float().cuda()
        kps_vis = (kp[:, :, 2] > 0) * 1.0
        kps = kp * 1.0
        # kps = (kp * 0.5 + 0.5) * self.opts.img_size

        assert self.opts.shuffle_test, 'kp eval requires shuffle_test'
        bsz_half = bsz // 2
        # assert bsz == 2 * bsz_half, 'kp eval requires even batch size'

        transfer_kps, kp_transfer_error, min_dist, kps_mask = map_kp(
            kps_vis[:bsz_half].clone(), kps_vis[bsz_half:2*bsz_half].clone(), 
            kps[:bsz_half].clone(),     kps[bsz_half:2*bsz_half].clone(), 
            match[:bsz_half].clone(),   match[bsz_half:2*bsz_half].clone(), 
            mask[:bsz_half].clone(),    mask[bsz_half:2*bsz_half].clone()
        )
        kp_transfer_error = kp_transfer_error.cpu().numpy()
        transfer_kps = transfer_kps.cpu().numpy()
        kps = kps.cpu().numpy()
        kps_mask = kps_mask.cpu().numpy()
        img_vis = img.permute(0,2,3,1).cpu().numpy() * 255
        mask_vis = mask.cpu().numpy()[:,:,:,None]
        
        for i in range(bsz_half):
            # visualize
            if self.opts.vis_pred:
                print(video_id[i].item(), frame_id[i].item(), kp_transfer_error[i])
                img1, trans_img2, img2 = draw_kp(
                    img_vis[i].copy(), img_vis[i+bsz_half].copy(), 
                    kps[i].copy(), kps[i+bsz_half].copy(), 
                    transfer_kps[i].copy(), kps_mask[i].copy())
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_1.png'.format(video_id[i].item(), frame_id[i].item())), img1)  # "X" cross is on x axis
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_2.png'.format(video_id[i].item(), frame_id[i].item())), trans_img2)  # "X" cross is on x axis
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_2_gt.png'.format(video_id[i].item(), frame_id[i].item())), img2)  # "X" cross is on x axis

        kp_transfer_error = kp_transfer_error[kps_mask > 0]

        padding = 0.2
        base_padding = 0.0
        kp_scale = (1 + 2 * padding) / (1 + 2 * base_padding) / 2

        for i in range(kp_transfer_error.shape[0]):
            kps_result = [False] * len(self.kps_thresh)
            for j in range(len(self.kps_thresh)):
                if kp_transfer_error[i] * kp_scale < self.kps_thresh[j]:  # error is in -1~1, thresh need to double
                    kps_result[j] = True
            self.kps_result.append(kps_result)


    def eval_nocs(self, video_id, frame_id, pred_fit, gt):
        bbox_pred, verts, rotation, translation = pred_fit
        rot_gt, trans_gt, scale_gt = gt
        bsz = bbox_pred.shape[0]
        bbox_pred = bbox_pred.cpu().numpy()
        rot_gt = rot_gt.cpu().numpy()
        trans_gt = trans_gt.cpu().numpy()
        scale_gt = scale_gt.cpu().numpy()
        # import pdb; pdb.set_trace()  # use this when fixing base_rot

        for i in range(bsz):
            box_pred = box.Box(bbox_pred[i])

            best_iou, best_ae, best_pe = get_best_iou(self.opts.symmetry_idx, box_pred, rot_gt[i], trans_gt[i], scale_gt[i])
            angle_error, trans_error = get_best_deg_cm(self.opts.symmetry_idx, box_pred, rot_gt[i], trans_gt[i], scale_gt[i])

            iou_result = [False] * len(self.iou_thresh)
            for j in range(len(self.iou_thresh)):
                if best_iou >= self.iou_thresh[j]:
                    iou_result[j] = True
            self.iou_result.append(iou_result)
            
            deg_cm_result = [False] * len(self.deg_cm_thresh)
            for j in range(len(self.deg_cm_thresh)):
                if angle_error < self.deg_cm_thresh[j][0] and trans_error < self.deg_cm_thresh[j][1]:
                    deg_cm_result[j] = True
            self.deg_cm_result.append(deg_cm_result)


    def pose_fitting(self, batch, pred):
        img, mask, depth, occ, center, length, foc, foc_crop, pp, pp_crop, indices, *_ = batch
        pred_v, faces, tex, imatch, match, match_conf, *_ = pred
        bsz = img.shape[0]
        
        intr = torch.eye(3)[None].repeat(bsz, 1, 1).cuda()
        intr[:, 0, 0] = foc_crop[:, 0]
        intr[:, 1, 1] = foc_crop[:, 1]
        intr[:, 0, 2] = pp_crop[:, 0]
        intr[:, 1, 2] = pp_crop[:, 1]
        intr_inv = intr.inverse()

        h = self.opts.img_size
        w = self.opts.img_size

        mask_final = (depth > 0)[:, None] * mask[:, None] * match_conf
        mask_final_1d = mask_final.reshape(bsz, -1)
        mask_final_2d = mask_final.repeat(1,2,1,1).reshape(bsz, -1)
        mask_final_3d = mask_final.repeat(1,3,1,1).reshape(bsz, -1) # bsz, 3*h*w
        
        rot_list = []
        trans_list = []
        scale_list = []
        for i in range(bsz):
            ## mask meshgrid
            meshgrid = self.meshgrid[mask_final_2d[i] > 0]  # 2*h*w[2*h*w]->2*n
            meshgrid = meshgrid.reshape(2, -1).permute(1,0)  # n,2 
            ## flatten match
            match_i = match[i].reshape(-1)  # 3*h*w
            ## mask match
            match_i = match_i[mask_final_3d[i] > 0]  # 3*h*w[3*h*w]->3*n
            match_i = match_i.reshape(3, -1).permute(1,0)  # n,3
            ## flatten depth
            depth_i = depth[i].reshape(-1)  # h*w
            ## mask depth
            depth_i = depth_i[mask_final_1d[i] > 0]  # h*w[h*w]->n
            depth_i = depth_i[:, None]  # n,1

            ## reproject points
            length_i = meshgrid.shape[0]
            ones = torch.ones((length_i, 1), device=meshgrid.device)
            meshgrid_hom = torch.cat((meshgrid, ones), dim=-1)  # n,3
            xyz = meshgrid_hom.mm(intr_inv[i].permute(1,0)).contiguous()  # n,3 @ 3,3 -> n,3
            pts = xyz * depth_i / xyz[:, 2:]  # n,3

            try:
                # print('running Umeyama algorithm')
                scale, rotation, translation, outtransform = estimateSimilarityTransform(
                    match_i, pts
                )
            except:
                print('Umeyama algorithm fails, using default pose')
                scale = torch.tensor([100, 100, 100], device=pred_v.device, dtype=pred_v.dtype)
                rotation = torch.eye(3, device=pred_v.device, dtype=pred_v.dtype)
                translation = torch.tensor([0, 0, 500], device=pred_v.device, dtype=pred_v.dtype)
                outtransform = None

            scale = scale.reshape(-1)
            rotation = rotation.reshape(3, 3)
            translation = translation.reshape(-1)

            rot_list.append(rotation)
            trans_list.append(translation)
            scale_list.append(scale)

        rotation = torch.stack(rot_list, 0)
        translation = torch.stack(trans_list, 0) * 0.001
        scale_fit = torch.stack(scale_list, 0) * 0.001

        rotation = rotation.view(-1, 3, 3)  # bsz, 3, 3
        translation = translation.reshape(-1, 1, 3)  # bsz, 1, 3
        scale_fit = scale_fit.reshape(-1, 1, 3)  # bsz, 3


        # read depth and mask, compute depth ratio
        verts = pred_v.bmm(rotation) + translation
        verts_proj = pinhole_cam(verts, pp_crop, foc_crop)

        base_rot = self.base_rot.repeat(bsz,1,1)
        pred_v = pred_v.bmm(base_rot.permute(0,2,1))
        rotation = base_rot.bmm(rotation)

        minx, maxx, miny, maxy, minz, maxz = \
                pred_v[:, :, 0].min(1).values, pred_v[:, :, 0].max(1).values, \
                pred_v[:, :, 1].min(1).values, pred_v[:, :, 1].max(1).values, \
                pred_v[:, :, 2].min(1).values, pred_v[:, :, 2].max(1).values
        scale = torch.stack([maxx-minx, maxy-miny, maxz-minz], dim=-1)[:, None]  # bsz, 1, 3, overwrite scale in pred
        scale *= scale_fit

        bbox = torch.stack([
            torch.stack([(minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2], dim=-1),
            torch.stack([minx, miny, minz],  dim=-1),
            torch.stack([minx, miny, maxz], dim=-1),
            torch.stack([minx, maxy, minz], dim=-1),
            torch.stack([minx, maxy, maxz], dim=-1),
            torch.stack([maxx, miny, minz], dim=-1),
            torch.stack([maxx, miny, maxz], dim=-1),
            torch.stack([maxx, maxy, minz], dim=-1),
            torch.stack([maxx, maxy, maxz], dim=-1)
        ], dim=-2)  # bsz, 9, 3

        bbox = (bbox * scale_fit).bmm(rotation) + translation
        pred_v = (pred_v * scale_fit).bmm(rotation) + translation
        return bbox, pred_v, rotation, translation


    def save(self, prefix):
        if self.opts.local_rank <= 0:
            save_filename = '{}_net_{}.pth'.format('pred', prefix)
            save_path = os.path.join(self.save_dir, save_filename)
            save_dict = self.model.state_dict()
            save_dict['faces'] = self.model.faces.cpu()
            torch.save(save_dict, save_path)


    def reset_model(self, states):
        self.model.load_state_dict(states, strict=False)  # reload the model
        self.model.iters = 0
        self.model.total_steps = 0
        print('model reset complete')
    

    def render(self, renderer, verts, faces, tex, foc, pp, rotation, translation, \
                rotation_detach=False, translation_detach=False, render_depth=False, render_mask=False, texture_type='vertex'):
        if rotation_detach: rot = rotation.clone().detach()
        else: rot = rotation.clone()
        if translation_detach: trans = translation.clone().detach()
        else: trans = translation.clone()
        verts = verts.bmm(rot) + trans
        verts = pinhole_cam(verts, pp, foc)
        verts[:, :, 1] *= -1
        if render_depth: tex = verts.clone()
        if render_mask: result = renderer.render_mesh(sr.Mesh(verts, faces))
        else: result = renderer.render_mesh(sr.Mesh(verts, faces, tex, texture_type=texture_type))
        return result


    def visualize(self, batch_id, video_id, frame_id, batch, pred, pred_fit, gt):  # record result from a batch forward, assume batch size == 1
        if self.opts.eval_cub: return
        img, mask, depth, occ, center, length, foc, foc_crop, pp, pp_crop, indices, *_ = batch
        pred_v, faces, tex, imatch, match, match_conf, rotation, translation, scale, *_ = pred
        bbox, verts, *_ = pred_fit
        bbox = bbox.cpu().numpy()
        verts = verts.cpu().numpy()
        if self.opts.eval: 
            if self.opts.eval_nocs:
                rot_gt, trans_gt, scale_gt = gt
            elif self.opts.eval_cub:
                sfm_pose, kp = gt
                sfm_pose = sfm_pose
                kp = kp.cpu().numpy()
                rot_gt = kornia.geometry.quaternion_to_rotation_matrix(sfm_pose[:, 3:7])
                trans_gt = translation
                scale_gt = scale
            verts_gt = pred_v.bmm(rot_gt.permute(0,2,1).cuda().float()) + trans_gt.reshape(-1, 1, 3).cuda().float()
            verts_gt = verts_gt.cpu().numpy()
            rot_gt = rot_gt.cpu().numpy()
            trans_gt = trans_gt.cpu().numpy()
            scale_gt = scale_gt.cpu().numpy()
        pp = pp.cpu().numpy()
        foc = foc.cpu().numpy()
        center = center.cpu().numpy()
        length = length.cpu().numpy()

        bsz = img.shape[0]
        for i in range(bsz):
                    
            img_orig = cv2.imread(self.dataset.imglist[video_id[i].item()][frame_id[i].item()])
            if self.opts.dataset_name == 'nocs':
                mask_orig = cv2.imread(self.dataset.masklist[video_id[i].item()][frame_id[i].item()])
                meta_orig = self.dataset.metalist[video_id[i].item()][frame_id[i].item()]
                frame_obj_id = meta_orig['id']
                mask_orig = (mask_orig == frame_obj_id).astype(bool) * 1.0
            else:
                mask_orig = cv2.imread(self.dataset.masklist[video_id[i].item()][frame_id[i].item()]) / 255.
            h, w = img_orig.shape[0], img_orig.shape[1]

            bbox_i = bbox[i]
            cc = bbox[i][0]
            xx = bbox_i[[2,4,6,8]].mean(0) - cc
            yy = bbox_i[[1,2,5,6]].mean(0) - cc
            zz = bbox_i[[5,6,7,8]].mean(0) - cc
            x_len = np.linalg.norm(xx)
            y_len = np.linalg.norm(yy)
            z_len = np.linalg.norm(zz)
            dir_len = 1 * min(x_len, y_len, z_len)
            xx = xx / x_len * dir_len + cc
            yy = yy / y_len * dir_len + cc
            zz = zz / z_len * dir_len + cc
            dir_pts = np.stack([cc, xx, yy, zz], axis=0)

            minx, maxx, miny, maxy, minz, maxz = \
                pred_v[i, :, 0].min().item(), pred_v[i, :, 0].max().item(), \
                pred_v[i, :, 1].min().item(), pred_v[i, :, 1].max().item(), \
                pred_v[i, :, 2].min().item(), pred_v[i, :, 2].max().item()

            ## visualize mesh
            if self.opts.visualize_mesh:
                mesh_save = trimesh.Trimesh(pred_v[i].cpu().numpy(), 
                                            faces[i].cpu().numpy(), 
                                            process=False, 
                                            vertex_colors=tex[i].cpu().numpy())
                mesh_save.export(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_pred_v.obj'.format(video_id[i].item(), frame_id[i].item())), file_type='obj')

            ## visualize conf
            if self.opts.visualize_conf:
                conf = match_conf[i].repeat(3,1,1).permute(1,2,0).cpu().numpy()
                conf = (conf - conf.min()) / (conf.max() - conf.min()) * 255
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_conf.png'.format(video_id[i].item(), frame_id[i].item())), conf)   
            
            ## visualize match
            if self.opts.visualize_match:
                match_vis = match[i].permute(1,2,0).cpu().numpy()
                match_vis[:, :, 0] = (match_vis[:, :, 0] - minx) / (maxx - minx)
                match_vis[:, :, 1] = (match_vis[:, :, 1] - miny) / (maxy - miny)
                match_vis[:, :, 2] = (match_vis[:, :, 2] - minz) / (maxz - minz)
                match_vis *= 255.
                img_vis = img[i].permute(1,2,0).cpu().numpy() * 255
                img_vis = np.flip(img_vis, axis=-1)
    
                match_vis = cv2.resize(match_vis, (2*length[i, 0], 2*length[i, 1]), interpolation=cv2.INTER_LINEAR)
                img_vis = cv2.resize(img_vis, (2*length[i, 0], 2*length[i, 1]), interpolation=cv2.INTER_LINEAR)
                match_orig_vis = img_orig.copy()
                x1, x2, y1, y2 = center[i,0]-length[i,0], center[i,0]+length[i,0], center[i,1]-length[i,1], center[i,1]+length[i,1]
                if x1 < 0:
                    match_vis = match_vis[:, -x1:]
                    img_vis = img_vis[:, -x1:]
                    x1 = 0
                if x2 > w-1:
                    match_vis = match_vis[:, :-(x2-w+1)]
                    img_vis = img_vis[:, :-(x2-w+1)]
                    x2 = w-1
                if y1 < 0:
                    match_vis = match_vis[-y1:]
                    img_vis = img_vis[-y1:]
                    y1 = 0
                if y2 > h-1:
                    match_vis = match_vis[:-(y2-h+1)]
                    img_vis = img_vis[:-(y2-h+1)]
                    y2 = h-1
                mix_ratio = 0.7
                dim_ratio = 1
                match_orig_vis[y1:y2, x1:x2] = match_orig_vis[y1:y2, x1:x2] * (1-mix_ratio) + match_vis * mix_ratio
                match_orig_vis = match_orig_vis * mask_orig + img_orig * (1 - mask_orig) * dim_ratio
                
                if self.opts.match_with_bbox:
                    bbox_i_proj = bbox_i.copy()
                    dir_pts_proj = dir_pts.copy()
                    bbox_i_proj[:, 0] = pp[i, 0] + bbox_i[:, 0] * foc[i, 0] / bbox_i[:, 2]
                    bbox_i_proj[:, 1] = pp[i, 1] + bbox_i[:, 1] * foc[i, 1] / bbox_i[:, 2]
                    dir_pts_proj[:, 0] = pp[i, 0] + dir_pts[:, 0] * foc[i, 0] / dir_pts[:, 2]
                    dir_pts_proj[:, 1] = pp[i, 1] + dir_pts[:, 1] * foc[i, 1] / dir_pts[:, 2]
                    proj = []
                    for j in range(bbox_i_proj.shape[0]):
                        proj.append((int(bbox_i_proj[j, 0]), int(bbox_i_proj[j, 1])))
                    dir_proj = []
                    for j in range(dir_pts_proj.shape[0]):
                        dir_proj.append((int(dir_pts_proj[j, 0]), int(dir_pts_proj[j, 1])))
                    match_orig_vis = draw_bboxes(match_orig_vis, proj, dir_proj)
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_match.png'.format(video_id[i].item(), frame_id[i].item())), match_orig_vis)
        
            ## visualize imatch
            if self.opts.visualize_imatch:
                imatch_vis = img_orig.copy()
                imatch_i = imatch[i].T.detach().cpu().numpy()
                x1, x2, y1, y2 = center[i,0]-length[i,0], center[i,0]+length[i,0], center[i,1]-length[i,1], center[i,1]+length[i,1]
                
                for pi, point in enumerate(imatch_i):  
                    # for imatch_gt (projected vertex locations), (x,y), x: right, y: up, center: 0,0
                    point[0] = (point[0]+1) * (x2-x1) * 0.5 + x1
                    point[1] = (point[1]+1) * (y2-y1) * 0.5 + y1
                    color = (int((pred_v[0, pi, 2] - minz) / (maxz - minz) * 255), \
                             int((pred_v[0, pi, 1] - miny) / (maxy - miny) * 255), \
                             int((pred_v[0, pi, 0] - minx) / (maxx - minx) * 255))
                    cv2.circle(imatch_vis, (int(point[0]), int(point[1])), 4, color, -1)
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_imatch.png'.format(video_id[i].item(), frame_id[i].item())), imatch_vis)

            ## visualize gt bbox
            if self.opts.eval and self.opts.visualize_gt:
                ## visualize gt bbox
                img_gt = copy.deepcopy(img_orig)
                bbox_gt_i = box.Box.from_transformation(rot_gt[i], trans_gt[i], scale_gt[i])
                bbox_gt_i = bbox_gt_i.vertices
                draw_bboxes_3d(
                    savedir=os.path.join(self.opts.vis_path, '{:03d}_{:03d}_3d.png'.format(video_id[i].item(), frame_id[i].item())),
                    boxes=[bbox_i, bbox_gt_i])
                bbox_gt_i[:, 0] = pp[i, 0] + bbox_gt_i[:, 0] * foc[i, 0] / bbox_gt_i[:, 2]
                bbox_gt_i[:, 1] = pp[i, 1] + bbox_gt_i[:, 1] * foc[i, 1] / bbox_gt_i[:, 2]
                proj_gt = []
                for j in range(bbox_gt_i.shape[0]):
                    proj_gt.append((int(bbox_gt_i[j, 0]), int(bbox_gt_i[j, 1])))
                img_gt = draw_bboxes(img_gt, proj_gt, color=(0,255,0))
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_gt.png'.format(video_id[i].item(), frame_id[i].item())), img_gt)  # "X" cross is on x axis
                
                depth_orig = cv2.imread(self.dataset.depthlist[video_id[i].item()][frame_id[i].item()], -1) * 1.0
                depth_orig = (depth_orig - depth_orig.min()) / (depth_orig.max() - depth_orig.min()) * 255.0
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_depth_gt.png'.format(video_id[i].item(), frame_id[i].item())), depth_orig)  # "X" cross is on x axis

            ## visualize bbox
            if self.opts.visualize_bbox:
                bbox_i_proj = bbox_i.copy()
                dir_pts_proj = dir_pts.copy()
                bbox_i_proj[:, 0] = pp[i, 0] + bbox_i[:, 0] * foc[i, 0] / bbox_i[:, 2]
                bbox_i_proj[:, 1] = pp[i, 1] + bbox_i[:, 1] * foc[i, 1] / bbox_i[:, 2]
                dir_pts_proj[:, 0] = pp[i, 0] + dir_pts[:, 0] * foc[i, 0] / dir_pts[:, 2]
                dir_pts_proj[:, 1] = pp[i, 1] + dir_pts[:, 1] * foc[i, 1] / dir_pts[:, 2]
                proj = []
                for j in range(bbox_i_proj.shape[0]):
                    proj.append((int(bbox_i_proj[j, 0]), int(bbox_i_proj[j, 1])))
                dir_proj = []
                for j in range(dir_pts_proj.shape[0]):
                    dir_proj.append((int(dir_pts_proj[j, 0]), int(dir_pts_proj[j, 1])))
                img_orig = draw_bboxes(img_orig, proj, dir_proj)
                cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_bbox.png'.format(video_id[i].item(), frame_id[i].item())), img_orig)

            ## visualize depth and tex render
            if self.opts.visualize_depth or self.opts.visualize_tex or self.opts.visualize_mask:
                pp[i, 0] = pp[i, 0] / (w/2.) - 1. 
                pp[i, 1] = pp[i, 1] / (h/2.) - 1.
                foc[i, 0] = foc[i, 0] / (w/2.)
                foc[i, 1] = foc[i, 1] / (h/2.)
                verts_i = verts[i]
                verts_i[:, 0] = pp[i, 0] + verts_i[:, 0] * foc[i, 0] / verts_i[:, 2]
                verts_i[:, 1] = pp[i, 1] + verts_i[:, 1] * foc[i, 1] / verts_i[:, 2]
                verts_i[:, 1] *= -1
                renderer = sr.SoftRenderer(image_size=h, sigma_val=1e-12, gamma_val=1e-4,
                        camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                        light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
                renderer = sr.SoftRenderer(image_size=h, sigma_val=1e-4, gamma_val=1e-4,
                       camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
                verts_i = torch.tensor(verts_i)[None].cuda()
                faces_i = faces[i][None]

                if self.opts.visualize_depth:
                    depth_render = renderer.render_mesh(sr.Mesh(verts_i, faces_i, verts_i, texture_type=self.texture_type))
                    depth_mask = depth_render[:, 3]
                    depth_render = depth_render[:, 2]
                    try:
                        depth_render[depth_mask == 0] = depth_render[depth_mask > 0].max() * 1.1
                    except:
                        pass
                    depth_render = depth_render[0, :, :, None].repeat(1,1,3).cpu().numpy()
                    try:
                        depth_render = (depth_render - depth_render.min()) / (depth_render.max() - depth_render.min()) * 255
                    except:
                        pass
                    depth_render = cv2.resize(depth_render, (w,h))
                    cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_depth.png'.format(video_id[i].item(), frame_id[i].item())), depth_render)

                if self.opts.visualize_tex:
                    tex_i = tex[i][None]
                    renderer.rasterizer.background_color = [255, 255, 255]
                    tex_render = renderer.render_mesh(sr.Mesh(verts_i, faces_i, tex_i, texture_type=self.texture_type))
                    tex_render = tex_render[:, :3]
                    tex_render = tex_render[0].permute(1,2,0).flip(-1).cpu().numpy() * 255.
                    tex_render = cv2.resize(tex_render, (w,h))

                    cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_tex.png'.format(video_id[i].item(), frame_id[i].item())), tex_render)
                
                if self.opts.visualize_mask:
                    mask_render = renderer.render_mesh(sr.Mesh(verts_i, faces_i, verts_i, texture_type=self.texture_type))[:, 3]
                    mask_render = mask_render[0,:,:,None].repeat(1,1,3).cpu().numpy() * 255.
                    mask_render = cv2.resize(mask_render, (w,h))
                    cv2.imwrite(os.path.join(self.opts.vis_path, '{:03d}_{:03d}_mask.png'.format(video_id[i].item(), frame_id[i].item())), mask_render)


