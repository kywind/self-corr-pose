from absl import flags
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.transforms import InterpolationMode

from model.util.loss_utils import divide_by_frame, divide_by_instance


flags.DEFINE_float('tau_img', 10, 'tau of image')
flags.DEFINE_float('tau_mesh', 10, 'tau of mesh')

flags.DEFINE_integer('topk_img', 100, 'number of k top pixels')
flags.DEFINE_integer('topk_mesh', 100, 'number of k top vertices')

flags.DEFINE_integer('corr_h', 32, 'correspondence map height')
flags.DEFINE_integer('corr_w', 32, 'correspondence map width')


class Correspondence:

    def __init__(self, opts):
        self.opts = opts
        self.tau_img = opts.tau_img
        self.tau_mesh = opts.tau_mesh
        self.k_img = opts.topk_img
        self.k_mesh = opts.topk_mesh
        self.hf = opts.corr_h
        self.wf = opts.corr_w
        meshgrid = torch.Tensor(np.array(np.meshgrid(range(self.wf), range(self.hf)))).cuda().reshape(2, -1) + 0.5 # 2,h*w
        meshgrid = meshgrid / (self.wf / 2) - 1
        self.meshgrid = meshgrid  # 2,h*w


    def match(self, img_feat, mesh_feat, mask, pred_v):
        bsz, h, w = mask.shape
        opts = self.opts

        mask_down = (F.interpolate(mask[:, None], (self.hf, self.wf), mode='bilinear') * 0.5).reshape(bsz, -1) * 1.0  # b,h*w

        pointcorr = mesh_feat.bmm(img_feat)  # b,n,h*w
        pointcorr = pointcorr.permute(0,2,1)  # b,h*w,n
        pointcorr = pointcorr * (mask_down[:,:,None] > 0) - 1e5 * (mask_down[:,:,None] == 0)  # b,h*w,n

        ## old softmax way
        pointcorr_mesh = torch.softmax(self.tau_mesh * pointcorr, dim=1) # b,h*w,n, corres. pixel for each 3d point
        pointcorr_img = torch.softmax(self.tau_img * pointcorr, dim=2)  # b,h*w,n, corres. 3d point for each pixel
            
        ## get soft correspondence
        meshgrid = self.meshgrid[None].repeat(bsz,1,1)  # b,2,h*w
        imatch = meshgrid.bmm(pointcorr_mesh)  # b,2,k_mesh
        match = (pointcorr_img[:, :, :, None] * pred_v.detach()[:, None, :, :]).sum(2)  # b,k_img,3
        
        if opts.train:
            match_conf = None
        else:  # use conf only during evaluation
            with torch.no_grad():
                dis3d = (match[:, None] - pred_v[:, :, None]).norm(2, -1)  # b,n,h*w,3 -> b,n,h*w
                dis3d = dis3d.argmin(1).view(bsz, -1)  # b,h*w (entry values are vertex indices)
                fberr = torch.zeros(bsz, 1, self.hf, self.wf).cuda()
                for i in range(bsz):
                    ipred_i = imatch[i].permute(1, 0)[dis3d[i]]  # (h*w, 2)
                    fberr[i, 0] = (self.meshgrid.permute(1, 0) - ipred_i).norm(2, -1).view(self.hf, self.wf)
                match_conf = (-5 * fberr).exp() 
            match_conf = F.interpolate(match_conf, (h,w), mode='bilinear', align_corners=False).detach()
            conf_mean = match_conf[mask[:, None] > 0].mean().item()
            conf_mean = min(conf_mean, 0.5)
            match_conf[match_conf < conf_mean] = 0 # ~10px

        match = F.interpolate(match.reshape(bsz,self.hf,self.wf,3).permute(0,3,1,2), (h,w), mode='nearest')  # b,3,h,w
        
        return pointcorr, match, imatch, match_conf
    

    def compute_rotation_cycle_loss(self, src_img, src_mask, src_img_feat, encoder):
        bsz = src_img.shape[0]
        min_angle = 0
        max_angle = 360
        # min_scale = 0.8
        # max_scale = 1.2
        angle = torch.empty(1).uniform_(float(min_angle), float(max_angle)).item()
        grid = self.meshgrid.reshape(2, self.hf, self.wf)[None].repeat(bsz,1,1,1)
        grid = F.interpolate(grid, (self.hf//2, self.wf//2), mode='bilinear')

        src_mask = src_mask[:, None]
        tgt_img = torchvision.transforms.functional.rotate(src_img, angle, interpolation=InterpolationMode.BILINEAR)
        tgt_mask = torchvision.transforms.functional.rotate(src_mask, angle, interpolation=InterpolationMode.NEAREST)
        cycle_match_gt = torchvision.transforms.functional.rotate(grid, angle, interpolation=InterpolationMode.NEAREST).reshape(bsz,2,-1)

        _, tgt_img_feat = encoder.encode_img(tgt_img)  # b,c,h,w
        tgt_img_feat = tgt_img_feat.reshape(bsz, self.opts.n_corr_feat, -1)  # b,c,h*w
        tgt_img_feat = F.normalize(tgt_img_feat, 2, 1)  # b,c,h*w

        src_mask_down = (F.interpolate(src_mask, (self.hf//2, self.wf//2), mode='bilinear') > 0.5).reshape(bsz, -1) * 1.0  # b,h*w
        tgt_mask_down = (F.interpolate(tgt_mask, (self.hf//2, self.wf//2), mode='bilinear') > 0.5).reshape(bsz, -1) * 1.0  # b,h*w

        mask_down = src_mask_down[:,:,None] * tgt_mask_down[:,None,:]

        tgt_img_feat = tgt_img_feat.reshape(*tgt_img_feat.shape[:2], self.hf, self.wf)  # b,c,h,w
        src_img_feat = src_img_feat.reshape(*src_img_feat.shape[:2], self.hf, self.wf)  # b,c,h,w
        tgt_img_feat = F.interpolate(tgt_img_feat, (self.hf//2, self.wf//2), mode='bilinear').reshape(*tgt_img_feat.shape[:2], -1) * 1.0  # b,h*w
        src_img_feat = F.interpolate(src_img_feat, (self.hf//2, self.wf//2), mode='bilinear').reshape(*src_img_feat.shape[:2], -1) * 1.0  # b,h*w

        pointcorr = src_img_feat.permute(0,2,1).bmm(tgt_img_feat)  # b,h*w,h*w(tgt)
        pointcorr = pointcorr * (mask_down > 0) - 1e5 * (mask_down == 0)  # b,h*w,h*w
        pointcorr_tgt = torch.softmax(self.tau_mesh * pointcorr, dim=1) # b,h*w,h*w(tgt)

        grid = grid.reshape(bsz,2,-1)
        cycle_match = grid.bmm(pointcorr_tgt)  # b,2,h*w(tgt)

        cycle_loss = ((cycle_match - cycle_match_gt).norm(2, 1) * tgt_mask_down).mean()
        return cycle_loss, cycle_match, cycle_match_gt, tgt_mask_down


    def compute_cycle_loss(self, src_imatch, tgt_imatch, src_img_feat, tgt_img_feat, \
                        src_mask, tgt_mask, src_depth_weight, tgt_depth_weight):
        src_imatch = src_imatch.detach()
        tgt_imatch = tgt_imatch.detach()
        bsz = src_mask.shape[0]
        grid = self.meshgrid.reshape(2, self.hf, self.wf)[None].repeat(bsz,1,1,1)
        grid = F.interpolate(grid, (self.hf//2, self.wf//2), mode='bilinear')
        
        src_mask = src_mask[:, None]
        tgt_mask = tgt_mask[:, None]
        src_mask_down = (F.interpolate(src_mask, (self.hf//2, self.wf//2), mode='bilinear') > 0.5).reshape(bsz, -1) * 1.0  # b,h*w
        tgt_mask_down = (F.interpolate(tgt_mask, (self.hf//2, self.wf//2), mode='bilinear') > 0.5).reshape(bsz, -1) * 1.0  # b,h*w
        mask_down = src_mask_down[:,:,None] * tgt_mask_down[:,None,:]

        tgt_img_feat = tgt_img_feat.reshape(*tgt_img_feat.shape[:2], self.hf, self.wf)  # b,c,h,w
        src_img_feat = src_img_feat.reshape(*src_img_feat.shape[:2], self.hf, self.wf)  # b,c,h,w
        tgt_img_feat = F.interpolate(tgt_img_feat, (self.hf//2, self.wf//2), mode='bilinear').reshape(*tgt_img_feat.shape[:2], -1) * 1.0  # b,c,h*w
        src_img_feat = F.interpolate(src_img_feat, (self.hf//2, self.wf//2), mode='bilinear').reshape(*src_img_feat.shape[:2], -1) * 1.0  # b,c,h*w
        
        pointcorr = src_img_feat.permute(0,2,1).bmm(tgt_img_feat)  # b,h*w,h*w(tgt)
        pointcorr = pointcorr * (mask_down > 0) - 1e5 * (mask_down == 0)  # b,h*w,h*w
        pointcorr_tgt = torch.softmax(self.tau_mesh * pointcorr, dim=1) # b,h*w,h*w(tgt)

        grid = grid.reshape(bsz,2,-1)
        cycle_match = grid.bmm(pointcorr_tgt)  # b,2,h*w(tgt)

        tgt_pts = F.grid_sample(cycle_match.reshape(bsz,2,self.hf//2,self.wf//2), \
                tgt_imatch.permute(0,2,1)[:,None], align_corners=False)[:,:,0]  # b,2,h,w & b,1,n,2 -> b,2,1,n --> b,2,n
        
        depth_weight = src_depth_weight * tgt_depth_weight
        cycle_loss = ((src_imatch - tgt_pts).norm(2, 1) * depth_weight).mean()
        return cycle_loss, tgt_pts, src_imatch, depth_weight
    
        
