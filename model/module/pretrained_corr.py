from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from model.module.network.dino import DINO
from model.util.mesh_utils import sample_points_from_mesh
from model.util.loss_utils import divide_by_frame, divide_by_instance, divide_by_both, pinhole_cam


flags.DEFINE_string('divide_fn', 'frame', 'choose from [frame, instance, both]')
flags.DEFINE_integer('pretrain_k', 100, 'pretrain corr top k value')


class PretrainedCorrespondence(nn.Module):

    def __init__(self, opts, mesh, pretrained=True):
        super().__init__()
        self.opts = opts
        self.mesh = mesh
        self.net = DINO().eval()  # the pretrained net
        self.img_size = opts.img_size
        self.feat_size = opts.img_size // 8
        self.tau_img = opts.tau_img
        self.tau_mesh = opts.tau_mesh
        self.k = opts.pretrain_k

        self.hf = opts.corr_h
        self.wf = opts.corr_w
        meshgrid = torch.Tensor(np.array(np.meshgrid(range(self.wf), range(self.hf)))).cuda().reshape(2, -1) + 0.5 # 2,h*w
        meshgrid = meshgrid / (self.wf / 2) - 1
        self.meshgrid = meshgrid  # 2,h*w

        for param in self.net.parameters():
            param.requires_grad = False

        if opts.divide_fn == 'frame':
            self.divide_fn = divide_by_frame
        elif opts.divide_fn == 'instance':
            self.divide_fn = divide_by_instance
        elif opts.divide_fn == 'both':
            self.divide_fn = divide_by_both
        else: raise ValueError


    def match(self, src_img, tgt_img, src_mask, tgt_mask, grid):
        # return the match
        bsz = src_img.shape[0]
        src_mask = src_mask[:, None]
        tgt_mask = tgt_mask[:, None]

        # src_feat = self.net(src_img)
        # tgt_feat = self.net(tgt_img)

        all_img = torch.cat([src_img, tgt_img], dim=0)

        MAX_BATCH_SIZE = 64
        with torch.no_grad():
            torch.cuda.empty_cache()
            if all_img.shape[0] <= MAX_BATCH_SIZE:
                all_feat = self.net(all_img)
            
            # Process in chunks to avoid CUDA out-of-memory
            else:
                num_chunks = np.ceil(all_img.shape[0] / MAX_BATCH_SIZE).astype('int')
                data_chunks = []
                for i, ims_ in enumerate(all_img.chunk(num_chunks)):
                    data_chunks.append(self.net(ims_))
                all_feat = torch.cat(data_chunks, dim=0)

        src_feat = all_feat[:bsz]
        tgt_feat = all_feat[bsz:]

        src_feat = src_feat.reshape(*src_feat.shape[:2], -1)
        tgt_feat = tgt_feat.reshape(*tgt_feat.shape[:2], -1)
        hw = src_feat.shape[-1]

        src_mask_down = (F.interpolate(src_mask, (self.feat_size, self.feat_size), mode='bilinear') > 0.5).reshape(bsz, -1) * 1.0  # b,h*w
        tgt_mask_down = (F.interpolate(tgt_mask, (self.feat_size, self.feat_size), mode='bilinear') > 0.5).reshape(bsz, -1) * 1.0  # b,h*w
        
        mask_down = src_mask_down[:,:,None] * tgt_mask_down[:,None,:]

        pointcorr = src_feat.permute(0,2,1).bmm(tgt_feat)  # b,h*w,h*w(tgt)  # TODO chunked cosine similarity
        pointcorr = pointcorr * (mask_down > 0) - 1e5 * (mask_down == 0)  # b,h*w,h*w

        pointcorr_max_bw = pointcorr.max(1).indices  # b,h*w(tgt)
        pointcorr_max_fw = pointcorr.max(2).indices  # b,h*w(src)
        pointcorr_max_cy = torch.gather(pointcorr_max_fw, -1, pointcorr_max_bw)
        grid = grid.reshape(bsz,2,-1)
        match = torch.gather(grid, -1, pointcorr_max_bw[:, None].repeat(1,2,1))  # bsz,2,h*w
        cycle = torch.gather(grid, -1, pointcorr_max_cy[:, None].repeat(1,2,1))  # bsz,2,h*w

        distance = (cycle - grid).norm(2, 1)  # bsz,h*w
        distance = distance * (tgt_mask_down > 0) + 1e5 * (tgt_mask_down == 0)  # b,h*w
        _, indices = torch.topk(-distance, k=self.k, dim=1) # b,k
        match = torch.gather(match, -1, indices[:, None].repeat(1,2,1))  # b,2,k
        grid = torch.gather(grid, -1, indices[:, None].repeat(1,2,1))  # b,2,k
        match_mask = torch.gather(tgt_mask_down, -1, indices)  # b,k

        indices_match = torch.gather(pointcorr_max_bw, -1, indices)  # b,k

        return match, grid, indices_match, indices, match_mask  # b,2,k; b,2,k; b,k; b,k


    def compute_cycle_loss(self, img, mask, depth_weight, pointcorr):  # cycle_loss_pretrain
        num_verts = pointcorr.shape[-1]
        img_src, img_tgt = self.divide_fn(img, self.opts.batch_size, self.opts.repeat)
        mask_src, mask_tgt = self.divide_fn(mask, self.opts.batch_size, self.opts.repeat)
        depth_weight_src, depth_weight_tgt = self.divide_fn(depth_weight, self.opts.batch_size, self.opts.repeat)
        pointcorr_src, pointcorr_tgt = self.divide_fn(pointcorr, self.opts.batch_size, self.opts.repeat)
        bsz = img_src.shape[0]

        grid = self.meshgrid.reshape(2, self.hf, self.wf)[None].repeat(bsz,1,1,1)
        grid = F.interpolate(grid, (self.hf//2, self.wf//2), mode='bilinear')  # b,2,h,w

        pts_src, pts_tgt, indices_src, indices_tgt, mask = self.match(img_src, img_tgt, mask_src, mask_tgt, grid)

        pointcorr_src = F.interpolate(pointcorr_src.permute(0,2,1).reshape(bsz, num_verts, self.hf, self.wf), \
                (self.hf//2, self.wf//2), mode='bilinear').reshape(bsz, num_verts, (self.hf//2)*(self.wf//2)).permute(0,2,1)  # b,h*w,n
        pointcorr_tgt = F.interpolate(pointcorr_tgt.permute(0,2,1).reshape(bsz, num_verts, self.hf, self.wf), \
                (self.hf//2, self.wf//2), mode='bilinear').reshape(bsz, num_verts, (self.hf//2)*(self.wf//2)).permute(0,2,1)  # b,h*w,n
        pointcorr_img = torch.softmax(self.tau_img * pointcorr_tgt, dim=2)  # b,h*w,n, corres. 3d point for each tgt pixel
        pointcorr_mesh = torch.softmax(self.tau_mesh * pointcorr_src, dim=1) # b,h*w,n, corres. src pixel for each 3d point

        pointcorr_img = pointcorr_img * (depth_weight_tgt[:,None] >= 0.5)
        pointcorr_mesh = pointcorr_mesh * (depth_weight_src[:,None] >= 0.5)

        corr = pointcorr_mesh.bmm(pointcorr_img.permute(0,2,1))  # b,h*w,h*w(tgt)
        corr = corr / (corr.sum(1, keepdims=True) + 1e-5)

        ## get soft correspondence
        grid = self.meshgrid.reshape(2, self.hf, self.wf)[None].repeat(bsz,1,1,1)
        grid = F.interpolate(grid, (self.hf//2, self.wf//2), mode='bilinear').reshape(bsz, 2, -1)  # b,2,h,w
        match = grid.bmm(corr)  # b,2,h*w
        match = torch.gather(match, -1, indices_tgt[:,None].repeat(1,2,1))
        
        cycle_loss = ((match - pts_src).norm(2,1) * mask).mean()
        return cycle_loss, pts_src, pts_tgt, match, mask, img_src, img_tgt


