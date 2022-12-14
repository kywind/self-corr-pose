# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Loss Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soft_renderer as sr

import pytorch3d
import pytorch3d.structures
import pytorch3d.loss
import pytorch3d.ops

from model.util.chamfer import chamfer_distance_single_way


def pinhole_cam(verts, pp, foc):
    if len(verts.shape) == 3:
        verts[:, :, 1] = pp[:, 1][:, None] + verts[:, :, 1].clone() * foc[:, 1][:, None] / verts[:, :, 2].clone()
        verts[:, :, 0] = pp[:, 0][:, None] + verts[:, :, 0].clone() * foc[:, 0][:, None] / verts[:, :, 2].clone()
    elif len(verts.shape) == 2:
        verts[:, 1] = pp[1] + verts[:, 1].clone() * foc[1] / verts[:, 2].clone()
        verts[:, 0] = pp[0] + verts[:, 0].clone() * foc[0] / verts[:, 2].clone()
    else:
        raise ValueError("vertices shape must be (bsz, N, 3) or (N, 3).")
    return verts

def render(renderer, verts, faces, tex, foc, pp, rotation, translation, \
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

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            if laplacian[i, i]!=0: laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        x = x.pow(2)
        
        dims = tuple(range(x.ndimension())[1:])
        x = x.sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x

class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average
        
        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))
        
        vert_face = {}
        for k,v in enumerate(faces):
            for vx in v:
                if vx not in vert_face.keys():
                    vert_face[vx] = [k]
                else:
                    vert_face[vx].append(k)

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            #for face in faces:
            for faceid in sorted(list(set(vert_face[v0]) & set(vert_face[v1]))):
                face = faces[faceid]
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss

class ARAPLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(ARAPLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        diffdx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        for i in range(3):
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:,:,i])) # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:,:,i:i+1])
            
            x_sub = self.laplacian.matmul(torch.diag_embed(x[:,:,i])) # N, Nv, Nv)
            x_diff = (x_sub - x[:,:,i:i+1])
            
            diffdx += (dx_diff).pow(2)
            diffx +=   (x_diff).pow(2)

        diff = (diffx-diffdx).abs()
        diff = torch.stack([diff[i][self.laplacian.bool()].mean() for i in range(x.shape[0])])
        #diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return diff

def mesh_area(vs,faces):
    v1 = vs[faces[:, 1]] - vs[faces[:, 0]]
    v2 = vs[faces[:, 2]] - vs[faces[:, 0]]
    area = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    return area

def compute_camera_loss(m1, m2): # compute geodesic distance from two matrices
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1) / 2
    cos = torch.nn.functional.hardtanh(cos, -1, 1)
    theta = torch.acos(cos)
    return theta

def compute_mask_loss(img, mask, mask_pred):
    bsz = img.shape[0]
    mask_loss_sub = 0
    for i in range(5):  # 256,128,64,32,16
        diff_img = (F.interpolate(mask_pred, scale_factor=0.5**i, mode='area', recompute_scale_factor=False)
                  - F.interpolate(mask, scale_factor=0.5**i, mode='area', recompute_scale_factor=False)).pow(2)
        mask_loss_sub += F.interpolate(diff_img[:, None], mask_pred.shape[1:], mode='area')[:, 0]
    mask_loss_sub = 0.2 * mask_loss_sub.mean((1, 2))
    return mask_loss_sub

def compute_texture_loss(img, mask, tex_pred, tex_mask):
    img_gt = img * (mask > 0).float()[:, None]  # ground truth
    tex_pred_black = tex_pred * tex_mask[:, None]
    img_gt_white = 1 - (mask > 0).float()[:, None] + img_gt
    texture_loss_sub = 0.75 * ((img_gt - tex_pred_black).pow(2).sum(1)*1).mean((1,2))
    texture_loss_sub += ((img_gt_white - tex_pred).abs().mean(1)*1).mean((1,2))
    return texture_loss_sub

def compute_mask_loss_with_occ(img, mask, mask_pred, occ):
    mask_loss_sub = 0
    for i in range(5):  # 256,128,64,32,16
        diff_img = (F.interpolate(mask_pred, scale_factor=0.5**i, mode='area', recompute_scale_factor=False)
                  - F.interpolate(mask, scale_factor=0.5**i, mode='area', recompute_scale_factor=False)).pow(2)
        mask_loss_sub += F.interpolate(diff_img[:, None], mask_pred.shape[1:], mode='area')[:, 0]
    mask_loss_sub *= (1. - occ)
    mask_loss_sub = 0.2 * mask_loss_sub.mean((1, 2))
    return mask_loss_sub

def compute_texture_loss_with_occ(img, mask, tex_pred, tex_mask, occ):
    img_gt = img * (mask > 0).float()[:, None]  # ground truth
    tex_pred_black = tex_pred * tex_mask[:, None]
    img_gt_white = 1 - (mask > 0).float()[:, None] + img_gt
    texture_loss_sub = 0.75 * (img_gt - tex_pred_black).pow(2).sum(1) + (img_gt_white - tex_pred).abs().mean(1)
    texture_loss_sub *= (1. - occ)
    texture_loss_sub = texture_loss_sub.mean((1,2))
    return texture_loss_sub

def compute_depth_loss(depth, depth_pred, depth_mask, mask):
    depth_loss_mask = mask * depth_mask
    depth_loss_mask = depth_loss_mask.detach()
    depth_scale = depth_pred[depth_mask != 0].mean() / depth[mask * depth != 0].mean()
    depth_diff = depth_pred - depth_scale * depth
    depth_diff[depth_loss_mask == 0] = 0
    depth_diff[depth == 0] = 0
    thresh = 1.
    depth_loss_sub = depth_diff.pow(2)
    depth_loss_sub = thresh - torch.relu(thresh - depth_loss_sub)
    depth_loss_sub = depth_loss_sub.mean((1, 2))
    return depth_loss_sub, depth_diff

def compute_depth_loss_chamfer(pred_v, faces, depth, depth_pred, depth_mask, mask, pp_crop, foc_crop, rotation, translation):
    with torch.no_grad():
        depth_loss_mask = mask * depth_mask
        depth_loss_mask = depth_loss_mask.detach()
        depth_scale = depth_pred[depth_mask != 0].mean() / depth[mask * depth != 0].mean()
        depth *= depth_scale
        depth_diff = depth_pred - depth
        depth_diff[depth_loss_mask == 0] = 0
        depth_diff[depth == 0] = 0
        point_cloud = depth_to_point_cloud(depth, pp_crop, foc_crop).float()
    npts = 2000
    point_cloud = (point_cloud - translation).bmm(rotation.permute(0,2,1))
    # pred_v = pred_v.bmm(rotation) + translation
    point_cloud_pred, _ = pytorch3d.ops.sample_points_from_meshes(pytorch3d.structures.Meshes(pred_v, faces), npts, return_normals=True)
    depth_loss_sub = chamfer_distance_single_way(point_cloud, point_cloud_pred, point_reduction=None, batch_reduction=None)[0]
    depth_loss_sub = depth_loss_sub.reshape(mask.shape)
    depth_loss_sub[mask == 0] = 0
    depth_loss_sub[depth == 0] = 0
    return depth_loss_sub.mean((1,2)), depth_diff

def depth_to_point_cloud(depth, pp, foc):
    b, h, w = depth.shape
    u, v = np.meshgrid(range(w), range(h)) + 0.5
    u = torch.tensor(u*2/w-1, dtype=torch.float32, device=depth.device)
    v = torch.tensor(v*2/h-1, dtype=torch.float32, device=depth.device)
    Z = depth
    X = (u[None] - pp[:, 0][:, None, None]) * Z / foc[:, 0][:, None, None]
    Y = (v[None] - pp[:, 1][:, None, None]) * Z / foc[:, 1][:, None, None]
    pc = torch.stack((X, Y, Z), dim=-1).reshape(b, -1, 3)
    return pc

def compute_match_loss(match, match_gt, match_mask, mask):
    match_mask = (match_mask > 0) & (mask > 0)
    match_loss_sub = ((match - match_gt).norm(2, 1) * match_mask).mean((1,2))
    return match_loss_sub

def compute_imatch_loss(imatch, imatch_gt, depth_weight):
    imatch_loss_sub = ((imatch - imatch_gt).norm(2, 1) * depth_weight).mean(1)
    return imatch_loss_sub

def divide_by_frame(x, batch_size, repeat):
    src = x.reshape(batch_size, repeat, *x.shape[1:])
    tgt = torch.cat([src[:, 1:], src[:, :1]], dim=1)
    src = src.reshape(-1, *src.shape[2:])
    tgt = tgt.reshape(-1, *tgt.shape[2:])
    return src, tgt

def divide_by_instance(x, batch_size, repeat):
    src = x.reshape(batch_size, repeat, *x.shape[1:])
    tgt = torch.cat([src[1:], src[:1]], dim=0)
    src = src.reshape(-1, *src.shape[2:])
    tgt = tgt.reshape(-1, *tgt.shape[2:])
    return src, tgt

def divide_by_both(x, batch_size, repeat):
    src_frame, tgt_frame = divide_by_frame(x, batch_size, repeat)
    src_instance, tgt_instance = divide_by_instance(x, batch_size, repeat)
    src = torch.cat([src_frame, src_instance], dim=0)
    tgt = torch.cat([tgt_frame, tgt_instance], dim=0)
    return src, tgt

