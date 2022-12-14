import torch
import torch.nn.functional as F
import numpy as np


class SurfaceTexture:

    def __init__(self, opts):
        n = opts.n_tex_sample
        self.n = n
        xx = torch.zeros(n**2).cuda()
        yy = torch.arange((2*n-1)/(2.0*n), 0, step=-1.0/n)[None].cuda().repeat(n,1).reshape(-1)
        for i in range(n):
            xx[i*n:(i+1)*n] = (2*i+1) / (2.0*n)
            yy[i*n:(i+1)*n] -= i / (1.0*n)
        xx[yy < 0] = 1 - xx[yy < 0]
        yy[yy < 0] *= -1
        self.xx = xx.clone()  # (n**2,)
        self.yy = yy.clone()  # (n**2,)
    
    def get_texture(self, verts, faces, imatch, img):
        face_match = self.get_face_match(imatch, faces)
        tex = self.get_color(face_match, img)
        # import pdb; pdb.set_trace()
        return tex
    
    def get_color(self, match, img):
        # match: b,nf,n**2,2 --> (-1,1)*(-1,1)
        # img: b,3,h,w
        _,n_faces,n_sample,_ = match.shape
        bsz,_,h,w = img.shape
        match = match.reshape(bsz,-1,2)
        color = F.grid_sample(img, match[:,None], align_corners=False)[:,:,0].permute(0,2,1)  # b,3,h,w & b,1,n,2 -> b,3,1,n --> b,n,3
        color = color.reshape(bsz,n_faces,n_sample,3)
        return color.contiguous()

    def get_face_match(self, match, faces):
        # faces: b,nf,3(verts)
        # match: b,2(xy),nv --> permute to b,nv,2
        # vf: b,nf,3(verts),2(xy)
        match = match.permute(0,2,1)
        bsz,nf,_ = faces.shape
        _,nv,c = match.shape
        face_match = torch.gather(match, 1, faces.reshape(bsz,-1)[:,:,None].repeat(1,1,c)).reshape(bsz,nf,3,c)
        face_match = self.subsample(face_match)
        return face_match

    def subsample(self, match):  # b,nf,3,c
        match0 = match[:,:,0]  # b,nf,c
        match10 = match[:,:,1] - match0  # b,nf,c
        match20 = match[:,:,2] - match0  # b,nf,c
        xx = match10[:,:,None] * self.xx[None,None,:,None]  # b,nf,n**2,c
        yy = match20[:,:,None] * self.yy[None,None,:,None]  # b,nf,n**2,c
        match_sample = xx + yy + match0[:,:,None]
        return match_sample

