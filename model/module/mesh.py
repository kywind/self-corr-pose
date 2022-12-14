from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import subprocess
import soft_renderer as sr
import pytorch3d
import pytorch3d.structures
import pytorch3d.loss
import pytorch3d.ops

from model.module.surface_texture import SurfaceTexture
from model.util.symmetry import get_symm_rots
from model.util.chamfer import chamfer_distance_single_way


flags.DEFINE_integer('symmetry_idx', -1, 'symmetry index: -1: none, 0: y_rot, 1: x')
flags.DEFINE_list('init_scale', [1,1,1], 'initial scale of x')
flags.DEFINE_bool('shape_prior', False, 'use shape prior, else use ellipsoid initialization')
flags.DEFINE_string('shape_prior_path', '', 'path of shape prior')
flags.DEFINE_bool('prior_deform', False, 'deform shape prior (False: fix shape prior)')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')
flags.DEFINE_integer('n_faces', 1280, 'number of faces for remeshing')


class CanonicalMesh(nn.Module):

    def __init__(self, opts):
        super(CanonicalMesh, self).__init__()
        self.opts = opts
    
        self.mean_v, self.faces, self.symm_rots = self.init_shape()
        self.num_verts = self.mean_v.shape[0]
        self.num_faces = self.faces.shape[0]

        if opts.surface_texture: 
            self.texture_type = 'surface'
            self.surface_texture = SurfaceTexture(opts)
        else: 
            self.texture_type = 'vertex'
    
    
    def get_texture(self, pred_v, faces, imatch, img):
        if self.texture_type == 'surface':
            tex = self.surface_texture.get_texture(pred_v, faces, imatch, img)
        else:
            tex = F.grid_sample(img, imatch.permute(0, 2, 1)[:, None], align_corners=False)[:, :, 0].permute(0, 2, 1)  # b,3,h,w & b,1,n,2 -> b,3,1,n --> b,n,3
        return tex
    
    def compute_symmetry_loss(self, pred_v, faces):
        bsz = pred_v.shape[0]
        npts = 10000
        pred_v_symm = pred_v.clone()[:, None].repeat(1, self.symm_rots.shape[0], 1, 1).reshape(self.symm_rots.shape[0] * bsz, self.num_verts, 3)  # k*bsz,N,3
        faces_symm = faces.clone()[:, None].repeat(1, self.symm_rots.shape[0], 1, 1).reshape(self.symm_rots.shape[0] * bsz, self.num_faces, 3)
        sample_pts, _ = pytorch3d.ops.sample_points_from_meshes(pytorch3d.structures.Meshes(pred_v_symm, faces_symm), npts, return_normals=True)
        symm_rots = self.symm_rots[None].repeat(bsz, 1, 1, 1).reshape(self.symm_rots.shape[0] * bsz, 3, 3)  # k*bsz,N,3
        sample_pts_rot = sample_pts.bmm(symm_rots)
        symmetry_loss = chamfer_distance_single_way(pred_v_symm, sample_pts_rot)[0]
        return symmetry_loss

    def init_shape(self):
        opts = self.opts
        if opts.shape_prior:
            prior = trimesh.load_mesh(opts.shape_prior_path)
            verts = torch.tensor(np.array(prior.vertices)).float()
            faces = torch.tensor(np.array(prior.faces))
            verts -= verts.mean(0)
            verts /= verts.abs().max()
            verts[:, 0] *= float(opts.init_scale[0])
            verts[:, 1] *= float(opts.init_scale[1])
            verts[:, 2] *= float(opts.init_scale[2])
            if opts.symmetry_idx == 0:
                division = 17  # for the symmetry loss
                symm_rots = get_symm_rots(division)
            elif opts.symmetry_idx == 1:
                division = 2
                symm_rots = torch.zeros(division, 3, 3)
                symm_rots[0] = torch.eye(3)
                symm_rots[1] = torch.tensor(
                    [[-1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]
                )
            else:
                division = 1
                symm_rots = torch.eye(3)[None]
            verts = nn.Parameter(torch.Tensor(verts), requires_grad=opts.prior_deform)  # (num_verts, 3)
            faces = nn.Parameter(torch.LongTensor(faces), requires_grad=False)  # (num_faces, 3)
            symm_rots = nn.Parameter(symm_rots, requires_grad=False)
        else:  # sphere prior
            prior = trimesh.creation.icosphere(subdivisions=opts.subdivide, radius=1.0, color=None)
            verts = torch.tensor(np.array(prior.vertices)).float()
            faces = torch.tensor(np.array(prior.faces))
            verts[:, 0] *= opts.x_scale
            verts[:, 1] *= opts.y_scale
            verts[:, 2] *= opts.z_scale
            if opts.symmetry_idx == 0:
                division = 17
                symm_rots = get_symm_rots(division)
            elif opts.symmetry_idx == 1:
                division = 2
                symm_rots = torch.zeros(division, 3, 3)
                symm_rots[0] = torch.eye(3)
                symm_rots[1] = torch.tensor(
                    [[-1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]
                )
            else:
                division = 1
                symm_rots = torch.eye(3)[None]
            verts = nn.Parameter(torch.Tensor(verts), requires_grad=True)  # (num_verts, 3)
            faces = nn.Parameter(torch.LongTensor(faces), requires_grad=False)  # (num_faces, 3)
            symm_rots = nn.Parameter(symm_rots, requires_grad=False)
        return verts, faces, symm_rots
    

    def resample_meanv(self):
        n_faces = self.faces.shape[0]
        sr.Mesh(self.mean_v, self.faces).save_obj('temp/input.obj')
        subprocess.run(['/data/zhangkaifeng/self-pose/third-party/Manifold/build/manifold', 'temp/input.obj', 'temp/output.obj', '10000'])
        subprocess.run(['/data/zhangkaifeng/self-pose/third-party/Manifold/build/simplify', '-i', 'temp/output.obj', '-o', 'temp/simple.obj', '-m', '-f', str(n_faces)])
        new_mesh = sr.Mesh.from_obj('temp/simple.obj')
        self.mean_v.data = new_mesh.vertices[0]
        self.faces.data  = new_mesh.faces[0]
        self.num_verts = self.mean_v.shape[0]
        self.num_faces = self.faces.shape[0]
        print('resampled: mean_v shape', self.mean_v.shape, 'faces shape', self.faces.shape)

