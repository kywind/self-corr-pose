from cgitb import reset
import torch
import torch.nn.functional as F
import soft_renderer as sr

from model.util.loss_utils import render, pinhole_cam


class Renderer:

    def __init__(self, opts, mesh):
        self.opts = opts
        self.renderer_mask = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-4,
                       camera_mode='look_at', perspective=False, aggr_func_rgb='hard',
                       light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
        self.renderer_depth = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-4,
                       camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
        self.renderer_softtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-3, gamma_val=1e-2,
                       camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
        self.renderer_hardtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-3, 
                       camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        self.renderer_depth.rasterizer.background_color = [1, 1, 1]
        self.renderer_softtex.rasterizer.background_color = [1, 1, 1]
        self.mesh = mesh
    
    def render_mean_mesh(self, foc_crop, pp_crop, rotation, translation):
        bsz = rotation.shape[0]
        mean_v = self.mesh.mean_v[None].repeat(bsz, 1, 1)
        faces = self.mesh.faces[None].repeat(bsz, 1, 1)
        mean_v_render = render(self.renderer_depth, mean_v, faces, None, \
                foc_crop, pp_crop, rotation, translation, \
                rotation_detach=True, translation_detach=True, render_depth=True)
        return mean_v_render

    def render_all(self, pred_v, faces, tex, foc_crop, pp_crop, rotation, translation, scale):
        mask_render = render(self.renderer_mask, pred_v, faces, None, foc_crop, pp_crop, rotation, translation, \
                        rotation_detach=False, translation_detach=False, render_mask=True, texture_type='vertex')[:, -1]

        if tex is not None:
            tex_render = render(self.renderer_softtex, pred_v, faces, tex, foc_crop, pp_crop, rotation, translation, \
                            rotation_detach=False, translation_detach=False, render_mask=False, texture_type=self.mesh.texture_type)
            tex_mask = tex_render[:, -1]
            tex_render = tex_render[:, :3]
        else:
            tex_mask = None
            tex_render = None
        
        depth_render = render(self.renderer_depth, pred_v, faces, None, foc_crop, pp_crop, rotation, translation, \
                        rotation_detach=False, translation_detach=False, render_depth=True, texture_type='vertex')
        if not self.opts.use_depth: depth_render = depth_render.detach()
        depth_mask = depth_render[:, 3]
        depth_render = depth_render[:, 2].clone()

        # with torch.no_grad():
        match_gt = render(self.renderer_hardtex, pred_v.detach(), faces, pred_v.detach(), \
                    foc_crop, pp_crop, rotation, translation, texture_type='vertex')
        match_mask = match_gt[:, -1]
        match_gt = match_gt[:, :3]
        
        # with torch.no_grad():
        imatch_gt = pred_v.detach().bmm(rotation) + translation
        imatch_depth = imatch_gt[:, :, 2].clone()
        imatch_gt = pinhole_cam(imatch_gt, pp_crop, foc_crop)
        imatch_gt = imatch_gt[:, :, :2].permute(0, 2, 1)  # b,2,n
        
        imatch_depth_gt = F.grid_sample(depth_render[:, None], imatch_gt.permute(0, 2, 1)[:, None], align_corners=False)[:, 0, 0]  # b,1,h,w & b,1,n,2 -> b,1,1,n
        depth_weight = -F.relu(imatch_depth - imatch_depth_gt)
        depth_weight = (5 * depth_weight).exp().detach()

        return mask_render, tex_render, depth_render, match_gt, imatch_gt, tex_mask, depth_mask, match_mask, depth_weight
        



