import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.module.network.image_encoder import ResNet_Encoder, ResNet_Decoder
from model.module.network.mesh_encoder import MeshEncoder
from model.module.network.pose_predictor import PosePredictor
from model.module.network.shape_predictor import ShapePredictor



class Encoder(nn.Module):

    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.resnet_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.random_jitter = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        self.backbone = ResNet_Encoder()
        self.featnet = ResNet_Decoder(is_proj=True, out_channel=opts.n_corr_feat, downsample=opts.img_size//opts.corr_h)
        self.featnet_mesh = MeshEncoder(opts.n_corr_feat)

        self.shape_code_predictor = nn.Linear(512, opts.codedim)
        self.shape_predictor = ShapePredictor(opts) 
        self.pose_predictor = PosePredictor(opts, 512)
    
    def encode_img(self, img):
        bsz = img.shape[0]
        img_input = self.resnet_transform(self.random_jitter(img))
        conv2, conv3, conv4, conv5 = self.backbone(img_input)
        img_code = conv5.mean((2, 3))  # 512
        img_feat = self.featnet(conv2, conv3, conv4, conv5)  # b,c,h,w
        img_feat = img_feat.reshape(bsz, self.opts.n_corr_feat, -1)  # b,c,h*w
        img_feat = F.normalize(img_feat, 2, 1)  # b,c,h*w
        return img_code, img_feat
    
    def forward(self, img, mean_v, pp_crop, foc_crop):
        img_code, img_feat = self.encode_img(img)
        shape_code = self.shape_code_predictor(img_code)  # b,c
        pred_v = self.shape_predictor(mean_v, shape_code)

        mesh_feat = self.featnet_mesh(pred_v.detach())
        mesh_feat = F.normalize(mesh_feat, 2, -1)  # b,n,c
        
        rotation, translation, scale = self.pose_predictor(img_code)
        pred_v = pred_v * scale[:, None]
        translation[:, :2] -= (pp_crop / foc_crop) * translation[:, 2:].detach()  # bsz, 3
        rotation = rotation.reshape(-1, 3, 3)  # bsz*hypo, 3, 3
        translation = translation.reshape(-1, 1, 3)  # bsz, 1, 3

        return img_feat, mesh_feat, pred_v, rotation, translation, scale