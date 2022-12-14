from cv2 import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from zsp.zsp.method import vision_transformer_flexible as vits

class DINO(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch_size = 8
        self.feat_layer = 9
        self.high_res = False
        self.binning = 'None'

        if self.patch_size == 16:
            self.model_name = 'vit_base'
            self.stride = 8
            self.num_patches = 16
            self.padding = 5
            self.pretrain_path = 'pretrain/dino_vitbase16_pretrain.pth'

        elif self.patch_size == 8:
            self.model_name = 'vit_small'
            self.stride = 4
            self.num_patches = 32
            self.padding = 2
            self.pretrain_path = 'pretrain/dino_deitsmall8_pretrain.pth'
        else:
            raise ValueError('ViT models only supported with patch sizes 8 or 16')
        
        if self.high_res: 
            self.num_patches *= 2
        
        self.model = None
        self.load_model()
    
    def load_model(self):
        model = vits.__dict__[self.model_name](patch_size=self.patch_size)
        state_dict = torch.load(self.pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)
        # model.to(device)
        model.eval()

        if self.high_res: 
            model.patch_embed.proj.stride = (self.stride, self.stride)
            model.num_patches = self.num_patches ** 2
            model.patch_embed.patch_size = self.stride
            model.patch_embed.proj.padding = self.padding
        self.model = model

    
    def extract_features_and_attn(self, all_images):
        """
        A definition of relevant dimensions {all_b, nh, t, d}:
            image_size: Side length of input images (assumed square)
            all_b: The first dimension size of the input tensor - not necessarily
                the same as "batch size" in high-level script, as we assume that
                reference and target images are all flattened-then-concatenated
                along the batch dimension. With e.g. a batch size of 2, and 5 target
                images, 1 reference image; all_b = 2 * (5+1) = 12
            h: number of heads in ViT, e.g. 6
            t: number of items in ViT keys/values/tokens, e.g. 785 (= 28*28 + 1)
            d: feature dim in ViT, e.g. 64

        Args:
            all_images (torch.Tensor): shape (all_b, 3, image_size, image_size)
        Returns:
            features (torch.Tensor): shape (all_b, nh, t, d) e.g. (12, 6, 785, 64)
            attn (torch.Tensor): shape (all_b, nh, t, t) e.g. (12, 6, 785, 785)
            output_cls_tokens (torch.Tensor): shape (all_b, nh*d) e.g. (12, 384)
        """
        MAX_BATCH_SIZE = 50
        all_images_batch_size = all_images.size(0)
        c, img_h, img_w = all_images.shape[-3:]
        all_images = all_images.view(-1, c, img_h, img_w)

        with torch.no_grad():
            torch.cuda.empty_cache()

            if all_images_batch_size <= MAX_BATCH_SIZE:
                data = self.model.get_specific_tokens(all_images, layers_to_return=(9, 11))
                features = data[self.feat_layer]['k']
                attn = data[11]['attn']
                output_cls_tokens = data[11]['t'][:, 0, :]

            # Process in chunks to avoid CUDA out-of-memory
            else:
                num_chunks = np.ceil(all_images_batch_size / MAX_BATCH_SIZE).astype('int')
                data_chunks = []
                for i, ims_ in enumerate(all_images.chunk(num_chunks)):
                    data_chunks.append(self.model.get_specific_tokens(ims_, layers_to_return=(9, 11)))

                features = torch.cat([d[self.feat_layer]['k'] for d in data_chunks], dim=0)
                attn = torch.cat([d[11]['attn'] for d in data_chunks], dim=0)
                output_cls_tokens = torch.cat([d[11]['t'][:, 0, :] for d in data_chunks], dim=0)

        return features, attn, output_cls_tokens
    
    def forward(self, img):
        features, attn, output_cls_tokens = self.extract_features_and_attn(img)
        features = features[:, :, 1:, :]
        features = features.permute(0, 1, 3, 2)
        bsz, nh, d, t = features.shape
        hf, wf = int(np.sqrt(t)), int(np.sqrt(t))
        features = features.reshape(bsz, d*nh, hf, wf)  # bsz, d*nh, h, w
        return features