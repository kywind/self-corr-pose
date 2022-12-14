from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from model.module.network.net_blocks import conv2DBatchNormRelu, residualBlock, pyramidPooling


class PSPNet_Encoder(nn.Module):
    def __init__(self, in_planes=3):
        super(PSPNet_Encoder, self).__init__()
        self.inplanes = 32
        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=in_planes, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
    

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # H, W -> H/2, W/2
        x = x.contiguous()
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        conv2 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        conv3 = self.res_block3(conv2)
        conv4 = self.res_block5(conv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)
        return conv2, conv3, conv4, conv5, conv6  # (bsz,32,64,64) (bsz,64,32,32) (bsz,128,16,16) (bsz,128,8,8) (bsz,128,4,4)

class PSPNet_Decoder(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True, out_channel=64, downsample=4):
        super(PSPNet_Decoder, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj
        self.downsample = downsample

        # Iconvs
        self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv5 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv4 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32, padding=1, stride=1)
        self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64, padding=1, stride=1)

        if self.is_proj:
            self.proj = nn.Conv2d(64, out_channel, 1, padding=0, stride=1)
    
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if hasattr(m.bias,'data'):
        #             m.bias.data.zero_()

    def forward(self, conv2, conv3, conv4, conv5, conv6):
        conv6x = F.interpolate(conv6, (conv5.shape[2],conv5.shape[3]), mode='bilinear', align_corners=False)
        concat5 = torch.cat((conv5, self.upconv6(conv6x)), dim=1)
        conv5 = self.iconv5(concat5) 

        conv5x = F.interpolate(conv5, (conv4.shape[2],conv4.shape[3]), mode='bilinear', align_corners=False)
        concat4 = torch.cat((conv4, self.upconv5(conv5x)), dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.interpolate(conv4, (conv3.shape[2],conv3.shape[3]), mode='bilinear', align_corners=False)
        concat3 = torch.cat((conv3, self.upconv4(conv4x)), dim=1)
        conv3 = self.iconv3(concat3) 
        
        conv3x = F.interpolate(conv3, (conv2.shape[2],conv2.shape[3]), mode='bilinear', align_corners=False)
        concat2 = torch.cat((conv2, self.upconv3(conv3x)), dim=1)
        conv2 = self.iconv2(concat2)

        if self.is_proj:
            if self.downsample == 4:
                return self.proj(conv2)
            else:
                return self.proj(conv3)
        else:
            if self.downsample == 4:
                return conv2
            else:
                return conv3

class ResNet_Encoder(nn.Module):
    def __init__(self):
        super(ResNet_Encoder, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # for name, p in self.resnet.named_parameters():
        #     print(name)
        self.resnet.fc = None
        # self.conv_out = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv1 = self.resnet.conv1(x)
        conv1 = self.resnet.bn1(conv1)
        conv1 = self.resnet.relu(conv1)
        conv1 = self.resnet.maxpool(conv1)

        conv2 = self.resnet.layer1(conv1)
        conv3 = self.resnet.layer2(conv2)
        conv4 = self.resnet.layer3(conv3)
        conv5 = self.resnet.layer4(conv4)
        # conv6 = self.conv_out(conv5)
        return conv2, conv3, conv4, conv5# , conv6  # (bsz,64,64,64) (bsz,128,32,32) (bsz,256,16,16) (bsz,512,8,8) (bsz,128,4,4)

class ResNet_Decoder(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True, out_channel=64, downsample=4):
        super(ResNet_Decoder, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj
        self.downsample = downsample
        # in: (bsz,64,64,64) (bsz,128,32,32) (bsz,256,16,16) (bsz,512,8,8) (bsz,128,4,4)
        # Iconvs
        # self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        # self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv5 = conv2DBatchNormRelu(in_channels=512, k_size=3, n_filters=256, padding=1, stride=1)
        self.iconv4 = conv2DBatchNormRelu(in_channels=512, k_size=3, n_filters=256, padding=1, stride=1)
        self.upconv4 = conv2DBatchNormRelu(in_channels=256, k_size=3, n_filters=128, padding=1, stride=1)
        self.iconv3 = conv2DBatchNormRelu(in_channels=256, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        self.iconv2 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)

        if self.is_proj:
            if self.downsample == 4:
                self.proj = nn.Conv2d(64, out_channel, 1, padding=0, stride=1)
            else:
                self.proj = nn.Conv2d(128, out_channel, 1, padding=0, stride=1)

    def forward(self, conv2, conv3, conv4, conv5):
        # conv6x = F.interpolate(conv6, (conv5.shape[2],conv5.shape[3]), mode='bilinear', align_corners=False)
        # concat5 = torch.cat((conv5, self.upconv6(conv6x)), dim=1)
        # conv5 = self.iconv5(concat5) 

        conv5x = F.interpolate(conv5, (conv4.shape[2],conv4.shape[3]), mode='bilinear', align_corners=False)
        concat4 = torch.cat((conv4, self.upconv5(conv5x)), dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.interpolate(conv4, (conv3.shape[2],conv3.shape[3]), mode='bilinear', align_corners=False)
        concat3 = torch.cat((conv3, self.upconv4(conv4x)), dim=1)
        conv3 = self.iconv3(concat3) 
        
        conv3x = F.interpolate(conv3, (conv2.shape[2],conv2.shape[3]), mode='bilinear', align_corners=False)
        concat2 = torch.cat((conv2, self.upconv3(conv3x)), dim=1)
        conv2 = self.iconv2(concat2)

        if self.is_proj:
            if self.downsample == 4:
                return self.proj(conv2)
            else:
                return self.proj(conv3)
        else:
            if self.downsample == 4:
                return conv2
            else:
                return conv3


