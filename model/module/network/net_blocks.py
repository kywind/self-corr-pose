# MIT License
# 
# Copyright (c) 2018 akanazawa
# Copyright (c) 2021 Google LLC
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
CNN building blocks.
Taken from https://github.com/shubhtuls/factored3d/
'''
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


def fc(batch_norm, nc_inp, nc_out):
    if batch_norm:
        return nn.Sequential(
            nn.Linear(nc_inp, nc_out, bias=True),
            nn.BatchNorm1d(nc_out),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Linear(nc_inp, nc_out),
            nn.LeakyReLU(0.1,inplace=True)
        )

def fc_stack(nc_inp, nc_out, nlayers, use_bn=True):
    modules = []
    for l in range(nlayers):
        modules.append(fc(use_bn, nc_inp, nc_out))
        nc_inp = nc_out
    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder


def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )

def deconv2d(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2,inplace=True)
    )

def upconv2d(in_planes, out_planes, mode='bilinear'):
    if mode == 'nearest':
        print('Using NN upsample!!')
    upconv = nn.Sequential(
        nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(0.2,inplace=True)
    )
    return upconv


def decoder2d(nlayers, nz_shape, nc_input, use_bn=True, nc_final=1, nc_min=8, nc_step=1, init_fc=True, use_deconv=False, upconv_mode='bilinear'):
    ''' Simple 3D encoder with nlayers.
    
    Args:
        nlayers: number of decoder layers
        nz_shape: number of bottleneck
        nc_input: number of channels to start upconvolution from
        use_bn: whether to use batch_norm
        nc_final: number of output channels
        nc_min: number of min channels
        nc_step: double number of channels every nc_step layers
        init_fc: initial features are not spatial, use an fc & unsqueezing to make them 3D
    '''
    modules = []
    if init_fc:
        modules.append(fc(use_bn, nz_shape, nc_input))
        for d in range(3):
            modules.append(Unsqueeze(2))
    nc_output = nc_input
    for nl in range(nlayers):
        if (nl % nc_step==0) and (nc_output//2 >= nc_min):
            nc_output = nc_output//2
        if use_deconv:
            print('Using deconv decoder!')
            modules.append(deconv2d(nc_input, nc_output))
            nc_input = nc_output
            modules.append(conv2d(use_bn, nc_input, nc_output))
        else:
            modules.append(upconv2d(nc_input, nc_output, mode=upconv_mode))
            nc_input = nc_output
            modules.append(conv2d(use_bn, nc_input, nc_output))

    modules.append(nn.Conv2d(nc_output, nc_final, kernel_size=3, stride=1, padding=1, bias=True))
    decoder = nn.Sequential(*modules)
    net_init(decoder)
    return decoder


def conv3d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )

def deconv3d(batch_norm, in_planes, out_planes):
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:        
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )


def encoder3d(nlayers, use_bn=True, nc_input=1, nc_max=128, nc_l1=8, nc_step=1, nz_shape=20):
    ''' Simple 3D encoder with nlayers.
    
    Args:
        nlayers: number of encoder layers
        use_bn: whether to use batch_norm
        nc_input: number of input channels
        nc_max: number of max channels
        nc_l1: number of channels in layer 1
        nc_step: double number of channels every nc_step layers      
        nz_shape: size of bottleneck layer
    '''
    modules = []
    nc_output = nc_l1
    for nl in range(nlayers):
        if (nl>=1) and (nl%nc_step==0) and (nc_output <= nc_max*2):
            nc_output *= 2

        modules.append(conv3d(use_bn, nc_input, nc_output, stride=1))
        nc_input = nc_output
        modules.append(conv3d(use_bn, nc_input, nc_output, stride=1))
        modules.append(torch.nn.MaxPool3d(kernel_size=2, stride=2))

    modules.append(Flatten())
    modules.append(fc_stack(nc_output, nz_shape, 2, use_bn=True))
    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder, nc_output

def decoder3d(nlayers, nz_shape, nc_input, use_bn=True, nc_final=1, nc_min=8, nc_step=1, init_fc=True):
    ''' Simple 3D encoder with nlayers.
    
    Args:
        nlayers: number of decoder layers
        nz_shape: number of bottleneck
        nc_input: number of channels to start upconvolution from
        use_bn: whether to use batch_norm
        nc_final: number of output channels
        nc_min: number of min channels
        nc_step: double number of channels every nc_step layers
        init_fc: initial features are not spatial, use an fc & unsqueezing to make them 3D
    '''
    modules = []
    if init_fc:
        modules.append(fc(use_bn, nz_shape, nc_input))
        for d in range(3):
            modules.append(Unsqueeze(2))
    nc_output = nc_input
    for nl in range(nlayers):
        if (nl%nc_step==0) and (nc_output//2 >= nc_min):
            nc_output = nc_output//2

        modules.append(deconv3d(use_bn, nc_input, nc_output))
        nc_input = nc_output
        modules.append(conv3d(use_bn, nc_input, nc_output))

    modules.append(nn.Conv3d(nc_output, nc_final, kernel_size=3, stride=1, padding=1, bias=True))
    decoder = nn.Sequential(*modules)
    net_init(decoder)
    return decoder


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            #n = m.out_features
            #m.weight.data.normal_(0, 0.02 / n) #this modified initialization seems to work better, but it's very hacky
            #n = m.in_features
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #xavier
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d): #or isinstance(m, nn.ConvTranspose2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #this modified initialization seems to work better, but it's very hacky
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            # Initialize Deconv with bilinear weights.
            base_weights = bilinear_init(m.weight.data.size(-1))
            base_weights = base_weights.unsqueeze(0).unsqueeze(0)
            m.weight.data = base_weights.repeat(m.weight.data.size(0), m.weight.data.size(1), 1, 1)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def bilinear_init(kernel_size=4):
    import numpy as np
    width = kernel_size
    height = kernel_size
    f = int(np.ceil(width / 2.))
    cc = (2 * f - 1 - f % 2) / (2.*f)
    weights = torch.zeros((height, width))
    for y in range(height):
        for x in range(width):
            weights[y, x] = (1 - np.abs(x / f - cc)) * (1 - np.abs(y / f - cc))

    return weights


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1,with_bn=True):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1

        if with_bn:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
        else:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation,with_bn=False)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
        self.downsample = downsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        bias = not with_bn

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=False):
        super(conv2DBatchNormRelu, self).__init__()
        # bias = not with_bn
        bias = True
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True, levels=4):
        super(pyramidPooling, self).__init__()
        self.levels = levels

        self.paths = []
        for i in range(levels):
            self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in np.linspace(1,min(h,w)//2,self.levels,dtype=int):
            k_sizes.append((int(h/pool_size), int(w/pool_size)))
            strides.append((int(h/pool_size), int(w/pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]

        pp_sum = x

        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
            pp_sum = pp_sum + 1./self.levels*out
        pp_sum = self.relu(pp_sum/2.)

        return pp_sum

