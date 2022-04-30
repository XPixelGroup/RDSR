import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
import functools


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.dropout = nn.Dropout2d(p=0.1)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x

        # out = self.conv1(x)
        # out = self.dropout(out)
        # out = self.conv2(self.relu(out))

        out = self.conv2(self.relu(self.conv1(x)))
        # out = self.dropout(out)
        return identity + out * self.res_scale



# class MSRResNet(nn.Module):
#     ''' modified SRResNet'''
#
#     def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
#         super(MSRResNet, self).__init__()
#         self.upscale = upscale
#
#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         basic_block = functools.partial(ResidualBlock_noBN,nf=nf)
#         self.recon_trunk = make_layer(basic_block, nb)
#
#         # upsampling
#         if self.upscale == 2:
#             self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
#             self.pixel_shuffle = nn.PixelShuffle(2)
#         elif self.upscale == 3:
#             self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
#             self.pixel_shuffle = nn.PixelShuffle(3)
#         elif self.upscale == 4:
#             self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
#             self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
#             self.pixel_shuffle = nn.PixelShuffle(2)
#
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
#
#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#         # initialization
#         initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
#                                      0.1)
#         if self.upscale == 4:
#             initialize_weights(self.upconv2, 0.1)
#
#     def forward(self, x):
#         fea = self.lrelu(self.conv_first(x))
#         out = self.recon_trunk(fea)
#
#         if self.upscale == 4:
#             out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#             out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
#         elif self.upscale == 3 or self.upscale == 2:
#             out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#
#         out = self.conv_last(self.lrelu(self.HRconv(out)))
#         base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
#         out += base
#         return out


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN,nf=nf)
        self.body = make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.body(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class SRResNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(SRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)
        self.LRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.relu = nn.ReLU(inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.conv_first(x)
        out = self.recon_trunk(fea)
        out = self.LRconv(out)

        if self.upscale == 4:
            out = self.relu(self.pixel_shuffle(self.upconv1(out+fea)))
            out = self.relu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.relu(self.pixel_shuffle(self.upconv1(out+fea)))

        out = self.conv_last(self.relu(self.HRconv(out)))

        return out


class MSRResNet_wGR_details(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_wGR_details, self).__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        first = feat

        feat = self.body1(feat)
        b1  = feat
        feat = self.body2(feat)
        b2 = feat
        feat = self.body3(feat)
        b3 = feat
        feat = self.body4(feat)
        b4 = feat
        feat = self.body5(feat)
        b5 = feat
        feat = self.body6(feat)
        b6 = feat
        feat = self.body7(feat)
        b7 = feat
        feat = self.body8(feat)
        b8 = feat
        feat = self.body9(feat)
        b9 = feat
        feat = self.body10(feat)
        b10 = feat
        feat = self.body11(feat)
        b11 = feat
        feat = self.body12(feat)
        b12 = feat
        feat = self.body13(feat)
        b13 = feat
        feat = self.body14(feat)
        b14 = feat
        feat = self.body15(feat)
        b15 = feat
        out = self.body16(feat)
        b16 = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        middle = out

        out = self.lrelu(self.conv_hr(out))

        last = out

        out = self.conv_last(out)

        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return first, b1, b2, b3, b4, b5, b6, b7, b8, \
               b9, b10, b11, b12, b13, b14, b15, b16, \
               middle, last, out


class MSRResNet_dropoutquater_channel05_inres(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_dropoutquater_channel05_inres, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body1 = make_layer(ResidualBlockNoBN, int(3*num_block/4), num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN_dropchannel05_inres, int(num_block/4), num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights([self.upconv2,self.upconv1], 0.1)

    def forward(self, x):

        fea = self.lrelu(self.conv_first(x))

        fea = self.body1(fea)
        out = self.body2(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
    

class MSRResNet_details_lastob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_lastob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self,weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight)*10

        #return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        size_out = out.shape[1]

        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.conv_last.weight[0])
        normed_1 = self.normal_in_outchannel(self.conv_last.weight[1])
        normed_2 = self.normal_in_outchannel(self.conv_last.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out[0]).sum() - abs(temp).sum()) / abs(out[0]).sum())

            out_copy = self.conv_last(out_copy)

            out_copy += base

            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_hrob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_hrob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self,weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight)*10

        #return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        size_out = out.shape[1]
        out_out = out
        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.conv_hr.weight[0])
        normed_1 = self.normal_in_outchannel(self.conv_hr.weight[1])
        normed_2 = self.normal_in_outchannel(self.conv_hr.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b16ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b16ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self,weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight)*10

        #return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        size_out = out.shape[1]
        out_out = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.upconv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.upconv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.upconv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base

            # out_copy = sr - out_copy

            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2



class MSRResNet_details_b12ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b12ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body13[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body13[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body13[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b8ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b8ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body9[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body9[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body9[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2



class MSRResNet_details_b4ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b4ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body5[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body5[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body5[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body5(out_copy)
            out_copy = self.body6(out_copy)
            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2



class MSRResNet_details_firstob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_firstob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body1[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body1[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body1[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body1(out_copy)
            out_copy = self.body2(out_copy)
            out_copy = self.body3(out_copy)
            out_copy = self.body4(out_copy)
            out_copy = self.body5(out_copy)
            out_copy = self.body6(out_copy)
            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b15ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b15ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)

        size_out = feat.shape[1]
        out_out = feat

        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body16[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body16[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body16[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            # out_copy = self.body1(out_copy)
            # out_copy = self.body2(out_copy)
            # out_copy = self.body3(out_copy)
            # out_copy = self.body4(out_copy)
            # out_copy = self.body5(out_copy)
            # out_copy = self.body6(out_copy)
            # out_copy = self.body7(out_copy)
            # out_copy = self.body8(out_copy)
            # out_copy = self.body9(out_copy)
            # out_copy = self.body10(out_copy)
            # out_copy = self.body11(out_copy)
            # out_copy = self.body12(out_copy)
            # out_copy = self.body13(out_copy)
            # out_copy = self.body14(out_copy)
            # out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b14ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b14ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body15[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body15[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body15[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            # out_copy = self.body1(out_copy)
            # out_copy = self.body2(out_copy)
            # out_copy = self.body3(out_copy)
            # out_copy = self.body4(out_copy)
            # out_copy = self.body5(out_copy)
            # out_copy = self.body6(out_copy)
            # out_copy = self.body7(out_copy)
            # out_copy = self.body8(out_copy)
            # out_copy = self.body9(out_copy)
            # out_copy = self.body10(out_copy)
            # out_copy = self.body11(out_copy)
            # out_copy = self.body12(out_copy)
            # out_copy = self.body13(out_copy)
            # out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b13ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b13ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body14[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body14[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body14[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            # out_copy = self.body1(out_copy)
            # out_copy = self.body2(out_copy)
            # out_copy = self.body3(out_copy)
            # out_copy = self.body4(out_copy)
            # out_copy = self.body5(out_copy)
            # out_copy = self.body6(out_copy)
            # out_copy = self.body7(out_copy)
            # out_copy = self.body8(out_copy)
            # out_copy = self.body9(out_copy)
            # out_copy = self.body10(out_copy)
            # out_copy = self.body11(out_copy)
            # out_copy = self.body12(out_copy)
            # out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b11ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b11ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body12[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body12[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body12[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            # out_copy = self.body1(out_copy)
            # out_copy = self.body2(out_copy)
            # out_copy = self.body3(out_copy)
            # out_copy = self.body4(out_copy)
            # out_copy = self.body5(out_copy)
            # out_copy = self.body6(out_copy)
            # out_copy = self.body7(out_copy)
            # out_copy = self.body8(out_copy)
            # out_copy = self.body9(out_copy)
            # out_copy = self.body10(out_copy)
            # out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b10ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b10ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body11[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body11[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body11[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            # out_copy = self.body1(out_copy)
            # out_copy = self.body2(out_copy)
            # out_copy = self.body3(out_copy)
            # out_copy = self.body4(out_copy)
            # out_copy = self.body5(out_copy)
            # out_copy = self.body6(out_copy)
            # out_copy = self.body7(out_copy)
            # out_copy = self.body8(out_copy)
            # out_copy = self.body9(out_copy)
            # out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b9ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b9ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body10[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body10[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body10[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            # out_copy = self.body1(out_copy)
            # out_copy = self.body2(out_copy)
            # out_copy = self.body3(out_copy)
            # out_copy = self.body4(out_copy)
            # out_copy = self.body5(out_copy)
            # out_copy = self.body6(out_copy)
            # out_copy = self.body7(out_copy)
            # out_copy = self.body8(out_copy)
            # out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b7ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b7ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body8[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body8[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body8[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b6ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b6ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body7[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body7[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body7[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b5ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b5ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body6[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body6[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body6[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body6(out_copy)
            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b3ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b3ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body4[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body4[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body4[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body4(out_copy)
            out_copy = self.body5(out_copy)
            out_copy = self.body6(out_copy)
            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b2ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b2ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body3[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body3[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body3[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body3(out_copy)
            out_copy = self.body4(out_copy)
            out_copy = self.body5(out_copy)
            out_copy = self.body6(out_copy)
            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2


class MSRResNet_details_b1ob(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_b1ob, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        return (weight - min_weight) * 10

        # return (weight - min_weight) / denominator

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)

        size_out = feat.shape[1]
        out_out = feat

        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        sr = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr += base

        normed_0 = self.normal_in_outchannel(self.body2[0].conv1.weight[0])
        normed_1 = self.normal_in_outchannel(self.body2[0].conv1.weight[1])
        normed_2 = self.normal_in_outchannel(self.body2[0].conv1.weight[2])

        pic_list = []
        feature_list = []
        weight_list_0 = []
        weight_list_1 = []
        weight_list_2 = []

        for index in range(size_out):
            temp = out_out[0][index][:][:]
            feature_list.append(temp)
            lack_num = 1
            out_copy = out_out.clone()
            for lack in range(lack_num):
                if index + lack > (size_out - 1):
                    lack_ = lack - size_out
                else:
                    lack_ = lack
                out_copy[0][index + lack_] = 0

            out_copy = out_copy / ((abs(out_out[0]).sum() - abs(temp).sum()) / abs(out_out[0]).sum())

            out_copy = self.body2(out_copy)
            out_copy = self.body3(out_copy)
            out_copy = self.body4(out_copy)
            out_copy = self.body5(out_copy)
            out_copy = self.body6(out_copy)
            out_copy = self.body7(out_copy)
            out_copy = self.body8(out_copy)
            out_copy = self.body9(out_copy)
            out_copy = self.body10(out_copy)
            out_copy = self.body11(out_copy)
            out_copy = self.body12(out_copy)
            out_copy = self.body13(out_copy)
            out_copy = self.body14(out_copy)
            out_copy = self.body15(out_copy)
            out_copy = self.body16(out_copy)

            if self.upscale == 4:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv2(out_copy)))
            elif self.upscale in [2, 3]:
                out_copy = self.lrelu(self.pixel_shuffle(self.upconv1(out_copy)))

            out_copy = self.lrelu(self.conv_hr(out_copy))
            out_copy = self.conv_last(out_copy)

            out_copy += base
            pic_list.append(out_copy)
            weight_list_0.append(normed_0[index])
            weight_list_1.append(normed_1[index])
            weight_list_2.append(normed_2[index])

        return sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2

# class MSRResNet_details_lastob_onetoall(nn.Module):
#     def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
#         super(MSRResNet_details_lastob_onetoall, self).__init__()
#         self.upscale = upscale
#
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
#         # upsampling
#         if self.upscale in [2, 3]:
#             self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
#             self.pixel_shuffle = nn.PixelShuffle(self.upscale)
#         elif self.upscale == 4:
#             self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
#             self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
#             self.pixel_shuffle = nn.PixelShuffle(2)
#
#         self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#
#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#         # initialization
#         default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
#         if self.upscale == 4:
#             default_init_weights(self.upconv2, 0.1)
#
#     def forward(self, x):
#         feat = self.lrelu(self.conv_first(x))
#
#         feat = self.body1(feat)
#         feat = self.body2(feat)
#         feat = self.body3(feat)
#         feat = self.body4(feat)
#         feat = self.body5(feat)
#         feat = self.body6(feat)
#         feat = self.body7(feat)
#         feat = self.body8(feat)
#         feat = self.body9(feat)
#         feat = self.body10(feat)
#         feat = self.body11(feat)
#         feat = self.body12(feat)
#         feat = self.body13(feat)
#         feat = self.body14(feat)
#         feat = self.body15(feat)
#         out = self.body16(feat)
#
#         if self.upscale == 4:
#             out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#             out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
#         elif self.upscale in [2, 3]:
#             out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#
#         out = self.lrelu(self.conv_hr(out))
#         size_out = out.shape[1]
#
#         sr = self.conv_last(out)
#         base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
#         sr += base
#
#         pic_list = []
#
#         out_copy = out.clone()
#         for index in range(size_out-1):
#             abs_middle = out_copy[0][index]
#             out_copy[0][index] = 0
#
#             out_copy = out_copy / ((abs(out[0]).sum() - abs(abs_middle).sum()) / abs(out[0]).sum())
#
#             out_copy_conv = self.conv_last(out_copy)
#             out_copy_conv += base
#
#             pic_list.append(out_copy_conv)
#
#         return sr, pic_list

class MSRResNet_details_lastob_onetoall(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_lastob_onetoall, self).__init__()
        self.upscale = upscale
        self.num_feat=num_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        # return (weight - min_weight) * 10
        return (weight - min_weight) / denominator

    def forward(self, x):

        feat_out_convfirst = self.lrelu(self.conv_first(x))
        feat_out_body1 = self.body1(feat_out_convfirst)
        feat_out_body2 = self.body2(feat_out_body1)
        feat_out_body3 = self.body3(feat_out_body2)
        feat_out_body4 = self.body4(feat_out_body3)
        feat_out_body5 = self.body5(feat_out_body4)
        feat_out_body6 = self.body6(feat_out_body5)
        feat_out_body7 = self.body7(feat_out_body6)
        feat_out_body8 = self.body8(feat_out_body7)
        feat_out_body9 = self.body9(feat_out_body8)
        feat_out_body10 = self.body10(feat_out_body9)
        feat_out_body11 = self.body11(feat_out_body10)
        feat_out_body12 = self.body12(feat_out_body11)
        feat_out_body13 = self.body13(feat_out_body12)
        feat_out_body14 = self.body14(feat_out_body13)
        feat_out_body15 = self.body15(feat_out_body14)
        feat_out_body16 = self.body16(feat_out_body15)

        feat_out_up1 = self.lrelu(self.pixel_shuffle(self.upconv1(feat_out_body16)))
        feat_out_up2 = self.lrelu(self.pixel_shuffle(self.upconv2(feat_out_up1)))

        feat_out_hrconv = self.lrelu(self.conv_hr(feat_out_up2))

        feat_out_lastconv = self.conv_last(feat_out_hrconv)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr=feat_out_lastconv+base

        pic_list=[]
        out_copy = feat_out_hrconv.clone()
        list_num=list(range(self.num_feat))[::-1]
        list_num = list(range(self.num_feat))
        # print(list_num)

        # num=0
        # for index in list_num:
        #
        #     out_copy[0][index]=0
        #
        #     #sr_copy=out_copy/((self.num_feat-num)/self.num_feat)
        #
        #     sr_copy = self.conv_last(out_copy)
        #
        #     sr_copy += base
        #     pic_list.append(sr_copy)
        #     num+=1

        layer_sum=torch.sum(abs(feat_out_hrconv))

        num = 0
        channel_sum=0
        for index in list_num:
            channel_sum+=torch.sum(abs(out_copy[0][index]))
            #print(layer_sum)
            #print(channel_sum)
            #print((layer_sum-channel_sum)/layer_sum)
            out_copy[0][index] = 0

            sr_copy=out_copy/((layer_sum-channel_sum)/layer_sum)

            sr_copy = self.conv_last(sr_copy)

            sr_copy += base
            pic_list.append(sr_copy)
            num += 1

        return sr, pic_list

class MSRResNet_details_lastob_range_onetoall(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_lastob_range_onetoall, self).__init__()
        self.upscale = upscale
        self.num_feat = num_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def normal_in_outchannel(self, weight):
        min_weight = weight.min()
        max_weight = weight.max()

        denominator = max_weight - min_weight
        # return (weight - min_weight) * 10
        return (weight - min_weight) / denominator

    def forward(self, x, att_range_final):

        feat_out_convfirst = self.lrelu(self.conv_first(x))
        feat_out_body1 = self.body1(feat_out_convfirst)
        feat_out_body2 = self.body2(feat_out_body1)
        feat_out_body3 = self.body3(feat_out_body2)
        feat_out_body4 = self.body4(feat_out_body3)
        feat_out_body5 = self.body5(feat_out_body4)
        feat_out_body6 = self.body6(feat_out_body5)
        feat_out_body7 = self.body7(feat_out_body6)
        feat_out_body8 = self.body8(feat_out_body7)
        feat_out_body9 = self.body9(feat_out_body8)
        feat_out_body10 = self.body10(feat_out_body9)
        feat_out_body11 = self.body11(feat_out_body10)
        feat_out_body12 = self.body12(feat_out_body11)
        feat_out_body13 = self.body13(feat_out_body12)
        feat_out_body14 = self.body14(feat_out_body13)
        feat_out_body15 = self.body15(feat_out_body14)
        feat_out_body16 = self.body16(feat_out_body15)

        feat_out_up1 = self.lrelu(self.pixel_shuffle(self.upconv1(feat_out_body16)))
        feat_out_up2 = self.lrelu(self.pixel_shuffle(self.upconv2(feat_out_up1)))

        feat_out_hrconv = self.lrelu(self.conv_hr(feat_out_up2))

        feat_out_lastconv = self.conv_last(feat_out_hrconv)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        sr=feat_out_lastconv+base

        pic_list=[]
        out_copy = feat_out_hrconv.clone()
        list_num=list(range(self.num_feat))[::-1]
        list_num = list(range(self.num_feat))
        # print(list_num)

        # num=0
        # for index in list_num:
        #
        #     out_copy[0][index]=0
        #
        #     #sr_copy=out_copy/((self.num_feat-num)/self.num_feat)
        #
        #     sr_copy = self.conv_last(out_copy)
        #
        #     sr_copy += base
        #     pic_list.append(sr_copy)
        #     num+=1

        layer_sum=torch.sum(abs(feat_out_hrconv))

        num = 0
        channel_sum=0
        for index in att_range_final:
            index = int(index)
            channel_sum+=torch.sum(abs(out_copy[0][index]))
            #print(layer_sum)
            #print(channel_sum)
            #print((layer_sum-channel_sum)/layer_sum)
            out_copy[0][index] = 0

            sr_copy=out_copy/((layer_sum-channel_sum)/layer_sum)

            sr_copy = self.conv_last(sr_copy)

            sr_copy += base
            pic_list.append(sr_copy)
            num += 1

        return sr, pic_list