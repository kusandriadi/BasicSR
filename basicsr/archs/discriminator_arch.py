import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


@ARCH_REGISTRY.register()
class PatchGANDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(PatchGANDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(num_in_ch, num_feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf = num_feat
        for i in range(1, 3):  # Reduce the depth for a lightweight version
            layers += [
                nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            nf *= 2
        layers += [nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, ratio=2):
        super(GhostConv, self).__init__()
        init_ch = out_ch // ratio
        new_ch = init_ch * (ratio - 1)
        
        self.primary_conv = nn.Conv2d(in_ch, init_ch, kernel_size, stride, padding, bias=False)
        self.cheap_operation = nn.Conv2d(init_ch, new_ch, 3, 1, 1, groups=init_ch, bias=False)
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


@ARCH_REGISTRY.register()
class UltraLightDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=32):
        super(UltraLightDiscriminator, self).__init__()
        
        # Multi-scale input branches for better PI metrics
        self.branch1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(num_in_ch, num_feat//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Ghost convolutions (50% parameter reduction)
        self.ghost1 = GhostConv(num_feat, num_feat * 2, 4, 2, 1)
        self.eca1 = EfficientChannelAttention(num_feat * 2)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.ghost2 = GhostConv(num_feat * 2, num_feat * 4, 4, 2, 1)
        self.eca2 = EfficientChannelAttention(num_feat * 4) 
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Lightweight head with global features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = spectral_norm(nn.Conv2d(num_feat * 4, 1, 4, 1, 1))
        self.global_fc = spectral_norm(nn.Linear(num_feat * 4, 1))
        
    def forward(self, x):
        # Multi-scale input processing
        b1 = self.branch1(x)
        b2 = F.interpolate(self.branch2(x), size=b1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([b1, b2], dim=1)
        
        # Ghost convolutions with efficient attention
        x = self.ghost1(x)
        x = self.eca1(x) 
        x = self.act1(x)
        
        x = self.ghost2(x)
        x = self.eca2(x)
        x = self.act2(x)
        
        # Dual output: local patches + global score
        local_out = self.local_conv(x)
        global_feat = self.global_pool(x).flatten(1)
        global_out = self.global_fc(global_feat)
        
        # Combine for better discrimination
        return local_out + global_out.unsqueeze(-1).unsqueeze(-1)


@ARCH_REGISTRY.register()
class LightweightDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(LightweightDiscriminator, self).__init__()
        
        # Initial standard conv
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Depthwise separable convolutions with channel attention
        self.dsconv1 = DepthwiseSeparableConv(num_feat, num_feat * 2, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(num_feat * 2)
        self.ca1 = ChannelAttention(num_feat * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.dsconv2 = DepthwiseSeparableConv(num_feat * 2, num_feat * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_feat * 4)
        self.ca2 = ChannelAttention(num_feat * 4)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        # Final conv with spectral normalization
        self.final_conv = spectral_norm(nn.Conv2d(num_feat * 4, 1, kernel_size=4, stride=1, padding=1))
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        
        x = self.dsconv1(x)
        x = self.bn1(x)
        x = self.ca1(x)
        x = self.act2(x)
        
        x = self.dsconv2(x)
        x = self.bn2(x)
        x = self.ca2(x)
        x = self.act3(x)
        
        x = self.final_conv(x)
        return x


# ---- helper blocks ------------------------------------------------------


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm that works on NCHW tensors (follows ConvNeXt)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__(channels, eps=eps)

    def forward(self, x):
        # (N,C,H,W) -> (N,H,W,C) -> LN -> back
        return (
            super()
            .forward(x.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
        )


class ConvNeXtBlock(nn.Module):
    """Depthwise Conv + LayerNorm + PW-MLP block (kernel size 7)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


def make_stage(dim, depth):
    """Stack `depth` ConvNeXt blocks."""
    return nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])


# ---- discriminator ------------------------------------------------------


@ARCH_REGISTRY.register()
class ConvNeXtDiscriminator(nn.Module):
    """ConvNeXt-style discriminator (lightweight).

    Args:
        num_in_ch  (int): input channels, e.g. 3.
        num_feat   (int): base feature width. Standard ConvNeXt-tiny starts
                          at 96, but we scale it from `num_feat`.
        stage_depth (list[int]): number of blocks in the 4 stages.
        spectral_norm (bool): apply SN on the 1×1 head conv.
    """

    def __init__(
            self,
            num_in_ch: int = 3,
            num_feat: int = 64,
            stage_depth: list = (1, 1, 2, 1),
            apply_sn: bool = True):

        super().__init__()
        dims = [num_feat, num_feat * 2, num_feat * 4, num_feat * 8]

        # patch-embedding stem (4×4 /4)
        self.stem = nn.Sequential(
            nn.Conv2d(num_in_ch, dims[0], 4, 4, 0),
            LayerNorm2d(dims[0]),
        )

        # hierarchical stages with downsampling between them
        stages, downs = [], []
        for i in range(4):
            stages.append(make_stage(dims[i], stage_depth[i]))
            if i < 3:  # no downsample after last stage
                downs.append(
                    nn.Sequential(
                        LayerNorm2d(dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], 2, 2),
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(downs)

        # classification head – 1×1 conv → global mean → linear
        head_channels = dims[-1]
        head_conv = nn.Conv2d(head_channels, 1, 1)
        self.head = spectral_norm(head_conv) if apply_sn else head_conv

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return x

    def forward(self, x):
        """
        Input:  H×W image (e.g. 128×128/256×256)
        Output: PatchGAN-style score map (N × 1 × h' × w')
        """
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

@ARCH_REGISTRY.register()
class NextSRGANDiscriminator(nn.Module):
    """
    ConvNeXt-inspired discriminator from NextSRGAN (Fig.6 of the paper).

    Architecture:
      - conv3x3 stride1 + conv4x4 stride2 + BN
      - repeat stages doubling channels: [1,2,4,8]
      - global average pooling + linear output
    """
    def __init__(self, num_in_ch=3, num_feat=64, wd=0.):
        super().__init__()
        # helper conv layers
        def conv_k3s1(in_ch, out_ch, bias=True):
            layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            layer.weight.data.mul_(0.8)  # approximate GELU gain
            if bias:
                nn.init.zeros_(layer.bias)
            return layer

        def conv_k4s2(in_ch, out_ch, bias=False):
            layer = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            layer.weight.data.mul_(0.8)  # approximate GELU gain
            return layer


        self.gelu = nn.GELU()
        # stage 0
        self.conv0_0 = conv_k3s1(num_in_ch, num_feat)
        self.conv0_1 = conv_k4s2(num_feat, num_feat, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, eps=1e-5, momentum=0.9)
        # stage 1
        self.conv1_0 = conv_k3s1(num_feat, num_feat * 2, bias=False)
        self.conv1_1 = conv_k4s2(num_feat * 2, num_feat * 2, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, eps=1e-5, momentum=0.9)
        # stage 2
        self.conv2_0 = conv_k3s1(num_feat * 2, num_feat * 4, bias=False)
        self.conv2_1 = conv_k4s2(num_feat * 4, num_feat * 4, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, eps=1e-5, momentum=0.9)
        # stage 3
        self.conv3_0 = conv_k3s1(num_feat * 4, num_feat * 8, bias=False)
        self.conv3_1 = conv_k4s2(num_feat * 8, num_feat * 8, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, eps=1e-5, momentum=0.9)
        # stage 4
        self.conv4_0 = conv_k3s1(num_feat * 8, num_feat * 8, bias=False)
        self.conv4_1 = conv_k4s2(num_feat * 8, num_feat * 8, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, eps=1e-5, momentum=0.9)
        # final pooling and linear
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = spectral_norm(nn.Linear(num_feat * 8, 1))
        nn.init.zeros_(self.linear.bias)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='linear')

    def forward(self, x):
        # stage 0
        x = self.conv0_0(x)
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        # stage 1
        x = self.conv1_0(x)
        x = self.gelu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        # stage 2
        x = self.conv2_0(x)
        x = self.gelu(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        # stage 3
        x = self.conv3_0(x)
        x = self.gelu(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        # stage 4
        x = self.conv4_0(x)
        x = self.gelu(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        # head
        b, c, h, w = x.size()
        x = self.gap(x).view(b, c)
        x = self.linear(x)
        return x
