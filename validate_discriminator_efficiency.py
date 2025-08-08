#!/usr/bin/env python3
"""
Script untuk memvalidasi efisiensi UltraLightDiscriminator_v2
Membandingkan parameter count, memory usage, dan inference time vs discriminator lainnya
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import time
import math
from typing import Dict, List, Tuple

# Import discriminator classes (copy from discriminator_arch.py untuk avoid dependency)
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

class DetailEnhancedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_spectral_norm=False):
        super(DetailEnhancedConv, self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9)
        self.gelu = nn.GELU()
        
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        self.conv.weight.data.mul_(0.8)
        
    def forward(self, x):
        return self.gelu(self.bn(self.conv(x)))

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.structure_branch = nn.Sequential(
            GhostConv(in_ch, out_ch//2, 4, 2, 1),
            EfficientChannelAttention(out_ch//2)
        )
        
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_ch//4, out_ch//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch//2, eps=1e-5, momentum=0.9),
            nn.GELU()
        )
        
        self.fusion = nn.Conv2d(out_ch, out_ch, 1, 1, 0)
        
    def forward(self, x):
        structure_feat = self.structure_branch(x)
        detail_feat = self.detail_branch(x)
        combined = torch.cat([structure_feat, detail_feat], dim=1)
        return self.fusion(combined)

# Original UltraLight discriminator
class UltraLightDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=32):
        super(UltraLightDiscriminator, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(num_in_ch, num_feat//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.ghost1 = GhostConv(num_feat, num_feat * 2, 4, 2, 1)
        self.eca1 = EfficientChannelAttention(num_feat * 2)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.ghost2 = GhostConv(num_feat * 2, num_feat * 4, 4, 2, 1)
        self.eca2 = EfficientChannelAttention(num_feat * 4) 
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = spectral_norm(nn.Conv2d(num_feat * 4, 1, 4, 1, 1))
        self.global_fc = spectral_norm(nn.Linear(num_feat * 4, 1))
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = torch.nn.functional.interpolate(self.branch2(x), size=b1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([b1, b2], dim=1)
        
        x = self.ghost1(x)
        x = self.eca1(x) 
        x = self.act1(x)
        
        x = self.ghost2(x)
        x = self.eca2(x)
        x = self.act2(x)
        
        local_out = self.local_conv(x)
        global_feat = self.global_pool(x).flatten(1)
        global_out = self.global_fc(global_feat)
        
        return local_out + global_out.unsqueeze(-1).unsqueeze(-1)

# Enhanced v2 discriminator
class UltraLightDiscriminator_v2(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=32):
        super(UltraLightDiscriminator_v2, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(num_in_ch, num_feat//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.msfe1 = MultiScaleFeatureExtractor(num_feat, num_feat * 2)
        self.msfe2 = MultiScaleFeatureExtractor(num_feat * 2, num_feat * 4)
        
        self.detail_enhancer = DetailEnhancedConv(num_feat * 4, num_feat * 4, 3, 1, 1, use_spectral_norm=True)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = spectral_norm(nn.Conv2d(num_feat * 4, 1, 4, 1, 1))
        self.global_fc = spectral_norm(nn.Linear(num_feat * 4, 1))
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not hasattr(m, 'weight_initialized'):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.2)
                    m.weight.data.mul_(0.8)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    m.weight_initialized = True
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = torch.nn.functional.interpolate(self.branch2(x), size=b1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([b1, b2], dim=1)
        
        x = self.msfe1(x)
        x = self.msfe2(x)
        
        x = self.detail_enhancer(x)
        
        local_out = self.local_conv(x)
        global_feat = self.global_pool(x).flatten(1)
        global_out = self.global_fc(global_feat)
        
        alpha = 0.7
        beta = 0.3
        return alpha * local_out + beta * global_out.unsqueeze(-1).unsqueeze(-1)

# UNet Discriminator dengan Spectral Normalization
class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = torch.nn.functional.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = torch.nn.functional.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = torch.nn.functional.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = torch.nn.functional.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = torch.nn.functional.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = torch.nn.functional.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = torch.nn.functional.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = torch.nn.functional.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        out = torch.nn.functional.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = torch.nn.functional.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out

# PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(PatchGANDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(num_in_ch, num_feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf = num_feat
        for i in range(1, 3):
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

# Lightweight Discriminator with Depthwise Separable Conv
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

class LightweightDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(LightweightDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.dsconv1 = DepthwiseSeparableConv(num_feat, num_feat * 2, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(num_feat * 2)
        self.ca1 = ChannelAttention(num_feat * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.dsconv2 = DepthwiseSeparableConv(num_feat * 2, num_feat * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_feat * 4)
        self.ca2 = ChannelAttention(num_feat * 4)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
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

# NextSRGAN Discriminator
class NextSRGANDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, wd=0.):
        super().__init__()
        def conv_k3s1(in_ch, out_ch, bias=True):
            layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            layer.weight.data.mul_(0.8)
            if bias:
                nn.init.zeros_(layer.bias)
            return layer

        def conv_k4s2(in_ch, out_ch, bias=False):
            layer = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            layer.weight.data.mul_(0.8)
            return layer

        self.gelu = nn.GELU()
        self.conv0_0 = conv_k3s1(num_in_ch, num_feat)
        self.conv0_1 = conv_k4s2(num_feat, num_feat, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, eps=1e-5, momentum=0.9)
        self.conv1_0 = conv_k3s1(num_feat, num_feat * 2, bias=False)
        self.conv1_1 = conv_k4s2(num_feat * 2, num_feat * 2, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, eps=1e-5, momentum=0.9)
        self.conv2_0 = conv_k3s1(num_feat * 2, num_feat * 4, bias=False)
        self.conv2_1 = conv_k4s2(num_feat * 4, num_feat * 4, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, eps=1e-5, momentum=0.9)
        self.conv3_0 = conv_k3s1(num_feat * 4, num_feat * 8, bias=False)
        self.conv3_1 = conv_k4s2(num_feat * 8, num_feat * 8, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, eps=1e-5, momentum=0.9)
        self.conv4_0 = conv_k3s1(num_feat * 8, num_feat * 8, bias=False)
        self.conv4_1 = conv_k4s2(num_feat * 8, num_feat * 8, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, eps=1e-5, momentum=0.9)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = spectral_norm(nn.Linear(num_feat * 8, 1))
        nn.init.zeros_(self.linear.bias)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='linear')

    def forward(self, x):
        x = self.conv0_0(x)
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.conv1_0(x)
        x = self.gelu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.conv2_0(x)
        x = self.gelu(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.conv3_0(x)
        x = self.gelu(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.conv4_0(x)
        x = self.gelu(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        b, c, h, w = x.size()
        x = self.gap(x).view(b, c)
        x = self.linear(x)
        return x

# Ultra-Fast Quality Discriminator
class UltraFastQualityDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=48):
        super(UltraFastQualityDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 4, 2, 1)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_feat * 2, eps=1e-5, momentum=0.9)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_feat * 4, eps=1e-5, momentum=0.9)
        self.act3 = nn.GELU()
        
        k_size = 3
        self.eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.local_head = nn.Conv2d(num_feat * 4, 1, 4, 1, 1)
        self.global_conv = nn.Conv2d(num_feat * 4, 1, 1, 1, 0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                m.weight.data.mul_(0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.eca(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        x_enhanced = x * y.expand_as(x)
        
        local_out = self.local_head(x_enhanced)
        global_out = self.gap(self.global_conv(x_enhanced))
        
        return local_out + global_out

# Default VGG-style discriminator untuk comparison
class VGGStyleDiscriminator(nn.Module):
    """Default discriminator dari BasicSR"""
    def __init__(self, num_in_ch=3, num_feat=64, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size

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

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters by layer type"""
    param_counts = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if num_params > 0:
                param_counts[f"{name} ({type(module).__name__})"] = num_params
    return param_counts

def measure_memory_usage(model: nn.Module, input_size: Tuple[int, int, int, int], device: str = 'cuda') -> Dict[str, float]:
    """Measure GPU memory usage during forward pass"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    model = model.to(device)
    input_tensor = torch.randn(input_size, device=device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory before forward
    mem_before = torch.cuda.memory_allocated(device) / 1024**2  # MB
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Measure memory after forward
    mem_after = torch.cuda.memory_allocated(device) / 1024**2  # MB
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    
    return {
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_used_mb": mem_after - mem_before,
        "peak_memory_mb": mem_peak
    }

def measure_inference_time(model: nn.Module, input_size: Tuple[int, int, int, int], 
                          device: str = 'cuda', num_runs: int = 100) -> Dict[str, float]:
    """Measure inference time"""
    model = model.to(device)
    model.eval()
    input_tensor = torch.randn(input_size, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    return {
        "avg_inference_time_ms": avg_time_ms,
        "fps": 1000 / avg_time_ms
    }

def compare_discriminators():
    """Main comparison function"""
    print("=" * 80)
    print("DISCRIMINATOR EFFICIENCY COMPARISON")
    print("=" * 80)
    
    # Create models
    models = {
        "VGGStyle (Default)": VGGStyleDiscriminator(num_feat=64),
        "UNetDiscriminatorSN": UNetDiscriminatorSN(num_feat=64),
        "PatchGAN": PatchGANDiscriminator(num_feat=64),
        "LightweightDiscriminator": LightweightDiscriminator(num_feat=64),
        "NextSRGAN": NextSRGANDiscriminator(num_feat=64),
        "UltraLight (Original)": UltraLightDiscriminator(num_feat=32),
        "UltraLight_v2 (Enhanced)": UltraLightDiscriminator_v2(num_feat=32),
        "UltraFastQuality": UltraFastQualityDiscriminator(num_feat=48)
    }
    
    input_size = (1, 3, 128, 128)  # Batch=1, RGB, 128x128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing on device: {device}")
    print(f"Input size: {input_size}")
    print()
    
    results = {}
    
    for name, model in models.items():
        print(f"Analyzing {name}...")
        print("-" * 50)
        
        # Parameter count
        total_params = count_parameters(model)
        param_breakdown = count_parameters_by_layer(model)
        
        # Memory usage
        if device == 'cuda':
            memory_stats = measure_memory_usage(model, input_size, device)
            time_stats = measure_inference_time(model, input_size, device)
        else:
            memory_stats = {"error": "CPU mode - memory stats not available"}
            time_stats = {"error": "CPU mode - timing not accurate"}
        
        results[name] = {
            "parameters": total_params,
            "memory": memory_stats,
            "timing": time_stats
        }
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")  # Assume float32
        
        if device == 'cuda' and 'error' not in memory_stats:
            print(f"GPU Memory Used: {memory_stats['memory_used_mb']:.2f} MB")
            print(f"Peak GPU Memory: {memory_stats['peak_memory_mb']:.2f} MB")
            print(f"Inference Time: {time_stats['avg_inference_time_ms']:.2f} ms")
            print(f"FPS: {time_stats['fps']:.1f}")
        
        print()
    
    # Summary comparison
    print("=" * 80)
    print("EFFICIENCY SUMMARY & RANKING")
    print("=" * 80)
    
    baseline_params = results["VGGStyle (Default)"]["parameters"]
    
    # Sort by parameters (ascending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["parameters"])
    
    print("RANKING BY PARAMETERS (Lightest to Heaviest):")
    print("-" * 60)
    for i, (name, stats) in enumerate(sorted_results, 1):
        params = stats["parameters"]
        reduction = (1 - params / baseline_params) * 100
        
        print(f"{i}. {name}:")
        print(f"   Parameters: {params:,} ({reduction:+.1f}% vs Default)")
        print(f"   Model Size: {params * 4 / (1024**2):.2f} MB")
        
        if device == 'cuda' and 'error' not in stats['memory']:
            memory_mb = stats['memory']['memory_used_mb']
            inference_ms = stats['timing']['avg_inference_time_ms']
            fps = stats['timing']['fps']
            print(f"   GPU Memory: {memory_mb:.2f} MB")
            print(f"   Inference: {inference_ms:.2f} ms ({fps:.1f} FPS)")
        print()
    
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Speed ranking
    if device == 'cuda':
        speed_ranking = sorted(
            [(name, stats['timing']['avg_inference_time_ms']) 
             for name, stats in results.items() 
             if 'error' not in stats['timing']], 
            key=lambda x: x[1]
        )
        
        print("SPEED RANKING (Fastest to Slowest):")
        print("-" * 50)
        for i, (name, time_ms) in enumerate(speed_ranking, 1):
            fps = 1000 / time_ms
            print(f"{i}. {name}: {time_ms:.2f} ms ({fps:.1f} FPS)")
        print()
        
        # Memory efficiency ranking
        memory_ranking = sorted(
            [(name, stats['memory']['memory_used_mb']) 
             for name, stats in results.items() 
             if 'error' not in stats['memory']], 
            key=lambda x: x[1]
        )
        
        print("MEMORY EFFICIENCY RANKING (Least to Most GPU Memory):")
        print("-" * 60)
        for i, (name, memory_mb) in enumerate(memory_ranking, 1):
            print(f"{i}. {name}: {memory_mb:.2f} MB")
        print()
    
    # Architecture analysis
    print("=" * 80)
    print("ARCHITECTURE ANALYSIS")
    print("=" * 80)
    
    categories = {
        "Ultra Lightweight (< 100K params)": [],
        "Lightweight (100K - 1M params)": [],
        "Medium (1M - 10M params)": [],
        "Heavy (> 10M params)": []
    }
    
    for name, stats in results.items():
        params = stats["parameters"]
        if params < 100_000:
            categories["Ultra Lightweight (< 100K params)"].append((name, params))
        elif params < 1_000_000:
            categories["Lightweight (100K - 1M params)"].append((name, params))
        elif params < 10_000_000:
            categories["Medium (1M - 10M params)"].append((name, params))
        else:
            categories["Heavy (> 10M params)"].append((name, params))
    
    for category, models_in_cat in categories.items():
        if models_in_cat:
            print(f"{category}:")
            for name, params in sorted(models_in_cat, key=lambda x: x[1]):
                reduction = (1 - params / baseline_params) * 100
                print(f"  - {name}: {params:,} ({reduction:+.1f}% vs Default)")
            print()
    
    # UltraLight comparison
    print("=" * 80)
    print("ULTRALIGHT VARIANTS COMPARISON")
    print("=" * 80)
    
    ultra_v1_params = results["UltraLight (Original)"]["parameters"]
    ultra_v2_params = results["UltraLight_v2 (Enhanced)"]["parameters"]
    
    print(f"UltraLight (Original):    {ultra_v1_params:,} parameters")
    print(f"UltraLight_v2 (Enhanced): {ultra_v2_params:,} parameters")
    print(f"Difference: +{ultra_v2_params - ultra_v1_params:,} (+{((ultra_v2_params / ultra_v1_params - 1) * 100):.1f}%)")
    print()
    
    if device == 'cuda':
        v1_time = results["UltraLight (Original)"]["timing"]["avg_inference_time_ms"]
        v2_time = results["UltraLight_v2 (Enhanced)"]["timing"]["avg_inference_time_ms"]
        v1_memory = results["UltraLight (Original)"]["memory"]["memory_used_mb"]
        v2_memory = results["UltraLight_v2 (Enhanced)"]["memory"]["memory_used_mb"]
        
        print(f"Speed comparison:")
        print(f"  v1: {v1_time:.2f} ms ({1000/v1_time:.1f} FPS)")
        print(f"  v2: {v2_time:.2f} ms ({1000/v2_time:.1f} FPS)")
        print(f"  v2 is {((v2_time / v1_time - 1) * 100):+.1f}% slower")
        print()
        
        print(f"Memory comparison:")
        print(f"  v1: {v1_memory:.2f} MB")
        print(f"  v2: {v2_memory:.2f} MB")
        print(f"  v2 uses {((v2_memory / v1_memory - 1) * 100):+.1f}% more memory")
        print()
    
    if ultra_v2_params <= ultra_v1_params * 1.5:  # Allow 50% increase
        print("VERDICT: UltraLight_v2 maintains reasonable lightweight characteristics")
        print(f"Trade-off: +{((ultra_v2_params / ultra_v1_params - 1) * 100):.1f}% complexity for potential PSNR improvement")
    else:
        print("VERDICT: UltraLight_v2 significantly heavier than v1")
        print(f"Consider optimization: +{((ultra_v2_params / ultra_v1_params - 1) * 100):.1f}% complexity increase")
    
    print(f"\nBoth UltraLight variants remain {((1 - ultra_v2_params / baseline_params) * 100):.1f}% lighter than VGGStyle default!")

if __name__ == "__main__":
    compare_discriminators()