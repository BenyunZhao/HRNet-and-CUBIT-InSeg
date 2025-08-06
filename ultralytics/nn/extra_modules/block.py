import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union
from einops import rearrange, reduce
from ..modules.conv import Conv, DWConv, RepConv, autopad
from ..modules.block import *
from ..modules.block import C3k
from .rep_block import *

from ultralytics.utils.ops import make_divisible


from .mamba_vss import SS2D

__all__ = [
    'MBRConv',
    'MBR_HGBlock',
    'RSS_Attn_SS2D_Vmamba',
    'RSS_SS2D_Vmamba',
    ]


class MBRConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        # self.conv2 = RepConv(c2, c2, k, g=c2, act=act)

        # self.conv2 = DiverseBranchBlock(c2, c2, k, groups=c2)

        # self.conv2 = WideDiverseBranchBlock(c2, c2, k, internal_channels_1x1_3x3=c2*2, groups=c2)

        self.conv2 = MultiBranchReparam(c2, c2, k, internal_channels_1x1_3x3=c2*2, groups=c2)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class MBR_HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=True):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = MBRConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y



######################################## RSS attn with SS2D from Vmamba start ########################################

class RSS_Attn_SS2D_Vmamba(nn.Module):
    """
    
    Attributes:
        attn (SS2D): SS2D module from mamba_vss for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.
    
    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies SS2D attention and feed-forward processing to input tensor.
    """
    
    def __init__(self, dim, num_heads=None, mlp_ratio=1.2, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., bias=True, conv_bias=True, pscan=True):
        """
        
        Args:
            dim (int): Number of input channels.
            # num_heads (int): Number of heads (for compatibility, not used in SS2D).
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            # area (int): Area parameter (for compatibility, not used in SS2D).
            d_state (int): State dimension for SS2D.
            d_conv (int): Convolution kernel size for SS2D.
            expand (int): Expansion factor for SS2D.
            dt_rank (str or int): Rank for delta parameter.
            dt_min (float): Minimum value for delta.
            dt_max (float): Maximum value for delta.
            dt_init (str): Initialization method for delta.
            dt_scale (float): Scale factor for delta.
            dt_init_floor (float): Floor value for delta initialization.
            dropout (float): Dropout rate.
            bias (bool): Whether to use bias in linear layers.
            conv_bias (bool): Whether to use bias in convolution layers.
            pscan (bool): Whether to use parallel scan.
        """
        super().__init__()
        
        self.attn = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            dropout=dropout,
            conv_bias=conv_bias,
            bias=bias
        )
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1), 
            Conv(mlp_hidden_dim, dim, 1, act=False)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through ABlock with SS2D attention and MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        # SS2D (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        x = x + self.attn(x)
        
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        return x + self.mlp(x)


class RSS_SS2D_Vmamba(nn.Module):
    """
    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer.
        cv2 (Conv): Final 1x1 convolution layer.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling.
        m (nn.ModuleList): List of RSS_Attn_SS2D_Vmamba or C3k modules.
    
    Methods:
        forward: Processes input through SS2D-enhanced pathway.
    """
    
    def __init__(self, c1, c2, n=1, a2=True, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True, 
                 d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", 
                 dt_scale=1.0, dt_init_floor=1e-4, dropout=0., bias=True, conv_bias=True, pscan=True):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of RSS_Attn_SS2D_Vmamba or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks with SS2D.
            area (int): Number of areas for area attention. (for compatibility, not used in SS2D)
            residual (bool): Whether to use residual connections.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections.
            d_state (int): State dimension for SS2D.
            d_conv (int): Convolution kernel size for SS2D.
            expand (int): Expansion factor for SS2D.
            dt_rank (str or int): Rank for delta parameter.
            dt_min (float): Minimum value for delta.
            dt_max (float): Maximum value for delta.
            dt_init (str): Initialization method for delta.
            dt_scale (float): Scale factor for delta.
            dt_init_floor (float): Floor value for delta initialization.
            dropout (float): Dropout rate.
            bias (bool): Whether to use bias in linear layers.
            conv_bias (bool): Whether to use bias in convolution layers.
            pscan (bool): Whether to use parallel scan.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
                
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(RSS_Attn_SS2D_Vmamba(
                c_, c_ // 32, mlp_ratio, d_state, d_conv, expand, 
                dt_rank, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, 
                dropout, bias, conv_bias, pscan
            ) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )
        print('c1, c2, n, a2', c1, c2, n, a2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y
######################################## RSS attn with SS2D from Vmamba end ########################################