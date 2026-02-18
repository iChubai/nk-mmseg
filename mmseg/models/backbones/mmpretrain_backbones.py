# Copyright (c) OpenMMLab. All rights reserved.
"""Pure-Jittor compatibility backbones for common mmpretrain types."""

from __future__ import annotations

from typing import Sequence

import jittor as jt
import jittor.nn as nn

from mmcv.cnn.bricks.drop import DropPath
from mmengine.model import BaseModule

from mmseg.registry import MODELS


def _parameter(value, requires_grad: bool = True):
    """Torch-like parameter helper without jittor.nn.Parameter warning spam."""
    if isinstance(value, jt.Var):
        out = value
    else:
        out = jt.array(value)
    out.requires_grad = bool(requires_grad)
    return out


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = _parameter(jt.ones((channels, )))
        self.bias = _parameter(jt.zeros((channels, )))
        self.eps = float(eps)

    def execute(self, x):
        mean = x.mean(dim=1, keepdims=True)
        var = ((x - mean)**2).mean(dim=1, keepdims=True)
        x = (x - mean) / jt.sqrt(var + self.eps)
        return x * self.weight.reshape((1, -1, 1, 1)) + \
            self.bias.reshape((1, -1, 1, 1))


class ConvNeXtBlock(BaseModule):

    def __init__(self,
                 channels: int,
                 drop_path: float = 0.0,
                 layer_scale_init_value: float = 1.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=7, stride=1, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels, eps=1e-6)
        self.pwconv1 = nn.Conv2d(channels, 4 * channels, kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * channels, channels, kernel_size=1, stride=1)
        self.drop_path = DropPath(drop_path) if float(drop_path) > 0 else nn.Identity()
        if layer_scale_init_value is not None and float(layer_scale_init_value) > 0:
            self.gamma = _parameter(
                jt.ones((channels, )) * float(layer_scale_init_value))
        else:
            self.gamma = None

    def execute(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.reshape((1, -1, 1, 1))
        x = identity + self.drop_path(x)
        return x


_CONVNEXT_ARCH_SETTINGS = {
    'tiny': dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    'small': dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    'base': dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    'large': dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    'xlarge': dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]),
}


def _parse_arch(arch, default_settings):
    if isinstance(arch, str):
        if arch not in default_settings:
            raise KeyError(f'Unsupported arch: {arch}')
        cfg = default_settings[arch]
        return list(cfg['depths']), list(cfg['dims'])

    if isinstance(arch, dict):
        depths = arch.get('depths', arch.get('depth', None))
        dims = arch.get('dims', arch.get('channels', arch.get('embed_dims', None)))
        if depths is None or dims is None:
            raise KeyError('Custom arch must provide "depths" and "dims/channels/embed_dims".')
        return list(depths), list(dims)

    raise TypeError(f'Unsupported arch type: {type(arch)}')


@MODELS.register_module(name=['ConvNeXt', 'mmpretrain.ConvNeXt'])
class ConvNeXt(BaseModule):
    """Lightweight ConvNeXt backbone compatible with mmpretrain configs."""

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 drop_path_rate=0.0,
                 layer_scale_init_value=1.0,
                 gap_before_final_norm=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        del gap_before_final_norm, kwargs
        super().__init__(init_cfg=init_cfg)
        depths, dims = _parse_arch(arch, _CONVNEXT_ARCH_SETTINGS)

        self.out_indices = tuple(out_indices)
        self.frozen_stages = int(frozen_stages)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0),
            LayerNorm2d(dims[0], eps=1e-6))

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(
                        dims[i], dims[i + 1], kernel_size=2, stride=2, padding=0)))

        total_depth = int(sum(depths))
        dpr = jt.linspace(0.0, float(drop_path_rate), total_depth).numpy().tolist()

        self.stages = nn.ModuleList()
        dp_offset = 0
        for i in range(4):
            blocks = []
            for j in range(int(depths[i])):
                blocks.append(
                    ConvNeXtBlock(
                        channels=dims[i],
                        drop_path=float(dpr[dp_offset + j]),
                        layer_scale_init_value=layer_scale_init_value))
            self.stages.append(nn.Sequential(*blocks))
            dp_offset += int(depths[i])

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        if self.frozen_stages >= 0:
            self.stem.eval()
            for p in self.stem.parameters():
                p.requires_grad = False
        for i in range(min(self.frozen_stages, len(self.stages) - 1) + 1):
            self.stages[i].eval()
            for p in self.stages[i].parameters():
                p.requires_grad = False
            if i < len(self.downsample_layers):
                self.downsample_layers[i].eval()
                for p in self.downsample_layers[i].parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def execute(self, x):
        x = self.stem(x)
        outs = []
        for i in range(4):
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
            if i < 3:
                x = self.downsample_layers[i](x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)


class PoolFormerBlock(BaseModule):

    def __init__(self,
                 channels: int,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0,
                 drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(float(mlp_ratio) * channels)
        self.norm1 = LayerNorm2d(channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Dropout(float(drop)),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1),
            nn.Dropout(float(drop)))
        self.drop_path = DropPath(drop_path) if float(drop_path) > 0 else nn.Identity()
        if layer_scale_init_value is not None and float(layer_scale_init_value) > 0:
            self.layer_scale_1 = _parameter(
                jt.ones((channels, )) * float(layer_scale_init_value))
            self.layer_scale_2 = _parameter(
                jt.ones((channels, )) * float(layer_scale_init_value))
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def execute(self, x):
        identity = x
        x1 = self.norm1(x)
        x1 = self.pool(x1) - x1
        if self.layer_scale_1 is not None:
            x1 = x1 * self.layer_scale_1.reshape((1, -1, 1, 1))
        x = identity + self.drop_path(x1)

        identity = x
        x2 = self.norm2(x)
        x2 = self.mlp(x2)
        if self.layer_scale_2 is not None:
            x2 = x2 * self.layer_scale_2.reshape((1, -1, 1, 1))
        x = identity + self.drop_path(x2)
        return x


_POOLFORMER_ARCH_SETTINGS = {
    's12': dict(layers=[2, 2, 6, 2], embed_dims=[64, 128, 320, 512]),
    's24': dict(layers=[4, 4, 12, 4], embed_dims=[64, 128, 320, 512]),
    's36': dict(layers=[6, 6, 18, 6], embed_dims=[64, 128, 320, 512]),
    'm36': dict(layers=[6, 6, 18, 6], embed_dims=[96, 192, 384, 768]),
    'm48': dict(layers=[8, 8, 24, 8], embed_dims=[96, 192, 384, 768]),
}


@MODELS.register_module(name=['PoolFormer', 'mmpretrain.PoolFormer'])
class PoolFormer(BaseModule):
    """Lightweight PoolFormer backbone compatible with mmpretrain configs."""

    def __init__(self,
                 arch='s12',
                 in_channels=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 out_indices=(0, 2, 4, 6),
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-5,
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        del kwargs
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            if arch not in _POOLFORMER_ARCH_SETTINGS:
                raise KeyError(f'Unsupported arch: {arch}')
            arch_cfg = _POOLFORMER_ARCH_SETTINGS[arch]
            layers = list(arch_cfg['layers'])
            embed_dims = list(arch_cfg['embed_dims'])
        elif isinstance(arch, dict):
            layers = list(arch.get('layers', arch.get('depths', [])))
            embed_dims = list(arch.get('embed_dims', arch.get('channels', [])))
            if not layers or not embed_dims:
                raise KeyError('Custom PoolFormer arch must provide layers and embed_dims.')
        else:
            raise TypeError(f'Unsupported arch type: {type(arch)}')

        self.out_indices = tuple(out_indices)
        self.frozen_stages = int(frozen_stages)

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dims[0],
            kernel_size=int(in_patch_size),
            stride=int(in_stride),
            padding=int(in_pad))

        total_depth = int(sum(layers))
        dpr = jt.linspace(0.0, float(drop_path_rate), total_depth).numpy().tolist()

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_offset = 0
        for i, (depth, channels) in enumerate(zip(layers, embed_dims)):
            blocks = []
            for j in range(int(depth)):
                blocks.append(
                    PoolFormerBlock(
                        channels=channels,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        drop_path=float(dpr[dp_offset + j]),
                        layer_scale_init_value=layer_scale_init_value))
            self.stages.append(nn.Sequential(*blocks))
            dp_offset += int(depth)
            if i < len(embed_dims) - 1:
                self.downsamples.append(
                    nn.Conv2d(
                        embed_dims[i],
                        embed_dims[i + 1],
                        kernel_size=int(down_patch_size),
                        stride=int(down_stride),
                        padding=int(down_pad)))

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        self.patch_embed.eval()
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        for i in range(min(self.frozen_stages, len(self.stages) - 1) + 1):
            self.stages[i].eval()
            for p in self.stages[i].parameters():
                p.requires_grad = False
            if i < len(self.downsamples):
                self.downsamples[i].eval()
                for p in self.downsamples[i].parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def execute(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            stage_id = i * 2
            if i in self.out_indices or stage_id in self.out_indices:
                outs.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)
