"""Minimal cnn building blocks for compatibility."""

from __future__ import annotations

import jittor as jt
import jittor.nn as nn


Conv1d = nn.Conv1d
Conv2d = nn.Conv
Conv3d = nn.Conv3d
Linear = nn.Linear


class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act'),
                 **kwargs):
        super().__init__()
        del with_spectral_norm, padding_mode, kwargs
        conv_cfg = conv_cfg or {}
        conv_type = conv_cfg.get('type', 'Conv2d') \
            if isinstance(conv_cfg, dict) else 'Conv2d'
        self._conv_dim = 1 if str(conv_type) in ('Conv1d',) else (
            3 if str(conv_type) in ('Conv3d',) else 2)
        if bias == 'auto':
            bias = norm_cfg is None
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.order = tuple(order) if order is not None else ('conv', 'norm', 'act')

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = None
        self.norm_name = None
        if self.with_norm:
            norm_channels = out_channels
            if 'norm' in self.order and 'conv' in self.order:
                if self.order.index('norm') < self.order.index('conv'):
                    norm_channels = in_channels
            self.norm_name, self.norm = build_norm_layer(
                norm_cfg, norm_channels, dim=self._conv_dim)
        self.activate = None
        if self.with_activation:
            act_cfg_local = dict(act_cfg) if isinstance(act_cfg, dict) else dict(type='ReLU')
            act_cfg_local.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_local)

    def execute(self, x):
        for op in self.order:
            if op == 'conv':
                x = self.conv(x)
            elif op == 'norm' and self.norm is not None:
                x = self.norm(x)
            elif op == 'act' and self.activate is not None:
                x = self.activate(x)
        return x


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = jt.array([float(scale)])

    def execute(self, x):
        return x * self.scale


class ContextBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def execute(self, x):
        return x


class NonLocal2d(nn.Module):

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 sub_sample=False,
                 **kwargs):
        super().__init__()
        del norm_cfg, kwargs
        self.in_channels = int(in_channels)
        self.reduction = max(1, int(reduction))
        self.inter_channels = max(1, self.in_channels // self.reduction)
        self.use_scale = bool(use_scale)
        self.mode = str(mode)
        self.sub_sample = bool(sub_sample)

        self.g = build_conv_layer(
            conv_cfg,
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.theta = build_conv_layer(
            conv_cfg,
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.phi = build_conv_layer(
            conv_cfg,
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.conv_out = build_conv_layer(
            conv_cfg,
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if self.sub_sample else None

    def _pairwise_logits(self, theta_x, phi_x):
        logits = jt.matmul(theta_x, phi_x)
        if self.use_scale:
            logits = logits / (float(theta_x.shape[-1])**0.5)
        return logits

    def gaussian(self, theta_x, phi_x):
        return nn.softmax(self._pairwise_logits(theta_x, phi_x), dim=-1)

    def embedded_gaussian(self, theta_x, phi_x):
        return nn.softmax(self._pairwise_logits(theta_x, phi_x), dim=-1)

    def dot_product(self, theta_x, phi_x):
        logits = self._pairwise_logits(theta_x, phi_x)
        return logits / max(1, int(logits.shape[-1]))

    def concatenation(self, theta_x, phi_x):
        return nn.softmax(self._pairwise_logits(theta_x, phi_x), dim=-1)

    def execute(self, x):
        n, _, h, w = x.shape
        g_x = self.g(x)
        if self.pool is not None:
            g_x = self.pool(g_x)
        g_x = g_x.reshape((n, self.inter_channels, -1)).permute(0, 2, 1)

        theta_x = self.theta(x).reshape((n, self.inter_channels, -1)).permute(0, 2, 1)
        phi_x = self.phi(x)
        if self.pool is not None:
            phi_x = self.pool(phi_x)
        phi_x = phi_x.reshape((n, self.inter_channels, -1))

        pairwise_func = getattr(self, self.mode, self.embedded_gaussian)
        pairwise_weight = pairwise_func(theta_x, phi_x)

        y = jt.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape((n, self.inter_channels, h, w))
        return x + self.conv_out(y)


def build_norm_layer(cfg, num_features, postfix='', dim=2):
    cfg = cfg or {}
    norm_type = cfg.get('type', 'BN')
    requires_grad = cfg.get('requires_grad', True)
    dim = int(cfg.get('dim', dim))
    if norm_type in ('BN', 'BN2d', 'SyncBN', 'BN1d', 'BN3d'):
        if dim == 1 and hasattr(nn, 'BatchNorm1d'):
            layer = nn.BatchNorm1d(num_features)
        elif dim == 3 and hasattr(nn, 'BatchNorm3d'):
            layer = nn.BatchNorm3d(num_features)
        else:
            layer = nn.BatchNorm(num_features)
        name = f'bn{postfix}'
    elif norm_type in ('LN', 'LayerNorm'):
        layer = nn.LayerNorm(num_features)
        name = f'ln{postfix}'
    elif norm_type in ('GN', 'GroupNorm'):
        num_groups = int(cfg.get('num_groups', 32))
        layer = nn.GroupNorm(num_groups, num_features)
        name = f'gn{postfix}'
    elif norm_type in ('IN', 'IN2d', 'InstanceNorm'):
        layer = nn.InstanceNorm(num_features)
        name = f'in{postfix}'
    else:
        layer = nn.BatchNorm(num_features)
        name = f'bn{postfix}'
    for p in layer.parameters():
        p.requires_grad = bool(requires_grad)
    return name, layer


def build_activation_layer(cfg):
    cfg = dict(cfg) if isinstance(cfg, dict) else {'type': 'ReLU'}
    t = str(cfg.pop('type', 'ReLU'))
    if t.lower() == 'relu':
        return nn.ReLU()
    if t.lower() == 'gelu':
        return nn.GELU()
    if t.lower() == 'quickgelu':
        try:
            from mmseg.registry import MODELS
            return MODELS.build(dict(type='QuickGELU'))
        except Exception:
            return nn.GELU()
    try:
        from mmseg.registry import MODELS
        return MODELS.build(dict(type=t, **cfg))
    except Exception:
        pass
    return nn.ReLU()


def build_conv_layer(cfg, *args, **kwargs):
    conv_type = 'Conv2d'
    if isinstance(cfg, dict):
        conv_type = cfg.get('type', 'Conv2d')
    conv_name = conv_type.__name__ if isinstance(conv_type, type) else str(conv_type)

    if conv_name in ('Conv1d',):
        return nn.Conv1d(*args, **kwargs)
    if conv_name in ('Conv2d', 'Conv', 'None'):
        return nn.Conv(*args, **kwargs)
    if conv_name in ('Conv3d',):
        return nn.Conv3d(*args, **kwargs)
    if conv_name in ('Conv2dAdaptivePadding',):
        from mmcv.cnn.bricks import Conv2dAdaptivePadding
        return Conv2dAdaptivePadding(*args, **kwargs)
    if conv_name in ('DeformConv2d', 'DeformConv'):
        from mmcv.ops import DeformConv2d
        return DeformConv2d(*args, **kwargs)
    if conv_name in ('DCN', 'DCNv1', 'DeformConvPack', 'DeformConv2dPack'):
        from mmcv.ops import DeformConv2dPack
        return DeformConv2dPack(*args, **kwargs)
    if conv_name in ('ModulatedDeformConv2d',):
        from mmcv.ops import ModulatedDeformConv2d
        return ModulatedDeformConv2d(*args, **kwargs)
    if conv_name in ('DCNv2', 'ModulatedDeformConv2dPack'):
        from mmcv.ops import ModulatedDeformConv2dPack
        return ModulatedDeformConv2dPack(*args, **kwargs)

    # Fallback to regular conv for unsupported custom layers.
    return nn.Conv(*args, **kwargs)


def build_upsample_layer(cfg, *args, **kwargs):
    if cfg is None:
        raise TypeError('cfg must be a dict, but got None')

    cfg = dict(cfg) if isinstance(cfg, dict) else dict(type=cfg)
    layer_type = cfg.pop('type', 'nearest')
    layer_name = layer_type.__name__ if isinstance(layer_type, type) else str(layer_type)
    layer_key = layer_name.lower()

    if layer_key in ('nearest', 'bilinear', 'upsample'):
        mode = cfg.pop('mode', layer_key if layer_key != 'upsample' else 'nearest')
        size = cfg.pop('size', None)
        scale_factor = cfg.pop('scale_factor', None)
        align_corners = cfg.pop('align_corners', None)
        cfg.pop('recompute_scale_factor', None)
        return nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners)

    if layer_key in ('deconv', 'deconv2d', 'convtranspose2d'):
        params = dict(cfg)
        params.update(kwargs)
        return nn.ConvTranspose2d(*args, **params)

    try:
        from mmseg.registry import MODELS
        return MODELS.build(dict(type=layer_type, **cfg, **kwargs))
    except Exception:
        pass

    raise KeyError(f'Unsupported upsample layer type: {layer_name}')


def build_plugin_layer(cfg, postfix='', **kwargs):
    del cfg, postfix, kwargs
    return 'identity', nn.Identity()


class DepthwiseSeparableConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        del kwargs
        super().__init__()
        self.depthwise = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.pointwise = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def execute(self, x):
        return self.pointwise(self.depthwise(x))
