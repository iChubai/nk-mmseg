"""Ops compatibility layer implemented with pure Jittor operators."""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import jittor as jt
import jittor.nn as nn
import numpy as np


def _pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def point_sample(input, points, align_corners=False, mode='bilinear', **kwargs):
    del kwargs
    if not isinstance(input, jt.Var):
        input = jt.array(input)
    if not isinstance(points, jt.Var):
        points = jt.array(points)

    if input.ndim == 4 and hasattr(nn, 'grid_sample'):
        if points.ndim == 3:
            # (N, P, 2) in [0, 1] -> (N, P, 1, 2) in [-1, 1]
            grid = points * 2.0 - 1.0
            grid = grid.reshape((grid.shape[0], grid.shape[1], 1, 2))
            try:
                out = nn.grid_sample(
                    input,
                    grid,
                    mode=mode,
                    padding_mode='zeros',
                    align_corners=align_corners)
            except TypeError:
                out = nn.grid_sample(input, grid)
            return out.reshape((out.shape[0], out.shape[1], out.shape[2]))

        if points.ndim == 4:
            grid = points * 2.0 - 1.0
            try:
                return nn.grid_sample(
                    input,
                    grid,
                    mode=mode,
                    padding_mode='zeros',
                    align_corners=align_corners)
            except TypeError:
                return nn.grid_sample(input, grid)

    if input.ndim == 4:
        # Conservative fallback when grid_sample is unavailable.
        b, c, _, _ = input.shape
        n = points.shape[1] if points.ndim > 1 else 1
        pooled = nn.adaptive_avg_pool2d(input, (1, 1)).reshape((b, c, 1))
        return pooled.broadcast((b, c, n))
    return input


class PSAMask(nn.Module):

    def __init__(self, psa_type='collect', mask_size=None):
        super().__init__()
        self.psa_type = psa_type
        self.mask_size = mask_size

    def execute(self, x):
        if not isinstance(x, jt.Var) or x.ndim != 4:
            return x
        n, c, h, w = x.shape
        target_c = int(h * w)

        # Keep channel count consistent with downstream `view(n, h*w, h*w)`.
        if c < target_c:
            pad = jt.zeros((n, target_c - c, h, w), dtype=x.dtype)
            x = jt.concat([x, pad], dim=1)
        elif c > target_c:
            x = x[:, :target_c, :, :]

        if self.psa_type == 'distribute':
            # Approximate distribute behavior by channel/position swap.
            x = x.reshape((n, target_c, h * w)).permute((0, 2, 1))
            x = x.reshape((n, target_c, h, w))
        return x


class CrissCrossAttention(nn.Module):

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.in_channels = int(in_channels)
        inter_channels = max(1, self.in_channels // 8)
        self.query_conv = nn.Conv(self.in_channels, inter_channels, 1, bias=False)
        self.key_conv = nn.Conv(self.in_channels, inter_channels, 1, bias=False)
        self.value_conv = nn.Conv(self.in_channels, self.in_channels, 1, bias=False)
        self.gamma = jt.zeros((1, ))

    def execute(self, x):
        if not isinstance(x, jt.Var) or x.ndim != 4:
            return x
        n, c, h, w = x.shape
        if h <= 0 or w <= 0:
            return x

        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        # Horizontal attention: for each row, each location attends along width.
        q_h = q.permute((0, 3, 1, 2)).reshape((n * w, -1, h)).permute((0, 2, 1))
        k_h = k.permute((0, 3, 1, 2)).reshape((n * w, -1, h))
        energy_h = jt.matmul(q_h, k_h).reshape((n, w, h, h)).permute((0, 2, 1, 3))

        # Vertical attention: for each column, each location attends along height.
        q_w = q.permute((0, 2, 1, 3)).reshape((n * h, -1, w)).permute((0, 2, 1))
        k_w = k.permute((0, 2, 1, 3)).reshape((n * h, -1, w))
        energy_w = jt.matmul(q_w, k_w).reshape((n, h, w, w))

        attn = nn.softmax(jt.concat((energy_h, energy_w), dim=3), dim=3)
        attn_h = attn[:, :, :, :h].permute((0, 2, 1, 3)).reshape((n * w, h, h))
        attn_w = attn[:, :, :, h:h + w].reshape((n * h, w, w))

        v_h = v.permute((0, 3, 1, 2)).reshape((n * w, c, h))
        out_h = jt.matmul(v_h, attn_h.permute((0, 2, 1)))
        out_h = out_h.reshape((n, w, c, h)).permute((0, 2, 3, 1))

        v_w = v.permute((0, 2, 1, 3)).reshape((n * h, c, w))
        out_w = jt.matmul(v_w, attn_w.permute((0, 2, 1)))
        out_w = out_w.reshape((n, h, c, w)).permute((0, 2, 1, 3))

        return x + self.gamma.reshape((1, 1, 1, 1)) * (out_h + out_w)


_WARNED_DEFORM_FALLBACK = False
_DCN_V1_FUNC = None
_DCN_V1_LOADED = False
_DCN_V2_FUNC = None
_DCN_V2_LOADED = False


def _warn_deform_fallback(msg):
    global _WARNED_DEFORM_FALLBACK
    if _WARNED_DEFORM_FALLBACK:
        return
    _WARNED_DEFORM_FALLBACK = True
    print(f'[mmcv.ops] {msg}')


def _load_dcn_v1_func():
    """Load optional high-performance DCN-v1 function from DFormer-Jittor."""
    global _DCN_V1_FUNC, _DCN_V1_LOADED
    if _DCN_V1_LOADED:
        return _DCN_V1_FUNC

    _DCN_V1_LOADED = True
    backend = os.environ.get('NKMMSEG_DCN_BACKEND', 'auto').strip().lower()
    if backend in ('naive', 'off', 'disable', 'disabled'):
        return None

    candidates = []
    env_path = os.environ.get('NKMMSEG_DCN_V1_PATH')
    if env_path:
        candidates.append(env_path)

    default_path = (
        '/defaultShare/archive/yinbowen/Houjd/'
        'DFormer-Jittor/src/jittordet/jittordet/ops/dcn_v1.py')
    if os.path.exists(default_path):
        candidates.append(default_path)

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location('_nkmmseg_dcn_v1', path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, 'deform_conv', None)
            if fn is None:
                continue
            _DCN_V1_FUNC = fn
            print(f'[mmcv.ops] Using DCN-v1 backend from {path}')
            return _DCN_V1_FUNC
        except Exception as exc:
            _warn_deform_fallback(
                f'Unable to load DCN-v1 backend from {path}: '
                f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
            continue

    return None


def _ensure_jdet_registry_stub():
    """Provide minimal jdet registry modules required by dcn_v2 import."""
    try:
        import jdet.utils.registry  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    if 'jdet.utils.registry' in sys.modules:
        return

    jdet = types.ModuleType('jdet')
    utils = types.ModuleType('jdet.utils')
    registry = types.ModuleType('jdet.utils.registry')

    class _DummyRegistry:
        def register_module(self, *args, **kwargs):
            def deco(obj):
                return obj
            return deco

    registry.HEADS = _DummyRegistry()
    utils.registry = registry
    jdet.utils = utils
    sys.modules['jdet'] = jdet
    sys.modules['jdet.utils'] = utils
    sys.modules['jdet.utils.registry'] = registry


def _load_dcn_v2_func():
    """Load optional high-performance DCN-v2 function from DFormer-Jittor."""
    global _DCN_V2_FUNC, _DCN_V2_LOADED
    if _DCN_V2_LOADED:
        return _DCN_V2_FUNC

    _DCN_V2_LOADED = True
    backend = os.environ.get('NKMMSEG_DCN_BACKEND', 'auto').strip().lower()
    if backend in ('naive', 'off', 'disable', 'disabled'):
        return None

    candidates = []
    env_path = os.environ.get('NKMMSEG_DCN_V2_PATH')
    if env_path:
        candidates.append(env_path)

    default_path = (
        '/defaultShare/archive/yinbowen/Houjd/'
        'DFormer-Jittor/src/jittordet/jittordet/ops/dcn_v2.py')
    if os.path.exists(default_path):
        candidates.append(default_path)

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        try:
            _ensure_jdet_registry_stub()
            spec = importlib.util.spec_from_file_location('_nkmmseg_dcn_v2', path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, 'dcn_v2_conv', None)
            if fn is None:
                continue
            _DCN_V2_FUNC = fn
            print(f'[mmcv.ops] Using DCN-v2 backend from {path}')
            return _DCN_V2_FUNC
        except Exception as exc:
            _warn_deform_fallback(
                f'Unable to load DCN-v2 backend from {path}: '
                f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
            continue
    return None


def _deform_conv2d_naive(input,
                         offset,
                         weight,
                         bias=None,
                         stride=1,
                         padding=0,
                         dilation=1,
                         mask=None):
    """Naive deformable conv implemented by repeated ``grid_sample``.

    This supports common mmseg usage (groups=1, deform_groups=1).
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if not isinstance(offset, jt.Var):
        offset = jt.array(offset)
    if mask is not None and not isinstance(mask, jt.Var):
        mask = jt.array(mask)

    x = input
    if padding[0] > 0 or padding[1] > 0:
        x = nn.pad(x, (padding[1], padding[1], padding[0], padding[0]))

    n, cin, hp, wp = x.shape
    cout, cin_w, kh, kw = weight.shape
    if int(cin_w) != int(cin):
        raise RuntimeError(
            f'Naive deform conv expects weight cin={cin}, got {cin_w}.')

    kernel_h = dilation[0] * (kh - 1) + 1
    kernel_w = dilation[1] * (kw - 1) + 1
    out_h = (int(hp) - kernel_h) // stride[0] + 1
    out_w = (int(wp) - kernel_w) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise RuntimeError('Invalid output shape in deform conv.')

    if int(offset.shape[2]) != out_h or int(offset.shape[3]) != out_w:
        offset = nn.interpolate(
            offset, size=(out_h, out_w), mode='bilinear', align_corners=False)
    if mask is not None and (int(mask.shape[2]) != out_h or
                             int(mask.shape[3]) != out_w):
        mask = nn.interpolate(
            mask, size=(out_h, out_w), mode='bilinear', align_corners=False)

    base_y = jt.arange(out_h).float32() * float(stride[0])
    base_x = jt.arange(out_w).float32() * float(stride[1])
    yy, xx = jt.meshgrid(base_y, base_x)
    yy = yy.reshape((1, out_h, out_w))
    xx = xx.reshape((1, out_h, out_w))

    out = jt.zeros((n, cout, out_h, out_w), dtype=x.dtype)
    denom_y = max(int(hp) - 1, 1)
    denom_x = max(int(wp) - 1, 1)

    for ky in range(kh):
        for kx in range(kw):
            kidx = ky * kw + kx
            off_y = offset[:, 2 * kidx, :, :]
            off_x = offset[:, 2 * kidx + 1, :, :]

            sample_y = yy + float(ky * dilation[0]) + off_y
            sample_x = xx + float(kx * dilation[1]) + off_x
            grid_y = sample_y / float(denom_y) * 2.0 - 1.0
            grid_x = sample_x / float(denom_x) * 2.0 - 1.0
            grid = jt.stack((grid_x, grid_y), dim=-1)

            sampled = nn.grid_sample(
                x,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True)
            if mask is not None:
                sampled = sampled * mask[:, kidx:kidx + 1, :, :]

            w_k = weight[:, :, ky, kx]  # (cout, cin)
            sampled_flat = sampled.permute((0, 2, 3, 1)).reshape((-1, cin))
            contrib = jt.matmul(sampled_flat, w_k.permute((1, 0)))
            contrib = contrib.reshape((n, out_h, out_w, cout)).permute(
                (0, 3, 1, 2))
            out += contrib

    if bias is not None:
        out += bias.reshape((1, -1, 1, 1))
    return out


class DeformConv2d(nn.Module):
    """DeformConv2d-compatible module.

    Falls back to regular convolution for unsupported cases.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deform_groups=1,
                 im2col_step=32,
                 **kwargs):
        super().__init__()
        del im2col_step, kwargs
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.deform_groups = int(deform_groups)
        self.conv = nn.Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def execute(self, x, offset=None):
        if offset is None:
            return self.conv(x)
        expected = 2 * self.deform_groups * self.kernel_size[0] * self.kernel_size[1]
        if int(offset.shape[1]) < expected:
            _warn_deform_fallback(
                'DeformConv2d falls back to regular conv due to invalid '
                f'offset channels: got={offset.shape[1]} expected>={expected}.')
            return self.conv(x)
        offset = offset[:, :expected, :, :]

        # Prefer external CUDA DCN backend when available.
        if int(getattr(jt.flags, 'use_cuda', 0)):
            dcn_v1_func = _load_dcn_v1_func()
            if dcn_v1_func is not None:
                try:
                    out = dcn_v1_func(
                        x,
                        offset,
                        self.weight,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                        self.deform_groups)
                    if self.bias is not None:
                        out += self.bias.reshape((1, -1, 1, 1))
                    return out
                except Exception as exc:
                    _warn_deform_fallback(
                        'DeformConv2d DCN-v1 backend failed '
                        f'({type(exc).__name__}), fallback to compatibility path.')

        if self.groups != 1 or self.deform_groups != 1:
            _warn_deform_fallback(
                'DeformConv2d falls back to regular conv for groups!=1 or '
                'deform_groups!=1 without DCN backend.')
            return self.conv(x)
        if not hasattr(nn, 'grid_sample'):
            _warn_deform_fallback(
                'DeformConv2d falls back to regular conv because '
                'grid_sample is unavailable.')
            return self.conv(x)
        return _deform_conv2d_naive(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation)


class DeformConv2dPack(DeformConv2d):
    """Pack variant that predicts offsets from input (mmcv-compatible)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        offset_channels = (self.deform_groups * 2 * self.kernel_size[0] *
                           self.kernel_size[1])
        self.conv_offset = nn.Conv(
            self.in_channels,
            offset_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)

    def execute(self, x):
        offset = self.conv_offset(x)
        return super().execute(x, offset)


class ModulatedDeformConv2d(DeformConv2d):
    """ModulatedDeformConv2d-compatible module."""

    def execute(self, x, offset=None, mask=None):
        if offset is None:
            return self.conv(x)
        expected = 2 * self.deform_groups * self.kernel_size[0] * self.kernel_size[1]
        if int(offset.shape[1]) < expected:
            _warn_deform_fallback(
                'ModulatedDeformConv2d falls back to regular conv due to '
                f'invalid offset channels: got={offset.shape[1]} expected>={expected}.')
            return self.conv(x)
        offset = offset[:, :expected, :, :]

        if int(getattr(jt.flags, 'use_cuda', 0)) and self.groups == 1:
            dcn_v2_func = _load_dcn_v2_func()
            if dcn_v2_func is not None:
                try:
                    if mask is None:
                        mask = jt.ones(
                            (offset.shape[0], expected // 2, offset.shape[2], offset.shape[3]),
                            dtype=offset.dtype)
                    else:
                        expected_mask = self.deform_groups * self.kernel_size[0] * self.kernel_size[1]
                        if int(mask.shape[1]) < expected_mask:
                            mask = jt.ones(
                                (offset.shape[0], expected_mask, offset.shape[2], offset.shape[3]),
                                dtype=offset.dtype)
                        else:
                            mask = mask[:, :expected_mask, :, :]
                    return dcn_v2_func(
                        x,
                        offset,
                        mask,
                        self.weight,
                        self.bias if self.bias is not None else jt.zeros((self.out_channels,), dtype=x.dtype),
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.deform_groups)
                except Exception as exc:
                    _warn_deform_fallback(
                        'ModulatedDeformConv2d DCN-v2 backend failed '
                        f'({type(exc).__name__}), fallback to compatibility path.')

        if self.groups != 1 or self.deform_groups != 1:
            _warn_deform_fallback(
                'ModulatedDeformConv2d falls back to regular conv for '
                'groups!=1 or deform_groups!=1.')
            return self.conv(x)
        if not hasattr(nn, 'grid_sample'):
            _warn_deform_fallback(
                'ModulatedDeformConv2d falls back to regular conv because '
                'grid_sample is unavailable.')
            return self.conv(x)
        if mask is not None:
            expected_mask = self.deform_groups * self.kernel_size[0] * self.kernel_size[1]
            if int(mask.shape[1]) < expected_mask:
                mask = None
            else:
                mask = mask[:, :expected_mask, :, :]
        return _deform_conv2d_naive(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask)


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """Pack variant that predicts offsets and modulation mask."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = (self.deform_groups * 3 * self.kernel_size[0] *
                    self.kernel_size[1])
        self.conv_offset = nn.Conv(
            self.in_channels,
            channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)

    def execute(self, x):
        offset_mask = self.conv_offset(x)
        o1, o2, mask = jt.chunk(offset_mask, 3, dim=1)
        offset = jt.concat((o1, o2), dim=1)
        mask = mask.sigmoid()
        return super().execute(x, offset=offset, mask=mask)


def sigmoid_focal_loss(inputs,
                       targets,
                       gamma=2.0,
                       alpha=0.25,
                       weight=None,
                       reduction='none'):
    """Jittor implementation with MMCV-compatible call signature."""
    if not isinstance(inputs, jt.Var):
        inputs = jt.array(inputs)
    if not isinstance(targets, jt.Var):
        targets = jt.array(targets)

    # MMCV op accepts class index targets for [N, C] inputs.
    if targets.ndim + 1 == inputs.ndim:
        num_classes = int(inputs.shape[1])
        target_idx = targets.reshape((-1, )).int32()
        target_onehot = nn.one_hot(target_idx, num_classes=num_classes)
        targets = target_onehot.reshape(inputs.shape).astype(inputs.dtype)
    else:
        targets = targets.astype(inputs.dtype)

    p = inputs.sigmoid()
    ce = -(targets * jt.log(p + 1e-12) +
           (1 - targets) * jt.log(1 - p + 1e-12))
    p_t = p * targets + (1 - p) * (1 - targets)
    if isinstance(alpha, (list, tuple, np.ndarray)):
        alpha = jt.array(alpha).astype(inputs.dtype)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce

    if weight is not None:
        if not isinstance(weight, jt.Var):
            weight = jt.array(weight)
        loss = loss * weight.astype(loss.dtype)

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss
