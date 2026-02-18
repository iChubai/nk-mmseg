"""torch.nn compatibility layer backed by jittor.nn."""

from __future__ import annotations

import jittor as jt
import jittor.nn as _nn

from . import functional


class Module(_nn.Module):

    def execute(self, *args, **kwargs):
        if hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement execute/forward')


class ModuleList(_nn.ModuleList):
    """Torch-like ModuleList.

    In torch, ModuleList is only a container and is not called directly.
    But many mmseg modules subclass ModuleList and implement ``forward``.
    Jittor's ModuleList executes children sequentially, which is incompatible.
    """

    def __init__(self, modules=None):
        super().__init__([] if modules is None else modules)

    def __setitem__(self, idx, module):
        if not isinstance(idx, int):
            raise TypeError('ModuleList indices must be integers')
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError('ModuleList assignment index out of range')
        # Jittor ModuleList may append int-keyed entries on assignment.
        # Enforce torch-like replacement into the existing string key.
        key = str(idx)
        if hasattr(self, 'layers'):
            self.layers[key] = module
            if idx in self.layers:
                del self.layers[idx]
            return
        raise TypeError('Underlying ModuleList has no mutable layer storage')

    def execute(self, *args, **kwargs):
        if self.__class__ is not ModuleList and hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        raise NotImplementedError(
            'ModuleList is a container and has no execute/forward by default')


Sequential = _nn.Sequential
Identity = _nn.Identity


class ReLU(_nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self._op = _nn.ReLU()

    def execute(self, x):
        return self._op(x)


class GELU(_nn.Module):

    def __init__(self, approximate='none'):
        del approximate
        super().__init__()
        self._op = _nn.GELU()

    def execute(self, x):
        return self._op(x)


class SiLU(_nn.Module):

    def __init__(self, inplace=False):
        del inplace
        super().__init__()
        self._op = getattr(_nn, 'SiLU', _nn.ReLU)()

    def execute(self, x):
        return self._op(x)


Sigmoid = _nn.Sigmoid
Softmax = _nn.Softmax
Dropout = _nn.Dropout
Dropout2d = _nn.Dropout2d
Linear = _nn.Linear
Conv1d = _nn.Conv1d
Conv2d = _nn.Conv
ConvTranspose2d = _nn.ConvTranspose2d
Conv3d = _nn.Conv3d
BatchNorm1d = _nn.BatchNorm1d
BatchNorm2d = _nn.BatchNorm
BatchNorm3d = _nn.BatchNorm3d
InstanceNorm1d = _nn.InstanceNorm1d
InstanceNorm2d = _nn.InstanceNorm
InstanceNorm3d = _nn.InstanceNorm3d
LayerNorm = _nn.LayerNorm
GroupNorm = _nn.GroupNorm
MaxPool2d = _nn.MaxPool2d
AvgPool2d = _nn.AvgPool2d
AdaptiveAvgPool2d = _nn.AdaptiveAvgPool2d
AdaptiveMaxPool2d = _nn.AdaptiveMaxPool2d
Embedding = _nn.Embedding
PReLU = _nn.PReLU
LeakyReLU = _nn.LeakyReLU
PixelShuffle = _nn.PixelShuffle
Mish = _nn.Mish


class Upsample(_nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None,
                 recompute_scale_factor=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, (tuple, list)):
            self.scale_factor = tuple(float(v) for v in scale_factor)
        elif scale_factor is None:
            self.scale_factor = None
        else:
            self.scale_factor = float(scale_factor)
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def execute(self, x):
        return functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)


class Unfold(_nn.Module):

    def __init__(self,
                 kernel_size,
                 dilation=1,
                 padding=0,
                 stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def execute(self, x):
        return _nn.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride)


class Fold(_nn.Module):

    def __init__(self,
                 output_size,
                 kernel_size,
                 dilation=1,
                 padding=0,
                 stride=1):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def execute(self, x):
        return _nn.fold(
            x,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride)


ParameterList = list


def Parameter(x, requires_grad=True):
    if isinstance(x, jt.Var):
        x.requires_grad = requires_grad
        return x
    t = jt.array(x)
    t.requires_grad = requires_grad
    return t


class init:
    xavier_uniform_ = staticmethod(_nn.init.xavier_uniform_)
    xavier_normal_ = staticmethod(_nn.init.xavier_gauss_)
    kaiming_uniform_ = staticmethod(_nn.init.kaiming_uniform_)
    kaiming_normal_ = staticmethod(_nn.init.kaiming_normal_)
    constant_ = staticmethod(_nn.init.constant_)
    normal_ = staticmethod(_nn.init.gauss_)


__all__ = [k for k in list(globals().keys()) if not k.startswith('_')]
