import math

import jittor.nn as nn

from .drop import DropPath, build_dropout
from .registry import ATTENTION
from .transformer import (BaseTransformerLayer, FFN, MultiheadAttention,
                          build_transformer_layer)


class Conv2dAdaptivePadding(nn.Conv):
    """2D convolution with dynamic SAME-style padding.

    The layer computes runtime padding so output spatial size follows
    ``ceil(input / stride)`` when ``padding=0`` in config.
    """

    @staticmethod
    def _pair(v):
        if isinstance(v, (tuple, list)):
            return int(v[0]), int(v[1])
        return int(v), int(v)

    def _pad_shape(self, input_hw):
        in_h, in_w = int(input_hw[0]), int(input_hw[1])
        k_h, k_w = self._pair(self.kernel_size)
        s_h, s_w = self._pair(self.stride)
        d_h, d_w = self._pair(self.dilation)

        out_h = math.ceil(in_h / s_h)
        out_w = math.ceil(in_w / s_w)
        pad_h = max((out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h, 0)
        pad_w = max((out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w, 0)
        return pad_h, pad_w

    def execute(self, x):
        pad_h, pad_w = self._pad_shape(x.shape[-2:])
        if pad_h > 0 or pad_w > 0:
            x = nn.pad(
                x,
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ),
            )
        return super().execute(x)


__all__ = [
    'ATTENTION', 'build_dropout', 'BaseTransformerLayer', 'FFN',
    'MultiheadAttention', 'build_transformer_layer', 'DropPath',
    'Conv2dAdaptivePadding'
]
