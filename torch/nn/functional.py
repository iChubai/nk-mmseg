"""torch.nn.functional compatibility layer."""

from __future__ import annotations

import jittor as jt
import jittor.nn as nn


class _Reduction:

    @staticmethod
    def get_enum(reduction):
        if reduction == 'none':
            return 0
        if reduction == 'mean':
            return 1
        if reduction == 'sum':
            return 2
        raise ValueError(f'Unsupported reduction: {reduction}')


def relu(x, inplace=False):
    del inplace
    return nn.relu(x)


def relu_(x):
    return nn.relu(x)


def gelu(x):
    return nn.gelu(x)


def sigmoid(x):
    return x.sigmoid()


def softmax(x, dim=-1):
    return nn.softmax(x, dim=dim)


def log_softmax(x, dim=-1):
    return nn.log_softmax(x, dim=dim)


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
    return nn.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def pad(input, pad, mode='constant', value=0):
    return nn.pad(input, pad, mode=mode, value=value)


def linear(input, weight, bias=None):
    return nn.linear(input, weight, bias)


def conv2d(input,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1):
    return nn.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups)


def normalize(input, p=2.0, dim=1, eps=1e-12):
    p = float(p)
    norm = (jt.abs(input)**p).sum(dim=dim, keepdims=True)**(1.0 / p)
    return input / (norm + float(eps))


def adaptive_avg_pool2d(input, output_size):
    return nn.AdaptiveAvgPool2d(output_size)(input)


def adaptive_max_pool2d(input, output_size):
    return nn.AdaptiveMaxPool2d(output_size)(input)


def max_pool2d(input,
               kernel_size,
               stride=None,
               padding=0,
               dilation=1,
               ceil_mode=False):
    if stride is None:
        stride = kernel_size
    return nn.max_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode)


def dropout(input, p=0.5, training=True, inplace=False):
    del inplace
    if not training or float(p) <= 0.0:
        return input
    return nn.dropout(input, p=float(p))


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    return nn.unfold(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride)


def binary_cross_entropy_with_logits(input,
                                     target,
                                     weight=None,
                                     pos_weight=None,
                                     reduction='mean'):
    pred = input.sigmoid()
    target = target.astype(pred.dtype)
    eps = 1e-12
    loss = -(target * jt.log(pred + eps) +
             (1 - target) * jt.log(1 - pred + eps))

    if pos_weight is not None:
        if not isinstance(pos_weight, jt.Var):
            pos_weight = jt.array(pos_weight)
        pos_weight = pos_weight.astype(loss.dtype)
        loss = loss * (pos_weight * target + (1 - target))

    if weight is not None:
        if not isinstance(weight, jt.Var):
            weight = jt.array(weight)
        loss = loss * weight.astype(loss.dtype)

    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


def cross_entropy(input,
                  target,
                  weight=None,
                  ignore_index=-100,
                  reduction='mean',
                  **kwargs):
    del kwargs
    return nn.cross_entropy_loss(
        input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def mse_loss(input, target, reduction='mean'):
    loss = (input - target)**2
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


def kl_div(input, target, reduction='mean', log_target=False):
    eps = 1e-12
    if log_target:
        target_prob = jt.exp(target)
        loss = target_prob * (target - input)
    else:
        target_prob = jt.clamp(target, min_v=eps)
        loss = target_prob * (jt.log(target_prob) - input)

    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'batchmean':
        return loss.sum() / max(1, int(input.shape[0]))
    return loss.mean()


def one_hot(input, num_classes=-1):
    return nn.one_hot(input, num_classes=num_classes)

