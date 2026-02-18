"""Weight initialization helpers with mmengine-like signatures."""

from __future__ import annotations

import math
from typing import Tuple

import jittor as jt
import jittor.nn as nn


def _resolve_weight_bias(module) -> Tuple[jt.Var | None, jt.Var | None]:
    if isinstance(module, jt.Var):
        return module, None
    return getattr(module, 'weight', None), getattr(module, 'bias', None)


def _init_bias(bias_var: jt.Var | None, bias: float) -> None:
    if bias_var is not None:
        nn.init.constant_(bias_var, float(bias))


def normal_init(module, mean=0, std=1, bias=0):
    weight, bias_var = _resolve_weight_bias(module)
    if weight is not None:
        nn.init.gauss_(weight, mean=float(mean), std=float(std))
    _init_bias(bias_var, bias)


def trunc_normal_init(module, mean=0, std=1, a=-2, b=2, bias=0):
    weight, bias_var = _resolve_weight_bias(module)
    if weight is not None:
        nn.init.trunc_normal_(
            weight,
            mean=float(mean),
            std=float(std),
            a=float(a),
            b=float(b))
    _init_bias(bias_var, bias)


def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
    nn.init.trunc_normal_(
        tensor,
        mean=float(mean),
        std=float(std),
        a=float(a),
        b=float(b))
    return tensor


def uniform_init(module, a=0, b=1, bias=0):
    weight, bias_var = _resolve_weight_bias(module)
    if weight is not None:
        nn.init.uniform_(weight, low=float(a), high=float(b))
    _init_bias(bias_var, bias)


def constant_init(module, val=0, bias=0):
    weight, bias_var = _resolve_weight_bias(module)
    if weight is not None:
        nn.init.constant_(weight, float(val))
    _init_bias(bias_var, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    distribution = str(distribution).lower()
    if distribution not in ('uniform', 'normal'):
        raise ValueError(
            f'Invalid distribution={distribution!r}, expected "uniform" or "normal"'
        )

    weight, bias_var = _resolve_weight_bias(module)
    if weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                weight,
                a=float(a),
                mode=mode,
                nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                weight,
                a=float(a),
                mode=mode,
                nonlinearity=nonlinearity)
    _init_bias(bias_var, bias)


def caffe2_xavier_init(module, bias=0):
    # Caffe2 XavierFill matches Kaiming-uniform with these parameters.
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        bias=bias,
        distribution='uniform')


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    distribution = str(distribution).lower()
    if distribution not in ('uniform', 'normal'):
        raise ValueError(
            f'Invalid distribution={distribution!r}, expected "uniform" or "normal"'
        )

    weight, bias_var = _resolve_weight_bias(module)
    if weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(weight, gain=float(gain))
        else:
            nn.init.xavier_gauss_(weight, gain=float(gain))
    _init_bias(bias_var, bias)


def bias_init_with_prob(prior_prob):
    prior_prob = float(prior_prob)
    if not (0.0 < prior_prob < 1.0):
        raise ValueError(f'prior_prob must be in (0, 1), got {prior_prob}')
    return float(-math.log((1 - prior_prob) / prior_prob))


__all__ = [
    'normal_init',
    'trunc_normal_init',
    'trunc_normal_',
    'uniform_init',
    'constant_init',
    'kaiming_init',
    'caffe2_xavier_init',
    'xavier_init',
    'bias_init_with_prob',
]
