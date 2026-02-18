"""Dropout building utilities compatible with MMCV configs."""

from __future__ import annotations

import jittor as jt
import jittor.nn as nn


class DropPath(nn.Module):
    """Stochastic depth (per-sample path drop)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def execute(self, x):
        if self.drop_prob <= 0.0 or not self.is_training():
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape)
        random_tensor = jt.floor(random_tensor)
        return x / keep_prob * random_tensor


def build_dropout(cfg):
    if cfg is None:
        return nn.Identity()
    if not isinstance(cfg, dict):
        raise TypeError(f'dropout cfg should be dict, got {type(cfg)}')

    cfg = dict(cfg)
    drop_type = str(cfg.pop('type', 'Dropout'))
    p = float(cfg.pop('drop_prob', cfg.pop('p', 0.0)))

    if drop_type in ('DropPath',):
        return DropPath(drop_prob=p)
    if drop_type in ('Identity',):
        return nn.Identity()
    return nn.Dropout(p=p)
