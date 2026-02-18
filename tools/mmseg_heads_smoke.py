#!/usr/bin/env python3
"""Smoke test for decode heads requiring mmcv/mmengine compatibility layers."""

from __future__ import annotations

import sys
from pathlib import Path

import jittor as jt

# Ensure repo root import for execution via file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmseg.models  # noqa: F401
from mmseg.registry import MODELS


def main():
    x = jt.randn((1, 64, 32, 32))

    psa = MODELS.build(
        dict(
            type='PSAHead',
            in_channels=64,
            in_index=0,
            channels=32,
            num_classes=19,
            dropout_ratio=0.1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            mask_size=(16, 16),
            psa_type='collect',
            compact=True,
            shrink_factor=2))
    psa_out = psa([x])
    assert tuple(psa_out.shape) == (1, 19, 32, 32), tuple(psa_out.shape)

    cc = MODELS.build(
        dict(
            type='CCHead',
            in_channels=64,
            in_index=0,
            channels=32,
            num_classes=19,
            dropout_ratio=0.1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False))
    cc_out = cc([x])
    assert tuple(cc_out.shape) == (1, 19, 32, 32), tuple(cc_out.shape)

    point = MODELS.build(
        dict(
            type='PointHead',
            in_channels=[32],
            in_index=[0],
            channels=32,
            num_classes=19,
            dropout_ratio=0.0,
            align_corners=False))
    fine = jt.randn((1, 32, 128))
    coarse = jt.randn((1, 19, 128))
    point_out = point(fine, coarse)
    assert tuple(point_out.shape) == (1, 19, 128), tuple(point_out.shape)

    print('MMSEG_HEADS_SMOKE_PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
