#!/usr/bin/env python3
"""Layer-by-layer stats in pure Jittor."""

import os
import sys

import jittor as jt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from importlib import import_module
    from models import build_model
    from utils.jt_utils import load_model

    cfg = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), 'C')
    model = build_model(cfg)
    model = load_model(model, 'checkpoints/trained/NYUv2_DFormer_Large.pth')
    model.eval()

    rgb = jt.randn(1, 3, 480, 640)
    depth = jt.randn(1, 1, 480, 640)
    with jt.no_grad():
        out, _ = model(rgb, depth, None)
        out = out[0] if isinstance(out, (list, tuple)) else out
    print(f'output shape: {out.shape}')
    print(f'output mean: {float(out.mean())}')
    print(f'output std: {float(out.std())}')


if __name__ == '__main__':
    main()

