#!/usr/bin/env python3
"""Debug checkpoint/model keys in pure Jittor workflow."""

import os
import sys

import jittor as jt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from importlib import import_module
    from models import build_model

    config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'),
                     'C')
    model = build_model(config)
    model_keys = set(model.state_dict().keys())

    ckpt = jt.load('checkpoints/trained/NYUv2_DFormer_Large.pth')
    ckpt = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    ckpt_keys = set(ckpt.keys())

    print(f'model keys: {len(model_keys)}')
    print(f'checkpoint keys: {len(ckpt_keys)}')
    print(f'missing in ckpt: {len(model_keys - ckpt_keys)}')
    print(f'unexpected in ckpt: {len(ckpt_keys - model_keys)}')


if __name__ == '__main__':
    main()

