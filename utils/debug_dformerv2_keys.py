#!/usr/bin/env python3
"""Inspect DFormerv2 key mapping in pure Jittor checkpoints."""

import os
import sys

import jittor as jt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from importlib import import_module
    from models import build_model

    config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'),
                     'C')
    model = build_model(config)
    model_keys = set(model.state_dict().keys())

    ckpt = jt.load('checkpoints/trained/DFormerv2_Large_NYU.pth')
    ckpt = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    ckpt_keys = set(ckpt.keys())

    gamma_model = sorted(k for k in model_keys if 'gamma' in k)
    gamma_ckpt = sorted(k for k in ckpt_keys if 'gamma' in k)
    print(f'model gamma keys: {len(gamma_model)}')
    print(f'ckpt gamma keys: {len(gamma_ckpt)}')
    print('model gamma samples:')
    for k in gamma_model[:10]:
        print(f'  {k}')
    print('ckpt gamma samples:')
    for k in gamma_ckpt[:10]:
        print(f'  {k}')


if __name__ == '__main__':
    main()

