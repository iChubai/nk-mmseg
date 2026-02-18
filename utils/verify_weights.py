#!/usr/bin/env python3
"""Verify checkpoint loading in pure Jittor."""

import os
import sys

import jittor as jt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from importlib import import_module
    from models import build_model
    from utils.jt_utils import load_model

    cfg = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), 'C')
    model = build_model(cfg)
    before = model.state_dict()

    ckpt_path = 'checkpoints/trained/NYUv2_DFormer_Large.pth'
    model = load_model(model, ckpt_path)
    after = model.state_dict()

    changed = 0
    for k in after.keys():
        if k not in before:
            continue
        a = after[k].numpy()
        b = before[k].numpy()
        if np.max(np.abs(a - b)) > 0:
            changed += 1
    print(f'parameters changed after loading: {changed}')


if __name__ == '__main__':
    main()

