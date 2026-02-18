#!/usr/bin/env python3
"""Smoke test for Jittor-only weight loading."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from importlib import import_module
    from models import build_model
    from utils.jt_utils import load_model

    cfg = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'), 'C')
    model = build_model(cfg)
    ckpt = 'checkpoints/trained/DFormerv2_Large_NYU.pth'
    model = load_model(model, ckpt)
    print('weight loading smoke test passed')


if __name__ == '__main__':
    main()

