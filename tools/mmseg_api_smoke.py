#!/usr/bin/env python3
"""Smoke test for mmseg API/inferencer runtime in pure Jittor mode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import jittor as jt
import numpy as np

# Ensure repo root import for execution via file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmseg.apis import MMSegInferencer, inference_model, init_model


def _write_dummy_inputs(work_dir: Path, h: int, w: int):
    work_dir.mkdir(parents=True, exist_ok=True)
    rgb = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    depth = (np.random.rand(h, w) * 255).astype(np.uint8)
    rgb_path = work_dir / 'rgb.png'
    depth_path = work_dir / 'depth.png'
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_path), depth)
    return str(rgb_path), str(depth_path)


def main():
    default_device = 'cuda:0' if jt.has_cuda else 'cpu'
    default_config = str(REPO_ROOT / 'local_configs/NYUDepthv2/DFormer_Large.py')
    parser = argparse.ArgumentParser(description='mmseg API smoke test')
    parser.add_argument(
        '--config',
        default=default_config,
        help='model config path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='optional checkpoint path')
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--work-dir', default='/tmp/nk_mmseg_api_smoke')
    args = parser.parse_args()

    rgb_path, depth_path = _write_dummy_inputs(
        Path(args.work_dir), args.height, args.width)
    sample = {'img': rgb_path, 'modal_x': depth_path}

    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)
    output = inference_model(model, sample)
    assert hasattr(output, 'pred_sem_seg'), 'inference_model output is invalid'

    inferencer = MMSegInferencer(
        model=args.config, weights=args.checkpoint, device=args.device)
    infer_result = inferencer(sample, return_datasample=False)
    preds = infer_result.get('predictions', [])
    assert preds, 'inferencer returned empty predictions'
    pred = np.array(preds[0])
    assert pred.ndim == 2, f'expected 2D mask, got shape={pred.shape}'
    assert pred.shape == (args.height, args.width), (
        f'prediction shape mismatch: got={pred.shape} '
        f'expected={(args.height, args.width)}')

    print('MMSEG_API_SMOKE_PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
