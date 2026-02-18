#!/usr/bin/env python3
"""Migration audit utility for nk-mmseg.

This tool verifies that key model configs can:
1) Build successfully.
2) Load target checkpoints in pure Jittor mode.
3) Run forward/eval smoke checks.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from importlib import import_module

import jittor as jt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import build_model
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import get_val_loader
from utils.engine.engine import Engine
from utils.jt_utils import check_runtime_compatibility, load_model
from utils.metric import SegmentationMetric


@dataclass(frozen=True)
class AuditCase:
    name: str
    config: str
    checkpoint: str


DEFAULT_CASES = [
    AuditCase(
        name='dformer_l_nyu',
        config='local_configs.NYUDepthv2.DFormer_Large',
        checkpoint='checkpoints/trained/NYUv2_DFormer_Large.pth',
    ),
    AuditCase(
        name='dformerv2_l_nyu',
        config='local_configs.NYUDepthv2.DFormerv2_L',
        checkpoint='checkpoints/trained/DFormerv2_Large_NYU.pth',
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description='nk-mmseg migration audit')
    parser.add_argument(
        '--cases',
        nargs='+',
        default=[c.name for c in DEFAULT_CASES],
        help='subset of cases to run',
    )
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=0,
        help='number of validation samples for quick metric check (0=skip)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='validation batch size for quick metric check',
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='run in CPU mode (default: CUDA)',
    )
    return parser.parse_args()


def _choose_cases(names):
    case_map = {c.name: c for c in DEFAULT_CASES}
    selected = []
    for name in names:
        if name not in case_map:
            raise ValueError(
                f'Unknown case "{name}". Available: {", ".join(case_map.keys())}')
        selected.append(case_map[name])
    return selected


def _resolve_hw(cfg):
    if hasattr(cfg, 'eval_crop_size'):
        h, w = cfg.eval_crop_size
        return int(h), int(w)
    h = int(getattr(cfg, 'image_height', 480))
    w = int(getattr(cfg, 'image_width', 640))
    return h, w


def _normalize_output(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 2:
        outputs = outputs[0]
    if isinstance(outputs, list):
        outputs = outputs[0]
    return outputs


def _forward_smoke(model, cfg):
    h, w = _resolve_hw(cfg)
    rgb = jt.randn((1, 3, h, w))
    modal = jt.randn((1, 1, h, w))
    with jt.no_grad():
        pred = _normalize_output(model(rgb, modal))
    return tuple(pred.shape)


def _quick_eval(model, cfg, eval_samples, batch_size):
    engine = Engine()
    val_loader, _ = get_val_loader(
        engine=engine,
        dataset_cls=RGBXDataset,
        config=cfg,
        val_batch_size=batch_size,
    )
    metric = SegmentationMetric(val_loader.dataset.num_classes)

    with jt.no_grad():
        for i, minibatch in enumerate(val_loader):
            if i >= eval_samples:
                break
            outputs = _normalize_output(model(minibatch['data'], minibatch['modal_x']))
            preds = jt.argmax(outputs, dim=1)
            if isinstance(preds, tuple):
                preds = preds[0]
            labels = minibatch['label']
            if isinstance(labels, tuple):
                labels = labels[0]
            metric.update(preds.numpy(), labels.numpy())

    return metric.get_results()


def run_case(case, eval_samples, batch_size):
    print(f'\n=== {case.name} ===')
    cfg = getattr(import_module(case.config), 'C')
    ckpt_path = case.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    t0 = time.time()
    model = build_model(cfg)
    t_build = time.time()
    model, load_summary = load_model(
        model, ckpt_path, strict=True, return_stats=True)
    t_load = time.time()
    model.eval()

    out_shape = _forward_smoke(model, cfg)
    t_smoke = time.time()

    print(f'build_time={t_build - t0:.2f}s')
    print(f'load_time={t_load - t_build:.2f}s')
    ls = load_summary.get('load_stats', {}) if isinstance(load_summary, dict) else {}
    if ls:
        print(
            f'loaded_params={ls.get("loaded_count", 0)} '
            f'skipped={ls.get("skipped_count", 0)} '
            f'shape_mismatch={ls.get("shape_mismatch_count", 0)}')
    print(f'smoke_time={t_smoke - t_load:.2f}s')
    print(f'forward_shape={out_shape}')

    if eval_samples > 0:
        t_eval0 = time.time()
        results = _quick_eval(model, cfg, eval_samples, batch_size)
        t_eval1 = time.time()
        print(
            f'quick_eval samples={eval_samples} '
            f'mIoU={results["mIoU"]:.4f} mAcc={results["mAcc"]:.4f} '
            f'Overall={results["Overall_Acc"]:.4f} FWIoU={results["FWIoU"]:.4f} '
            f'time={t_eval1 - t_eval0:.2f}s')


def main():
    args = parse_args()
    check_runtime_compatibility(raise_on_error=True)

    jt.flags.use_cuda = 0 if args.cpu else 1
    selected = _choose_cases(args.cases)

    print(f'Jittor={jt.__version__}')
    print(f'NumPy mode validated, use_cuda={jt.flags.use_cuda}')
    print(f'cases={", ".join(c.name for c in selected)}')

    failures = []
    for case in selected:
        try:
            run_case(case, args.eval_samples, args.batch_size)
        except Exception as e:
            failures.append((case.name, str(e)))
            print(f'[FAILED] {case.name}: {e}')

    if failures:
        print('\nAudit failed:')
        for name, msg in failures:
            print(f'  - {name}: {msg}')
        return 1

    print('\nAudit passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
