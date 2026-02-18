#!/usr/bin/env python3
"""Audit config runtime tensor-forward coverage for nk-mmseg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import jittor as jt

# Ensure repo root is importable when invoked via file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

import mmseg.models  # noqa: F401
from mmseg.registry import MODELS


DEFAULT_EXPECTED_FAIL_PATTERNS = [
    'optional dependencies',
    'requirements/optional.txt',
    "required either 'dataset_name' or 'vocabulary'",
    'decode_head is required for train/inference',
]


def iter_config_files(config_root: Path, include_base: bool) -> Iterable[Path]:
    for path in sorted(config_root.rglob('*.py')):
        if path.name.startswith('__'):
            continue
        if not include_base and '/_base_/' in f'/{path.as_posix()}/':
            continue
        yield path


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_hw(value):
    if isinstance(value, int):
        if value > 0:
            return int(value), int(value)
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        h, w = int(value[0]), int(value[1])
        if h > 0 and w > 0:
            return h, w
    return None


def _guess_input_channels(cfg, model_cfg) -> int:
    for k in ('backbone', 'image_encoder', 'neck'):
        part = _cfg_get(model_cfg, k, None)
        in_channels = _cfg_get(part, 'in_channels', None)
        if isinstance(in_channels, int) and in_channels > 0:
            return int(in_channels)

    data_preprocessor = _cfg_get(model_cfg, 'data_preprocessor', None)
    if data_preprocessor is None:
        data_preprocessor = _cfg_get(cfg, 'data_preprocessor', None)
    mean = _cfg_get(data_preprocessor, 'mean', None)
    if isinstance(mean, (list, tuple)) and len(mean) > 0:
        return int(len(mean))
    return 3


def _candidate_shapes(cfg, model_cfg, base_h: int, base_w: int, max_side: int):
    shapes = []

    def add(shape):
        if shape is None:
            return
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            return
        if max(h, w) > int(max_side):
            return
        item = (h, w)
        if item not in shapes:
            shapes.append(item)

    add((base_h, base_w))

    crop_size = _as_hw(_cfg_get(cfg, 'crop_size', None))
    add(crop_size)

    data_preprocessor = _cfg_get(model_cfg, 'data_preprocessor', None)
    if data_preprocessor is None:
        data_preprocessor = _cfg_get(cfg, 'data_preprocessor', None)
    add(_as_hw(_cfg_get(data_preprocessor, 'size', None)))

    for key in ('backbone', 'image_encoder'):
        part = _cfg_get(model_cfg, key, None)
        if part is None:
            continue
        add(_as_hw(_cfg_get(part, 'img_size', None)))
        add(_as_hw(_cfg_get(part, 'pretrain_img_size', None)))

    return shapes


def _is_retryable_small_input_error(msg: str) -> bool:
    patterns = (
        'integer division or modulo by zero',
        'size of var should be larger than kernel_size',
        'Shape not match',
        'Wrong inputs arguments',
    )
    return any(p in msg for p in patterns)


def _shape_desc(out):
    if isinstance(out, (list, tuple)):
        return [tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__ for v in out]
    if hasattr(out, 'shape'):
        return tuple(out.shape)
    return type(out).__name__


def main():
    parser = argparse.ArgumentParser(
        description='Tensor-forward audit for mmseg configs')
    parser.add_argument('--config-root', default='configs')
    parser.add_argument('--max-checks', type=int, default=0,
                        help='0 means no limit')
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--show-failures', type=int, default=50)
    parser.add_argument('--include-base', action='store_true')
    parser.add_argument(
        '--expected-fail-pattern',
        action='append',
        default=[],
        help='substring of expected failure message/path; repeatable')
    args = parser.parse_args()

    config_root = Path(args.config_root)
    if not config_root.exists():
        raise SystemExit(f'config root does not exist: {config_root}')

    expected_patterns = list(DEFAULT_EXPECTED_FAIL_PATTERNS)
    expected_patterns.extend(args.expected_fail_pattern)

    checked = 0
    ok = 0
    expected_fail = []
    unexpected_fail = []

    for cfg_path in iter_config_files(config_root, include_base=args.include_base):
        try:
            cfg = Config.fromfile(str(cfg_path))
            model_cfg = _cfg_get(cfg, 'model', None)
            if model_cfg is None:
                continue

            model = MODELS.build(model_cfg)
            in_ch = _guess_input_channels(cfg, model_cfg)
            shapes = _candidate_shapes(
                cfg, model_cfg, args.height, args.width, args.max_side)
            if not shapes:
                shapes = [(args.height, args.width)]

            out = None
            last_exc = None
            queued = list(shapes)
            seen = set(queued)
            while queued:
                h, w = queued.pop(0)
                x = jt.randn((1, in_ch, h, w))
                try:
                    try:
                        out = model(x, data_samples=None, mode='tensor')
                    except TypeError:
                        out = model(x)
                    _shape_desc(out)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    msg = f'{type(exc).__name__}: {exc}'
                    if _is_retryable_small_input_error(msg):
                        nh = min(int(args.max_side), max(h * 2, h + 64))
                        nw = min(int(args.max_side), max(w * 2, w + 64))
                        nxt = (nh, nw)
                        if nxt not in seen and nh > h and nw > w:
                            seen.add(nxt)
                            queued.append(nxt)
                    continue

            if last_exc is not None:
                raise last_exc

            ok += 1
            checked += 1
        except Exception as exc:
            checked += 1
            msg = f'{type(exc).__name__}: {exc}'
            item = (str(cfg_path), msg.split('\n')[0])
            if any(p in item[0] or p in item[1] for p in expected_patterns):
                expected_fail.append(item)
            else:
                unexpected_fail.append(item)

        if args.max_checks > 0 and checked >= args.max_checks:
            break

    print('CONFIG TENSOR AUDIT')
    print(f'checked={checked} ok={ok} '
          f'expected_fail={len(expected_fail)} unexpected_fail={len(unexpected_fail)}')

    if expected_fail:
        print('\nExpected failures:')
        for path, msg in expected_fail[:args.show_failures]:
            print(f'  - {path} :: {msg}')

    if unexpected_fail:
        print('\nUnexpected failures:')
        for path, msg in unexpected_fail[:args.show_failures]:
            print(f'  - {path} :: {msg}')
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == '__main__':
    main()
