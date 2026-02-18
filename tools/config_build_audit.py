#!/usr/bin/env python3
"""Audit config parsing + model building coverage for nk-mmseg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when invoked via file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

import mmseg.models  # noqa: F401
from mmseg.registry import MODELS


DEFAULT_EXPECTED_FAIL_PATTERNS = [
    'mmpretrain.',
    "text_encoder required either 'dataset_name' or 'vocabulary'",
    'DepthEstimator.__init__() missing 1 required positional argument: \'decode_head\'',
    'optional.txt',
]


def iter_config_files(config_root: Path):
    for path in sorted(config_root.rglob('*.py')):
        if path.name.startswith('__'):
            continue
        yield path


def main():
    parser = argparse.ArgumentParser(description='Build-audit mmseg configs')
    parser.add_argument('--config-root', default='configs')
    parser.add_argument('--max-checks', type=int, default=0,
                        help='0 means no limit')
    parser.add_argument('--show-failures', type=int, default=50)
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

    for cfg_path in iter_config_files(config_root):
        try:
            cfg = Config.fromfile(str(cfg_path))
            model_cfg = cfg.get('model', None)
            if model_cfg is None:
                continue
            MODELS.build(model_cfg)
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

    print('CONFIG BUILD AUDIT')
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
