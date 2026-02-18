#!/usr/bin/env python3
"""Unified training entry for nk-mmseg.

Phase-1 behavior:
1) local_configs/*  -> dispatch to legacy utils/train.py (keeps training parity)
2) mmseg configs/*  -> run mmengine Runner.from_cfg
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _resolve_config_path(config_arg: str) -> Path | None:
    candidate = Path(config_arg)
    if candidate.is_file():
        return candidate.resolve()

    rel = (REPO_ROOT / config_arg)
    if rel.is_file():
        return rel.resolve()

    if config_arg.endswith('.py'):
        return None

    # Support module style, e.g. local_configs.NYUDepthv2.DFormer_Large
    module_path = REPO_ROOT / (config_arg.replace('.', '/') + '.py')
    if module_path.is_file():
        return module_path.resolve()

    return None


def _is_legacy_config(config_arg: str, cfg_path: Path | None) -> bool:
    if config_arg.startswith('local_configs.') or config_arg.startswith('local_configs/'):
        return True
    if cfg_path is None:
        return False
    return 'local_configs' in cfg_path.parts


def _has_cli_flag(args: list[str], flag: str) -> bool:
    return any(x == flag or x.startswith(flag + '=') for x in args)


def _set_by_dotted_key(target: dict, dotted_key: str, value):
    keys = dotted_key.split('.')
    cur = target
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _parse_cfg_options(items: list[str] | None) -> dict:
    out = {}
    for item in (items or []):
        if '=' not in item:
            raise ValueError(f'Invalid --cfg-options item: {item}')
        key, raw_val = item.split('=', 1)
        try:
            value = ast.literal_eval(raw_val)
        except Exception:
            value = raw_val
        _set_by_dotted_key(out, key, value)
    return out


def _deep_update(dst, src):
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def _is_dformer_model_cfg(cfg) -> bool:
    model_cfg = cfg.get('model', {})
    model_type = str(model_cfg.get('type', '')).lower()
    if model_type in ('dformerlegacysegmentor', 'dformerrgbxsegmentor'):
        return True
    if 'dformer' in model_type:
        return True

    backbone = model_cfg.get('backbone')
    if isinstance(backbone, str) and 'dformer' in backbone.lower():
        return True
    if isinstance(backbone, dict):
        backbone_type = str(backbone.get('type', '')).lower()
        if 'dformer' in backbone_type:
            return True
    return False


def _run_legacy_train(args, passthrough: list[str]):
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'utils' / 'train.py'),
        f'--config={args.config}',
    ]
    cmd.extend(passthrough)

    if args.work_dir and not _has_cli_flag(passthrough, '--checkpoint_dir'):
        cmd.append(f'--checkpoint_dir={args.work_dir}')
    if args.resume_from and not _has_cli_flag(passthrough, '--continue_fpath'):
        cmd.append(f'--continue_fpath={args.resume_from}')
    elif args.load_from and not _has_cli_flag(passthrough, '--continue_fpath'):
        # legacy utils/train has only continue_fpath; treat load_from as model init
        # from checkpoint for phase-1 unified CLI behavior.
        cmd.append(f'--continue_fpath={args.load_from}')

    env = os.environ.copy()
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        if str(REPO_ROOT) not in current_pythonpath.split(':'):
            env['PYTHONPATH'] = f'{str(REPO_ROOT)}:{current_pythonpath}'
    else:
        env['PYTHONPATH'] = str(REPO_ROOT)

    print('[tools/train] Legacy dispatch:')
    print('  ' + ' '.join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _run_mmengine_train(args, cfg_path: Path):
    _ensure_repo_on_path()

    from mmengine.config import Config
    from mmengine.runner import Runner
    import mmseg.models  # noqa: F401  # trigger registry import side-effects
    import mmseg.datasets  # noqa: F401
    import mmseg.evaluation  # noqa: F401

    cfg = Config.fromfile(str(cfg_path))
    cfg_updates = _parse_cfg_options(args.cfg_options)
    if cfg_updates:
        _deep_update(cfg, cfg_updates)

    if args.work_dir:
        cfg['work_dir'] = args.work_dir
    if args.load_from:
        cfg['load_from'] = args.load_from
    if args.resume_from:
        cfg['resume_from'] = args.resume_from
        cfg['resume'] = True

    runner = Runner.from_cfg(cfg)
    if args.load_from and _is_dformer_model_cfg(cfg):
        from utils.jt_utils import load_model
        print('[tools/train] DFormer checkpoint loading via utils.jt_utils.load_model')
        print(f'  checkpoint: {args.load_from}')
        load_model(
            model=runner.model,
            model_file=args.load_from,
            strict=bool(args.strict_load))
        runner.load_from = None
        cfg['load_from'] = None
    runner.train()


def main():
    parser = argparse.ArgumentParser(description='Unified train entry for nk-mmseg')
    parser.add_argument('--config', required=True, help='config path or module path')
    parser.add_argument('--work-dir', default=None, help='work directory override')
    parser.add_argument('--load-from', default=None, help='model checkpoint to load')
    parser.add_argument('--resume-from', default=None, help='checkpoint to resume')
    parser.add_argument(
        '--strict-load',
        action='store_true',
        help='strict checkpoint loading when --load-from uses DFormer model')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=None,
        help='override cfg by key=value pairs, supports dotted keys')

    args, passthrough = parser.parse_known_args()
    cfg_path = _resolve_config_path(args.config)

    if _is_legacy_config(args.config, cfg_path):
        _run_legacy_train(args, passthrough)
        return 0

    if passthrough:
        parser.error(
            'Unknown args for mmengine config mode: '
            + ' '.join(passthrough))

    if cfg_path is None:
        raise FileNotFoundError(f'config not found: {args.config}')
    _run_mmengine_train(args, cfg_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
