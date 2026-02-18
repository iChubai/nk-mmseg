#!/usr/bin/env python3
"""Unified evaluation entry for nk-mmseg."""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jittor as jt


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


def _run_legacy_eval(args, passthrough: list[str]):
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'utils' / 'eval.py'),
        f'--config={args.config}',
    ]
    if args.gpus is not None:
        cmd.append(f'--gpus={int(args.gpus)}')
    if args.checkpoint:
        cmd.append(f'--continue_fpath={args.checkpoint}')
    if args.strict_load:
        cmd.append('--strict-load')
    if args.multi_scale:
        cmd.append('--multi_scale')
    if args.flip:
        cmd.append('--flip')
    if args.sliding:
        cmd.append('--sliding')
    if args.scales:
        cmd.append('--scales')
        cmd.extend([str(float(x)) for x in args.scales])
    if args.max_iters and int(args.max_iters) > 0:
        cmd.append(f'--max-iters={int(args.max_iters)}')
    if args.save_pred:
        cmd.append('--save-pred')
        cmd.append(f'--pred-dir={args.pred_dir}')
    if args.verbose:
        cmd.append('--verbose')
    cmd.extend(passthrough)

    env = os.environ.copy()
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        if str(REPO_ROOT) not in current_pythonpath.split(':'):
            env['PYTHONPATH'] = f'{str(REPO_ROOT)}:{current_pythonpath}'
    else:
        env['PYTHONPATH'] = str(REPO_ROOT)

    print('[tools/test] Legacy dispatch:')
    print('  ' + ' '.join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _build_eval_compat_cfg(cfg):
    model_cfg = cfg.get('model', {})
    num_classes = int(cfg.get('num_classes', model_cfg.get('num_classes', 40)))
    dataset_name = str(cfg.get('dataset_name', 'NYUDepthv2'))
    image_height = int(cfg.get('image_height', 480))
    image_width = int(cfg.get('image_width', 640))

    eval_crop_size = cfg.get('eval_crop_size', (image_height, image_width))
    if isinstance(eval_crop_size, list):
        eval_crop_size = tuple(eval_crop_size)
    if isinstance(eval_crop_size, int):
        eval_crop_size = (eval_crop_size, eval_crop_size)

    eval_scale_array = cfg.get('eval_scale_array', [1.0])
    if not isinstance(eval_scale_array, (list, tuple)):
        eval_scale_array = [float(eval_scale_array)]

    return SimpleNamespace(
        dataset_name=dataset_name,
        num_classes=num_classes,
        background=int(cfg.get('background', 255)),
        eval_crop_size=eval_crop_size,
        eval_stride_rate=float(cfg.get('eval_stride_rate', 2 / 3)),
        eval_scale_array=list(eval_scale_array),
        eval_flip=bool(cfg.get('eval_flip', True)),
        norm_mean=cfg.get('norm_mean', [0.485, 0.456, 0.406]),
        norm_std=cfg.get('norm_std', [0.229, 0.224, 0.225]),
    )


def _print_legacy_metric_results(results):
    if not isinstance(results, dict):
        print(results)
        return
    print('Evaluation Results:')
    for key in ('mIoU', 'mAcc', 'Overall_Acc', 'FWIoU'):
        if key in results:
            print(f'  {key}: {float(results[key]):.4f}')


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


def _load_mmengine_checkpoint(args, runner, cfg, force_default=False):
    ckpt = cfg.get('load_from')
    if not ckpt:
        return False

    if _is_dformer_model_cfg(cfg):
        from utils.jt_utils import load_model
        print('[tools/test] DFormer checkpoint loading via utils.jt_utils.load_model')
        print(f'  checkpoint: {ckpt}')
        load_model(
            model=runner.model,
            model_file=ckpt,
            strict=bool(args.strict_load))
        runner.load_from = None
        cfg['load_from'] = None
        return True

    if force_default:
        runner.load_checkpoint(ckpt, resume=False)
        runner.load_from = None
        cfg['load_from'] = None
        return True

    return False


def _run_mmengine_advanced_eval(args, runner, cfg, mode='val'):
    from utils.val_mm import evaluate, evaluate_msf

    eval_cfg = _build_eval_compat_cfg(cfg)
    save_dir = args.pred_dir if args.save_pred else None
    max_iters = int(args.max_iters) if args.max_iters else 0
    use_aug = bool(args.multi_scale or args.flip or args.sliding)

    if mode == 'test':
        runner._ensure_test_components()
        dataloader = runner.test_dataloader
    else:
        runner._ensure_val_components()
        dataloader = runner.val_dataloader

    if use_aug:
        scales = [float(x) for x in args.scales] if args.scales else list(eval_cfg.eval_scale_array)
        metric = evaluate_msf(
            model=runner.model,
            data_loader=dataloader,
            config=eval_cfg,
            scales=scales,
            flip=bool(args.flip),
            sliding=bool(args.sliding),
            save_dir=save_dir,
            max_iters=max_iters)
        results = metric.get_results() if hasattr(metric, 'get_results') else metric
    else:
        results = evaluate(
            runner.model,
            dataloader,
            config=eval_cfg,
            verbose=bool(args.verbose),
            save_dir=save_dir,
            max_iters=max_iters)
    _print_legacy_metric_results(results)


def _run_mmengine_eval(args, cfg_path: Path):
    _ensure_repo_on_path()

    from mmengine.config import Config
    from mmengine.runner import Runner
    import mmseg.models  # noqa: F401
    import mmseg.datasets  # noqa: F401
    import mmseg.evaluation  # noqa: F401

    cfg = Config.fromfile(str(cfg_path))
    cfg_updates = _parse_cfg_options(args.cfg_options)
    if cfg_updates:
        _deep_update(cfg, cfg_updates)

    if args.work_dir:
        cfg['work_dir'] = args.work_dir
    if args.checkpoint:
        cfg['load_from'] = args.checkpoint

    runner = Runner.from_cfg(cfg)
    preloaded = _load_mmengine_checkpoint(args, runner, cfg, force_default=False)
    advanced_eval = (
        bool(args.multi_scale)
        or bool(args.flip)
        or bool(args.sliding)
        or bool(args.save_pred)
        or int(args.max_iters or 0) > 0
    )

    mode = args.mode.lower()
    if advanced_eval:
        if not preloaded:
            _load_mmengine_checkpoint(args, runner, cfg, force_default=True)
        _run_mmengine_advanced_eval(args, runner, cfg, mode=mode)
        return

    if mode == 'test':
        has_test = cfg.get('test_dataloader') is not None and cfg.get('test_evaluator') is not None
        if has_test:
            runner.test()
        else:
            runner.val()
    else:
        runner.val()


def main():
    parser = argparse.ArgumentParser(description='Unified eval entry for nk-mmseg')
    parser.add_argument('--config', required=True, help='config path or module path')
    parser.add_argument('--gpus', type=int, default=1, help='legacy eval gpu count')
    parser.add_argument('--checkpoint', default=None, help='checkpoint path')
    parser.add_argument('--work-dir', default=None, help='work directory override')
    parser.add_argument(
        '--mode',
        default='val',
        choices=['val', 'test'],
        help='mmengine eval mode; legacy path always uses utils/eval.py')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=None,
        help='override cfg by key=value pairs, supports dotted keys')
    parser.add_argument(
        '--strict-load',
        action='store_true',
        help='strict checkpoint loading (legacy and DFormer mmengine path)')
    parser.add_argument(
        '--multi_scale',
        action='store_true',
        help='multi-scale evaluation (legacy and mmengine advanced path)')
    parser.add_argument(
        '--flip',
        action='store_true',
        help='enable horizontal flip during evaluation')
    parser.add_argument(
        '--sliding',
        action='store_true',
        help='enable sliding-window inference during evaluation')
    parser.add_argument(
        '--scales',
        nargs='+',
        type=float,
        default=None,
        help='evaluation scales, e.g. --scales 0.5 0.75 1.0 1.25 1.5')
    parser.add_argument(
        '--max-iters',
        type=int,
        default=0,
        help='maximum evaluation iterations (0 means full set)')
    parser.add_argument(
        '--save-pred',
        action='store_true',
        help='save predictions')
    parser.add_argument(
        '--pred-dir',
        default='./predictions',
        help='prediction output directory when --save-pred is set')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='verbose logging for advanced evaluation path')

    args, passthrough = parser.parse_known_args()

    # Keep eval path consistent with legacy utils/eval.py and force GPU runtime.
    # Without this, Jittor may stay on CPU and make paper-mode eval appear stuck.
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 1
    jt.flags.use_stat_allocator = 1
    print(f"[tools/test] Jittor runtime: use_cuda={int(jt.flags.use_cuda)}")
    print(f"[tools/test] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    cfg_path = _resolve_config_path(args.config)

    if _is_legacy_config(args.config, cfg_path):
        _run_legacy_eval(args, passthrough)
        return 0

    if passthrough:
        parser.error(
            'Unknown args for mmengine config mode: '
            + ' '.join(passthrough))

    if cfg_path is None:
        raise FileNotFoundError(f'config not found: {args.config}')
    _run_mmengine_eval(args, cfg_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
