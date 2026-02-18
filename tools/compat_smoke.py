#!/usr/bin/env python3
"""Compatibility smoke checks for nk-mmseg (pure Jittor)."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import jittor as jt


# Ensure repo root is importable when this script is executed by file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _iter_modules(package_name: str) -> List[str]:
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, '__path__'):
        return [package_name]
    return sorted(
        name for _, name, _ in pkgutil.walk_packages(
            pkg.__path__, prefix=f'{package_name}.'))


def import_audit(package_name: str) -> Tuple[int, int, List[Tuple[str, str, str]]]:
    modules = _iter_modules(package_name)
    failed: List[Tuple[str, str, str]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            failed.append(
                (name, type(exc).__name__, str(exc).splitlines()[0]))
    return len(modules), len(failed), failed


def backbone_forward_smoke() -> List[Tuple[str, bool, str]]:
    import mmseg.models  # noqa: F401
    from mmseg.registry import MODELS

    x1 = jt.randn((1, 3, 64, 64))
    x2 = jt.randn((1, 3, 56, 56))
    x3 = jt.randn((1, 3, 65, 67))
    cases = [
        (
            'ResNetV1c',
            dict(
                type='ResNetV1c',
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
            x1,
        ),
        (
            'MixVisionTransformer',
            dict(
                type='MixVisionTransformer',
                in_channels=3,
                embed_dims=32,
                num_stages=4,
                num_layers=[1, 1, 1, 1],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1),
            x1,
        ),
        (
            'SwinTransformer',
            dict(
                type='SwinTransformer',
                pretrain_img_size=224,
                in_channels=3,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[1, 1, 1, 1],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN', requires_grad=True)),
            x2,
        ),
        (
            'MobileNetV3',
            dict(
                type='MobileNetV3',
                arch='small',
                out_indices=(0, 1, 2, 3, 11),
                norm_cfg=dict(type='BN'),
                frozen_stages=-1,
                reduction_factor=1,
                norm_eval=False,
                with_cp=False),
            x3,
        ),
        (
            'ResNetV1c-DCN',
            dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
                dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, True, True, True)),
            x1,
        ),
    ]

    results: List[Tuple[str, bool, str]] = []
    for name, cfg, x in cases:
        try:
            model = MODELS.build(cfg)
            out = model(x)
            if isinstance(out, (list, tuple)):
                shapes = [tuple(v.shape) for v in out]
            else:
                shapes = [tuple(out.shape)]
            results.append((name, True, f'shapes={shapes}'))
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            results.append((name, False,
                            f'{type(exc).__name__}: {str(exc).splitlines()[0]}'))
    return results


def neck_forward_smoke() -> List[Tuple[str, bool, str]]:
    import mmseg.models  # noqa: F401
    from mmseg.registry import MODELS

    cases = [
        (
            'MultiLevelNeck',
            dict(
                type='MultiLevelNeck',
                in_channels=[32, 64, 160, 256],
                out_channels=64,
                scales=[0.5, 1, 2, 4],
                norm_cfg=None,
                act_cfg=None),
            [
                jt.randn((1, 32, 16, 16)),
                jt.randn((1, 64, 8, 8)),
                jt.randn((1, 160, 4, 4)),
                jt.randn((1, 256, 2, 2)),
            ],
        ),
    ]

    results: List[Tuple[str, bool, str]] = []
    for name, cfg, x in cases:
        try:
            model = MODELS.build(cfg)
            if hasattr(model, 'init_weights'):
                model.init_weights()
            out = model(x)
            if isinstance(out, (list, tuple)):
                shapes = [tuple(v.shape) for v in out]
            else:
                shapes = [tuple(out.shape)]
            results.append((name, True, f'shapes={shapes}'))
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            results.append((name, False,
                            f'{type(exc).__name__}: {str(exc).splitlines()[0]}'))
    return results


def segmentor_forward_smoke() -> List[Tuple[str, bool, str]]:
    import mmseg.models  # noqa: F401
    from mmseg.registry import MODELS

    x = jt.randn((1, 3, 64, 64))
    cases = [
        (
            'EncoderDecoder-ResNetV1c-FCN',
            dict(
                type='EncoderDecoder',
                backbone=dict(
                    type='ResNetV1c',
                    depth=18,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    dilations=(1, 1, 1, 1),
                    strides=(1, 2, 2, 2),
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=False,
                    style='pytorch',
                    contract_dilation=True),
                decode_head=dict(
                    type='FCNHead',
                    in_channels=512,
                    in_index=3,
                    channels=128,
                    num_convs=1,
                    concat_input=False,
                    dropout_ratio=0.1,
                    num_classes=19,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0)),
                train_cfg=dict(),
                test_cfg=dict(mode='whole')),
            x,
        ),
    ]

    results: List[Tuple[str, bool, str]] = []
    for name, cfg, inputs in cases:
        try:
            model = MODELS.build(cfg)
            if hasattr(model, 'init_weights'):
                model.init_weights()
            out = model(inputs, data_samples=None, mode='tensor')
            shape = tuple(out.shape)
            results.append((name, True, f'shape={shape}'))
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            results.append((name, False,
                            f'{type(exc).__name__}: {str(exc).splitlines()[0]}'))
    return results


def print_import_report(package_names: Iterable[str]) -> bool:
    ok = True
    for pkg in package_names:
        total, failed_n, failed = import_audit(pkg)
        print(f'[import] {pkg}: total={total} failed={failed_n}')
        for idx, (name, etype, msg) in enumerate(failed[:20], 1):
            print(f'  {idx:02d}. {name} :: {etype}: {msg}')
        ok = ok and failed_n == 0
    return ok


def print_backbone_report() -> bool:
    ok = True
    for name, passed, detail in backbone_forward_smoke():
        tag = 'ok' if passed else 'fail'
        print(f'[backbone] {name}: {tag} ({detail})')
        ok = ok and passed
    return ok


def print_neck_report() -> bool:
    ok = True
    for name, passed, detail in neck_forward_smoke():
        tag = 'ok' if passed else 'fail'
        print(f'[neck] {name}: {tag} ({detail})')
        ok = ok and passed
    return ok


def print_segmentor_report() -> bool:
    ok = True
    for name, passed, detail in segmentor_forward_smoke():
        tag = 'ok' if passed else 'fail'
        print(f'[segmentor] {name}: {tag} ({detail})')
        ok = ok and passed
    return ok


def main():
    parser = argparse.ArgumentParser(description='nk-mmseg compatibility smoke checks')
    parser.add_argument(
        '--import-packages',
        nargs='+',
        default=['mmcv', 'mmengine', 'mmseg', 'mmseg.models'],
        help='package names to run import audit')
    parser.add_argument(
        '--skip-backbone',
        action='store_true',
        help='skip backbone forward smoke')
    parser.add_argument(
        '--skip-neck',
        action='store_true',
        help='skip neck forward smoke')
    parser.add_argument(
        '--skip-segmentor',
        action='store_true',
        help='skip segmentor forward smoke')
    args = parser.parse_args()

    all_ok = print_import_report(args.import_packages)
    if not args.skip_backbone:
        all_ok = print_backbone_report() and all_ok
    if not args.skip_neck:
        all_ok = print_neck_report() and all_ok
    if not args.skip_segmentor:
        all_ok = print_segmentor_report() and all_ok

    if all_ok:
        print('SMOKE PASSED')
        return 0
    print('SMOKE FAILED')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
