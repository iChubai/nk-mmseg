# Copyright (c) OpenMMLab. All rights reserved.
"""Metric exports with optional dependency guards."""

import importlib
import warnings

__all__ = []
_SKIPPED = []


def _safe_import(module_name, symbol):
    try:
        mod = importlib.import_module(f'.{module_name}', __name__)
    except Exception as exc:
        _SKIPPED.append((module_name, str(exc)))
        return

    if hasattr(mod, symbol):
        globals()[symbol] = getattr(mod, symbol)
        __all__.append(symbol)


_safe_import('citys_metric', 'CityscapesMetric')
_safe_import('depth_metric', 'DepthMetric')
_safe_import('iou_metric', 'IoUMetric')
_safe_import('rgbx_iou_metric', 'RGBXIoUMetric')

if _SKIPPED:
    skipped_names = ', '.join(name for name, _ in _SKIPPED)
    warnings.warn(
        f'Skipped optional mmseg.evaluation.metrics modules '
        f'({len(_SKIPPED)}): {skipped_names}',
        RuntimeWarning,
    )
