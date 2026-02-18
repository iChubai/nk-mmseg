# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation exports with optional dependency guards."""

import importlib
import warnings

__all__ = []


def _safe_import(symbol):
    try:
        mod = importlib.import_module('.metrics', __name__)
    except Exception as exc:
        warnings.warn(
            f'Skipped optional mmseg.evaluation.metrics: {exc}',
            RuntimeWarning,
        )
        return

    if hasattr(mod, symbol):
        globals()[symbol] = getattr(mod, symbol)
        __all__.append(symbol)


for _metric in ('CityscapesMetric', 'DepthMetric', 'IoUMetric', 'RGBXIoUMetric'):
    _safe_import(_metric)
