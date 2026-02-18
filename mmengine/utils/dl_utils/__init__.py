"""DL utility compatibility stubs."""

from __future__ import annotations

import jittor as jt


def collect_env():
    return {
        'jittor': jt.__version__,
    }


def mmcv_full_available() -> bool:
    return False


__all__ = ['collect_env', 'mmcv_full_available']
