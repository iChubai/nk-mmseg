"""Transforms API compatibility."""

from __future__ import annotations

import jittor as jt
import numpy as np

from .base import BaseTransform
from .loading import LoadAnnotations, LoadImageFromFile
from .processing import RandomFlip, RandomResize, Resize, TestTimeAug


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def to_tensor(data):
    if isinstance(data, jt.Var):
        return data
    if isinstance(data, np.ndarray):
        return jt.array(data)
    return jt.array(np.array(data))


__all__ = [
    'BaseTransform',
    'Compose',
    'to_tensor',
    'LoadImageFromFile',
    'LoadAnnotations',
    'RandomFlip',
    'RandomResize',
    'Resize',
    'TestTimeAug',
]
