"""Optimizer compatibility exports."""

from .optimizer import OptimWrapper
from .scheduler import ConstantLR, LinearLR, PolyLR


class DefaultOptimWrapperConstructor:

    def __init__(self, optimizer_cfg=None, paramwise_cfg=None, **kwargs):
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = paramwise_cfg
        self.kwargs = kwargs

    def __call__(self, model):
        del model
        return dict(
            type=OptimWrapper,
            optimizer=self.optimizer_cfg,
            paramwise_cfg=self.paramwise_cfg,
            **self.kwargs)


__all__ = [
    'OptimWrapper', 'PolyLR', 'ConstantLR', 'LinearLR',
    'DefaultOptimWrapperConstructor'
]
