"""Lightweight optimizer wrapper compatible with mmengine style."""

from __future__ import annotations


class OptimWrapper:

    def __init__(self,
                 optimizer=None,
                 clip_grad=None,
                 accumulative_counts=1,
                 **kwargs):
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.accumulative_counts = int(accumulative_counts)
        self.kwargs = kwargs

    @property
    def lr(self):
        if self.optimizer is None:
            return 0.0
        return float(getattr(self.optimizer, 'lr', 0.0))

    def zero_grad(self):
        # Jittor optimizers do not require explicit zero_grad in common flow.
        return None

    def backward(self, loss):
        return loss

    def step(self, loss=None):
        if self.optimizer is None:
            raise RuntimeError('OptimWrapper.optimizer is not set')
        if loss is None:
            return self.optimizer.step()
        return self.optimizer.step(loss)

    def update_params(self, loss):
        return self.step(loss=loss)

    def state_dict(self):
        if self.optimizer is None:
            return {}
        if hasattr(self.optimizer, 'state_dict'):
            return self.optimizer.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self.optimizer is None:
            return None
        if hasattr(self.optimizer, 'load_state_dict'):
            return self.optimizer.load_state_dict(state_dict)
        return None
