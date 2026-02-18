"""Lightweight LR schedulers for pure Jittor runtime."""

from __future__ import annotations

from math import inf


class _BaseLR:

    def __init__(self,
                 optimizer,
                 begin=0,
                 end=inf,
                 by_epoch=True,
                 last_step=-1,
                 **kwargs):
        del kwargs
        self.optimizer = getattr(optimizer, 'optimizer', optimizer)
        self.begin = int(begin)
        self.end = int(end) if end != inf else inf
        self.by_epoch = bool(by_epoch)
        self.last_step = int(last_step)
        self.base_lr = float(getattr(self.optimizer, 'lr', 0.0))

    def _set_lr(self, lr):
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = float(lr)

    def _get_lr(self, step):
        return self.base_lr

    def step(self):
        self.last_step += 1
        cur = self.last_step
        if cur < self.begin:
            return None
        if self.end != inf and cur >= self.end:
            lr = self._get_lr(self.end - 1 if self.end > self.begin else self.begin)
        else:
            lr = self._get_lr(cur)
        self._set_lr(lr)
        return lr

    def state_dict(self):
        return {
            'begin': self.begin,
            'end': self.end,
            'by_epoch': self.by_epoch,
            'last_step': self.last_step,
            'base_lr': self.base_lr,
        }

    def load_state_dict(self, state_dict):
        self.begin = int(state_dict.get('begin', self.begin))
        self.end = state_dict.get('end', self.end)
        self.by_epoch = bool(state_dict.get('by_epoch', self.by_epoch))
        self.last_step = int(state_dict.get('last_step', self.last_step))
        self.base_lr = float(state_dict.get('base_lr', self.base_lr))
        if self.last_step >= self.begin:
            cur = self.last_step
            if self.end != inf and cur >= self.end:
                lr = self._get_lr(self.end - 1 if self.end > self.begin else self.begin)
            else:
                lr = self._get_lr(cur)
            self._set_lr(lr)


class PolyLR(_BaseLR):

    def __init__(self, optimizer, eta_min=0.0, power=1.0, **kwargs):
        super().__init__(optimizer=optimizer, **kwargs)
        self.eta_min = float(eta_min)
        self.power = float(power)

    def _get_lr(self, step):
        if self.end == inf:
            progress = 0.0
        else:
            total = max(1, self.end - self.begin)
            progress = min(max((step - self.begin) / total, 0.0), 1.0)
        coeff = (1.0 - progress) ** self.power
        return (self.base_lr - self.eta_min) * coeff + self.eta_min

    def state_dict(self):
        out = super().state_dict()
        out.update({'eta_min': self.eta_min, 'power': self.power})
        return out

    def load_state_dict(self, state_dict):
        self.eta_min = float(state_dict.get('eta_min', self.eta_min))
        self.power = float(state_dict.get('power', self.power))
        super().load_state_dict(state_dict)


class ConstantLR(_BaseLR):

    def __init__(self, optimizer, factor=1.0, total_iters=None, **kwargs):
        super().__init__(optimizer=optimizer, **kwargs)
        self.factor = float(factor)
        if total_iters is not None and self.end == inf:
            self.end = self.begin + int(total_iters)

    def _get_lr(self, step):
        if self.end == inf:
            return self.base_lr * self.factor
        if step < self.end:
            return self.base_lr * self.factor
        return self.base_lr

    def state_dict(self):
        out = super().state_dict()
        out.update({'factor': self.factor})
        return out

    def load_state_dict(self, state_dict):
        self.factor = float(state_dict.get('factor', self.factor))
        super().load_state_dict(state_dict)


class LinearLR(_BaseLR):

    def __init__(self,
                 optimizer,
                 start_factor=1.0 / 3.0,
                 end_factor=1.0,
                 total_iters=None,
                 **kwargs):
        super().__init__(optimizer=optimizer, **kwargs)
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        if total_iters is not None and self.end == inf:
            self.end = self.begin + int(total_iters)

    def _get_lr(self, step):
        if self.end == inf:
            factor = self.end_factor
        else:
            total = max(1, self.end - self.begin)
            progress = min(max((step - self.begin) / total, 0.0), 1.0)
            factor = self.start_factor + (self.end_factor - self.start_factor) * progress
        return self.base_lr * factor

    def state_dict(self):
        out = super().state_dict()
        out.update({
            'start_factor': self.start_factor,
            'end_factor': self.end_factor
        })
        return out

    def load_state_dict(self, state_dict):
        self.start_factor = float(state_dict.get('start_factor', self.start_factor))
        self.end_factor = float(state_dict.get('end_factor', self.end_factor))
        super().load_state_dict(state_dict)
