"""Sampler implementations compatible with mmengine configs."""

from __future__ import annotations

import random


class DefaultSampler:

    def __init__(self,
                 dataset=None,
                 shuffle=False,
                 seed=0,
                 round_up=False,
                 **kwargs):
        del kwargs
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.round_up = bool(round_up)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else 0

    def __iter__(self):
        n = len(self)
        indices = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        return iter(indices)


class InfiniteSampler(DefaultSampler):

    def __iter__(self):
        n = len(self)
        if n <= 0:
            return iter(())

        while True:
            indices = list(range(n))
            if self.shuffle:
                rng = random.Random(self.seed + self.epoch)
                rng.shuffle(indices)
                self.epoch += 1
            for idx in indices:
                yield idx
