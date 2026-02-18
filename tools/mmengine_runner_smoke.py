#!/usr/bin/env python3
"""Smoke test for lightweight mmengine Runner integration."""

from __future__ import annotations

import os
import shutil
import sys

import jittor as jt
import jittor.nn as nn
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from mmengine.dataset.sampler import DefaultSampler
from mmengine.evaluator import BaseMetric
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.runner import IterBasedTrainLoop, Runner, ValLoop


class ToyDataset:

    def __init__(self, size=8):
        self.size = int(size)
        self.metainfo = {'classes': ['v']}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = jt.array([[float(idx)]], dtype=jt.float32)
        y = x * 2.0 + 1.0
        return {'x': x, 'y': y}


class ToyMetric(BaseMetric):

    def process(self, data_batch: dict, data_samples):
        pred = data_samples[0]['pred']
        target = data_batch['y']
        err = jt.abs(pred - target).mean().item()
        self.results.append(float(err))

    def compute_metrics(self, results: list):
        if not results:
            return {'mae': 0.0}
        return {'mae': float(np.mean(results))}


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1)

    def execute(self, x, y=None, mode='tensor'):
        pred = self.proj(x)
        if mode == 'loss':
            if y is None:
                raise RuntimeError('target is required in loss mode')
            loss = ((pred - y) ** 2).mean()
            return {'loss': loss, 'mse': loss}
        if mode == 'predict':
            return {'pred': pred}
        return pred


def main():
    work_dir = '/tmp/nk_mmengine_runner_smoke'
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    cfg = dict(
        model=ToyModel(),
        work_dir=work_dir,
        train_dataloader=dict(
            batch_size=2,
            sampler=dict(type='DefaultSampler', shuffle=True, seed=42),
            dataset=ToyDataset(size=8),
        ),
        val_dataloader=dict(
            batch_size=1,
            sampler=dict(type=DefaultSampler, shuffle=False),
            dataset=ToyDataset(size=4),
        ),
        train_cfg=dict(type=IterBasedTrainLoop, max_iters=5, val_interval=5),
        val_cfg=dict(type=ValLoop),
        optim_wrapper=dict(
            type=OptimWrapper,
            optimizer=dict(type='SGD', lr=0.01, momentum=0.0),
        ),
        param_scheduler=[
            dict(
                type='PolyLR',
                eta_min=1e-5,
                power=0.9,
                begin=0,
                end=5,
                by_epoch=False)
        ],
        val_evaluator=ToyMetric(),
        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=1),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(
                type='CheckpointHook', by_epoch=False, interval=5),
            sampler_seed=dict(type='DistSamplerSeedHook'),
        ),
    )

    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.val()

    ckpt = os.path.join(work_dir, 'iter_5.pth')
    if not os.path.exists(ckpt):
        raise RuntimeError(f'checkpoint not found: {ckpt}')

    print('MMENGINE_RUNNER_SMOKE_PASSED')


if __name__ == '__main__':
    main()
