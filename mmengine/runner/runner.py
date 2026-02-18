"""Lightweight mmengine Runner for pure Jittor runtime."""

from __future__ import annotations

import copy
import os
import os.path as osp
import random
from typing import Any

import jittor as jt
import jittor.nn as nn
import numpy as np

from mmengine.dataset import BaseDataset
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmengine.evaluator import BaseMetric
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, Hook,
                            IterTimerHook, LoggerHook, ParamSchedulerHook)
from mmengine.logging import MMLogger
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import ConstantLR, LinearLR, PolyLR

from .checkpoint import _extract_state_dict, _load_checkpoint
from .loops import BaseLoop, IterBasedTrainLoop, TestLoop, ValLoop

try:
    from mmseg.registry import DATASETS, HOOKS as MMSEG_HOOKS, METRICS, MODELS
except Exception:  # pragma: no cover
    DATASETS = None
    MMSEG_HOOKS = None
    METRICS = None
    MODELS = None


def _ensure_mmseg_registries():
    global DATASETS, MMSEG_HOOKS, METRICS, MODELS
    if all(x is not None for x in (DATASETS, MMSEG_HOOKS, METRICS, MODELS)):
        return
    try:
        from mmseg.registry import DATASETS as _DATASETS
        from mmseg.registry import HOOKS as _MMSEG_HOOKS
        from mmseg.registry import METRICS as _METRICS
        from mmseg.registry import MODELS as _MODELS
        DATASETS = _DATASETS
        MMSEG_HOOKS = _MMSEG_HOOKS
        METRICS = _METRICS
        MODELS = _MODELS
    except Exception:
        pass


_HOOK_NAME_MAP = {
    'CheckpointHook': CheckpointHook,
    'DistSamplerSeedHook': DistSamplerSeedHook,
    'IterTimerHook': IterTimerHook,
    'LoggerHook': LoggerHook,
    'ParamSchedulerHook': ParamSchedulerHook,
}

_LOOP_NAME_MAP = {
    'IterBasedTrainLoop': IterBasedTrainLoop,
    'ValLoop': ValLoop,
    'TestLoop': TestLoop,
}

_SCHEDULER_NAME_MAP = {
    'PolyLR': PolyLR,
    'ConstantLR': ConstantLR,
    'LinearLR': LinearLR,
}

_PRIORITY_MAP = {
    'HIGHEST': 0,
    'VERY_HIGH': 10,
    'HIGH': 30,
    'ABOVE_NORMAL': 40,
    'NORMAL': 50,
    'BELOW_NORMAL': 60,
    'LOW': 70,
    'VERY_LOW': 90,
    'LOWEST': 100,
}


def _priority_value(priority):
    if isinstance(priority, int):
        return priority
    if isinstance(priority, str):
        return _PRIORITY_MAP.get(priority.upper(), 50)
    return 50


def _collate_batch(batch):
    def _collate_values(values):
        if not values:
            return values
        if all(isinstance(v, jt.Var) for v in values):
            return jt.stack(values, dim=0)
        if all(isinstance(v, np.ndarray) for v in values):
            return np.stack(values, axis=0)
        return values

    if len(batch) == 1:
        if isinstance(batch[0], dict):
            out = {}
            for key, value in batch[0].items():
                out[key] = _collate_values([value])
            return out
        return batch[0]
    if isinstance(batch[0], dict):
        keys = set()
        for item in batch:
            keys.update(item.keys())
        out = {}
        for key in keys:
            out[key] = _collate_values([item.get(key) for item in batch])
        return out
    return batch


class _SimpleDataLoader:

    def __init__(self, dataset, batch_size=1, sampler=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.sampler = sampler or DefaultSampler(dataset=dataset, shuffle=False)
        self.drop_last = bool(drop_last)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield _collate_batch(batch)
                batch = []
        if batch and not self.drop_last:
            yield _collate_batch(batch)


class _MetricListEvaluator:

    def __init__(self, metrics):
        self.metrics = metrics

    def process(self, data_batch, data_samples):
        for metric in self.metrics:
            metric.process(data_batch, data_samples)

    def evaluate(self, size=None):
        merged = {}
        for metric in self.metrics:
            out = metric.evaluate(size=size)
            if isinstance(out, dict):
                merged.update(out)
        return merged

    @property
    def dataset_meta(self):
        if not self.metrics:
            return {}
        return getattr(self.metrics[0], 'dataset_meta', {})

    @dataset_meta.setter
    def dataset_meta(self, meta):
        for metric in self.metrics:
            metric.dataset_meta = meta


class Runner:

    def __init__(self,
                 model,
                 work_dir='work_dirs/mmengine',
                 train_dataloader=None,
                 val_dataloader=None,
                 test_dataloader=None,
                 train_cfg=None,
                 val_cfg=None,
                 test_cfg=None,
                 optim_wrapper=None,
                 param_scheduler=None,
                 val_evaluator=None,
                 test_evaluator=None,
                 default_hooks=None,
                 custom_hooks=None,
                 load_from=None,
                 resume=False,
                 resume_from=None,
                 seed=None,
                 cfg=None,
                 **kwargs):
        del kwargs
        self.cfg = cfg
        self.work_dir = osp.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

        self._set_seed(seed)
        self.logger = MMLogger.get_current_instance()

        self.model = self._build_model(model)
        self.load_from = load_from
        self.resume = bool(resume)
        self.resume_from = resume_from

        self.train_dataloader_cfg = train_dataloader
        self.val_dataloader_cfg = val_dataloader
        self.test_dataloader_cfg = test_dataloader
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg

        self.optim_wrapper_cfg = optim_wrapper
        self.param_scheduler_cfg = param_scheduler
        self.val_evaluator_cfg = val_evaluator
        self.test_evaluator_cfg = test_evaluator

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.optim_wrapper = None
        self.param_schedulers = []
        self.val_evaluator = None
        self.test_evaluator = None
        self.train_loop = None
        self.val_loop = None
        self.test_loop = None

        self.iter = 0
        self._log_vars = {}
        self._hooks = self._build_hooks(default_hooks, custom_hooks)

    @classmethod
    def from_cfg(cls, cfg):
        cfg = copy.deepcopy(cfg)
        return cls(
            model=cfg.get('model'),
            work_dir=cfg.get('work_dir', 'work_dirs/mmengine'),
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            optim_wrapper=cfg.get('optim_wrapper', cfg.get('optimizer')),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks', cfg.get('hooks')),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            resume_from=cfg.get('resume_from'),
            seed=cfg.get('seed'),
            cfg=cfg,
        )

    @staticmethod
    def _set_seed(seed):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        jt.set_global_seed(seed)

    def _build_model(self, model):
        _ensure_mmseg_registries()
        if isinstance(model, nn.Module):
            return model
        if isinstance(model, dict):
            if MODELS is None:
                raise RuntimeError('mmseg.registry.MODELS is unavailable')
            return MODELS.build(copy.deepcopy(model))
        raise TypeError(f'Unsupported model type: {type(model)}')

    def _build_dataset(self, dataset):
        _ensure_mmseg_registries()
        if dataset is None:
            return None
        if isinstance(dataset, dict):
            if DATASETS is None:
                raise RuntimeError('mmseg.registry.DATASETS is unavailable')
            return DATASETS.build(copy.deepcopy(dataset))
        if isinstance(dataset, BaseDataset):
            return dataset
        if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            return dataset
        raise TypeError(f'Unsupported dataset type: {type(dataset)}')

    def _build_sampler(self, sampler_cfg, dataset):
        if sampler_cfg is None:
            return DefaultSampler(dataset=dataset, shuffle=False)
        if hasattr(sampler_cfg, '__iter__') and not isinstance(sampler_cfg, dict):
            return sampler_cfg
        if isinstance(sampler_cfg, dict):
            cfg = copy.deepcopy(sampler_cfg)
            sampler_type = cfg.pop('type', DefaultSampler)
        else:
            sampler_type = sampler_cfg
            cfg = {}
        if isinstance(sampler_type, str):
            sampler_type = {
                'DefaultSampler': DefaultSampler,
                'InfiniteSampler': InfiniteSampler
            }.get(sampler_type, DefaultSampler)
        return sampler_type(dataset=dataset, **cfg)

    def _build_dataloader(self, dataloader_cfg):
        if dataloader_cfg is None:
            return None
        if hasattr(dataloader_cfg, '__iter__') and hasattr(dataloader_cfg, '__len__') and not isinstance(dataloader_cfg, dict):
            return dataloader_cfg
        if not isinstance(dataloader_cfg, dict):
            raise TypeError(f'Unsupported dataloader config: {type(dataloader_cfg)}')

        cfg = copy.deepcopy(dataloader_cfg)
        dataset = self._build_dataset(cfg.pop('dataset', None))
        if dataset is None:
            raise RuntimeError('dataloader config must provide "dataset"')
        sampler = self._build_sampler(cfg.pop('sampler', None), dataset)
        batch_size = int(cfg.pop('batch_size', 1))
        drop_last = bool(cfg.pop('drop_last', False))
        return _SimpleDataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last)

    def _build_optimizer(self, optimizer_cfg):
        if optimizer_cfg is None:
            return None
        if hasattr(optimizer_cfg, 'step') and hasattr(optimizer_cfg, 'state_dict'):
            return optimizer_cfg
        if not isinstance(optimizer_cfg, dict):
            raise TypeError(f'Unsupported optimizer config: {type(optimizer_cfg)}')

        cfg = copy.deepcopy(optimizer_cfg)
        optim_type = cfg.pop('type')
        if isinstance(optim_type, str):
            optim_type = {
                'SGD': jt.optim.SGD,
                'AdamW': jt.optim.AdamW,
                'Adam': jt.optim.Adam,
            }.get(optim_type, None)
            if optim_type is None:
                raise KeyError(f'Unknown optimizer type: {optimizer_cfg["type"]}')
        params = self.model.parameters() if hasattr(self.model, 'parameters') else []
        return optim_type(params=params, **cfg)

    def _build_optim_wrapper(self, optim_wrapper_cfg):
        if optim_wrapper_cfg is None:
            return None
        if isinstance(optim_wrapper_cfg, OptimWrapper):
            return optim_wrapper_cfg
        if hasattr(optim_wrapper_cfg, 'update_params'):
            return optim_wrapper_cfg
        if not isinstance(optim_wrapper_cfg, dict):
            optimizer = self._build_optimizer(optim_wrapper_cfg)
            return OptimWrapper(optimizer=optimizer)

        cfg = copy.deepcopy(optim_wrapper_cfg)
        wrapper_type = cfg.pop('type', OptimWrapper)
        optimizer_cfg = cfg.pop('optimizer', None)

        if optimizer_cfg is None and 'lr' in cfg and 'type' in optim_wrapper_cfg:
            # allow passing optimizer config directly
            optimizer_cfg = copy.deepcopy(optim_wrapper_cfg)
            optimizer_cfg.pop('clip_grad', None)
            optimizer_cfg.pop('accumulative_counts', None)
            wrapper_type = OptimWrapper
            cfg = {}

        optimizer = self._build_optimizer(optimizer_cfg)
        if isinstance(wrapper_type, str):
            if wrapper_type != 'OptimWrapper':
                raise KeyError(f'Unknown optim wrapper type: {wrapper_type}')
            wrapper_type = OptimWrapper
        return wrapper_type(optimizer=optimizer, **cfg)

    def _build_param_schedulers(self, scheduler_cfg):
        if scheduler_cfg is None:
            return []
        if not isinstance(scheduler_cfg, (list, tuple)):
            scheduler_cfg = [scheduler_cfg]

        schedulers = []
        for item in scheduler_cfg:
            if hasattr(item, 'step') and hasattr(item, 'state_dict'):
                schedulers.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError(f'Unsupported scheduler config: {type(item)}')
            cfg = copy.deepcopy(item)
            scheduler_type = cfg.pop('type')
            if isinstance(scheduler_type, str):
                scheduler_type = _SCHEDULER_NAME_MAP.get(scheduler_type, None)
                if scheduler_type is None:
                    raise KeyError(f'Unknown scheduler type: {item["type"]}')
            schedulers.append(
                scheduler_type(optimizer=self.optim_wrapper, **cfg))
        return schedulers

    def _build_metric(self, metric_cfg):
        _ensure_mmseg_registries()
        if isinstance(metric_cfg, BaseMetric):
            return metric_cfg
        if not isinstance(metric_cfg, dict):
            raise TypeError(f'Unsupported evaluator type: {type(metric_cfg)}')

        cfg = copy.deepcopy(metric_cfg)
        metric_type = cfg.pop('type')
        if isinstance(metric_type, str):
            if METRICS is None:
                raise RuntimeError('mmseg.registry.METRICS is unavailable')
            metric_type = METRICS.get(metric_type)
        if metric_type is None:
            raise RuntimeError(f'Unable to resolve metric type: {metric_cfg}')
        return metric_type(**cfg)

    def _build_evaluator(self, evaluator_cfg, dataset):
        if evaluator_cfg is None:
            return None
        if isinstance(evaluator_cfg, list):
            metrics = [self._build_metric(x) for x in evaluator_cfg]
            evaluator = _MetricListEvaluator(metrics)
        elif isinstance(evaluator_cfg, BaseMetric):
            evaluator = evaluator_cfg
        elif isinstance(evaluator_cfg, dict):
            evaluator = self._build_metric(evaluator_cfg)
        else:
            evaluator = evaluator_cfg

        if hasattr(dataset, 'metainfo') and hasattr(evaluator, 'dataset_meta'):
            evaluator.dataset_meta = dataset.metainfo
        return evaluator

    def _build_loop(self, loop_cfg, default_type):
        if loop_cfg is None:
            if default_type is None:
                return None
            return default_type(runner=self)
        if isinstance(loop_cfg, BaseLoop):
            return loop_cfg
        if not isinstance(loop_cfg, dict):
            raise TypeError(f'Unsupported loop config: {type(loop_cfg)}')
        cfg = copy.deepcopy(loop_cfg)
        loop_type = cfg.pop('type', default_type)
        if isinstance(loop_type, str):
            loop_type = _LOOP_NAME_MAP.get(loop_type, None)
            if loop_type is None:
                raise KeyError(f'Unknown loop type: {loop_cfg["type"]}')
        return loop_type(runner=self, **cfg)

    def _build_single_hook(self, hook_cfg):
        _ensure_mmseg_registries()
        if isinstance(hook_cfg, Hook):
            return hook_cfg
        if not isinstance(hook_cfg, dict):
            raise TypeError(f'Unsupported hook config: {type(hook_cfg)}')
        cfg = copy.deepcopy(hook_cfg)
        hook_type = cfg.pop('type')
        if isinstance(hook_type, str):
            hook_cls = _HOOK_NAME_MAP.get(hook_type)
            if hook_cls is None and MMSEG_HOOKS is not None:
                hook_cls = MMSEG_HOOKS.get(hook_type)
            if hook_cls is None:
                raise KeyError(f'Unknown hook type: {hook_type}')
            hook_type = hook_cls
        return hook_type(**cfg)

    def _build_hooks(self, default_hooks=None, custom_hooks=None):
        hooks = []
        if default_hooks is None:
            default_hooks = {
                'timer': {
                    'type': IterTimerHook
                },
                'logger': {
                    'type': LoggerHook,
                    'interval': 50
                },
                'param_scheduler': {
                    'type': ParamSchedulerHook
                },
                'checkpoint': {
                    'type': CheckpointHook,
                    'interval': 1,
                    'by_epoch': True
                },
            }

        if isinstance(default_hooks, dict):
            default_hooks = list(default_hooks.values())
        for hook_cfg in default_hooks:
            hooks.append(self._build_single_hook(hook_cfg))

        if custom_hooks is not None:
            if isinstance(custom_hooks, dict):
                custom_hooks = [custom_hooks]
            for hook_cfg in custom_hooks:
                hooks.append(self._build_single_hook(hook_cfg))

        hooks.sort(key=lambda x: _priority_value(getattr(x, 'priority', 'NORMAL')))
        return hooks

    def set_log_var(self, key, value):
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except Exception:
                pass
        self._log_vars[key] = value

    def get_log_vars(self):
        return dict(self._log_vars)

    def clear_log_vars(self):
        self._log_vars.clear()

    def call_hook(self, fn_name, **kwargs):
        for hook in self._hooks:
            fn = getattr(hook, fn_name, None)
            if callable(fn):
                fn(self, **kwargs)

    def _forward_with_mode(self, data_batch, mode):
        model = self.model
        phase_map = {
            'loss': 'loss',
            'predict': 'predict',
            'tensor': 'tensor'
        }
        phase = phase_map[mode]

        if isinstance(data_batch, dict):
            # mmseg-style signature: model(inputs, data_samples, mode=...)
            if 'inputs' in data_batch:
                inputs = data_batch.get('inputs')
                data_samples = data_batch.get('data_samples')
                try:
                    return model(inputs, data_samples=data_samples, mode=mode)
                except TypeError:
                    pass
            try:
                return model(**data_batch, mode=mode)
            except TypeError:
                pass
            try:
                return model(data_batch, phase=phase)
            except TypeError:
                pass
            try:
                return model(data_batch, mode=mode)
            except TypeError:
                pass
            return model(data_batch)

        try:
            return model(data_batch, mode=mode)
        except TypeError:
            pass
        try:
            return model(data_batch, phase=phase)
        except TypeError:
            pass
        return model(data_batch)

    @staticmethod
    def _parse_train_outputs(outputs):
        if isinstance(outputs, tuple):
            if len(outputs) == 2:
                # common case: (loss, log_vars) or (pred, loss)
                a, b = outputs
                if isinstance(b, dict):
                    loss = a
                    log_vars = b
                    return loss, log_vars
                loss = b
                return loss, {'loss': loss}
            if len(outputs) >= 1:
                loss = outputs[-1]
                return loss, {'loss': loss}

        if isinstance(outputs, dict):
            if 'loss' in outputs:
                return outputs['loss'], outputs
            total = None
            log_vars = {}
            for key, value in outputs.items():
                if key.startswith('loss'):
                    total = value if total is None else total + value
                log_vars[key] = value
            if total is None:
                total = next(iter(outputs.values()))
            log_vars.setdefault('loss', total)
            return total, log_vars

        return outputs, {'loss': outputs}

    def run_train_step(self, data_batch):
        outputs = self._forward_with_mode(data_batch, mode='loss')
        loss, log_vars = self._parse_train_outputs(outputs)
        return loss, log_vars

    def run_val_step(self, data_batch):
        return self._forward_with_mode(data_batch, mode='predict')

    def run_test_step(self, data_batch):
        return self._forward_with_mode(data_batch, mode='predict')

    def _ensure_train_components(self):
        if self.train_dataloader is None:
            self.train_dataloader = self._build_dataloader(self.train_dataloader_cfg)
        if self.optim_wrapper is None:
            self.optim_wrapper = self._build_optim_wrapper(self.optim_wrapper_cfg)
        if not self.param_schedulers:
            self.param_schedulers = self._build_param_schedulers(self.param_scheduler_cfg)
        if self.train_loop is None:
            self.train_loop = self._build_loop(self.train_cfg, default_type=IterBasedTrainLoop)

    def _ensure_val_components(self):
        if self.val_dataloader is None and self.val_dataloader_cfg is not None:
            self.val_dataloader = self._build_dataloader(self.val_dataloader_cfg)
        if self.val_evaluator is None and self.val_evaluator_cfg is not None and self.val_dataloader is not None:
            dataset = getattr(self.val_dataloader, 'dataset', None)
            self.val_evaluator = self._build_evaluator(self.val_evaluator_cfg, dataset)
        if self.val_loop is None and self.val_dataloader is not None and self.val_evaluator is not None:
            self.val_loop = self._build_loop(self.val_cfg, default_type=ValLoop)

    def _ensure_test_components(self):
        if self.test_dataloader is None and self.test_dataloader_cfg is not None:
            self.test_dataloader = self._build_dataloader(self.test_dataloader_cfg)
        if self.test_evaluator is None and self.test_evaluator_cfg is not None and self.test_dataloader is not None:
            dataset = getattr(self.test_dataloader, 'dataset', None)
            self.test_evaluator = self._build_evaluator(self.test_evaluator_cfg, dataset)
        if self.test_loop is None and self.test_dataloader is not None and self.test_evaluator is not None:
            self.test_loop = self._build_loop(self.test_cfg, default_type=TestLoop)

    def load_checkpoint(self, filename, resume=False):
        ckpt = _load_checkpoint(filename)
        state_dict = _extract_state_dict(ckpt)
        try:
            self.model.load_state_dict(state_dict)
        except Exception:
            if hasattr(self.model, 'load_parameters'):
                self.model.load_parameters(state_dict)
            else:
                raise

        if resume:
            if self.train_loop is not None and 'loop' in ckpt:
                self.train_loop.load_state_dict(ckpt['loop'])
            if self.optim_wrapper is not None and 'optimizer' in ckpt:
                self.optim_wrapper.load_state_dict(ckpt['optimizer'])
            if self.param_schedulers and 'param_schedulers' in ckpt:
                sched_states = ckpt['param_schedulers']
                for scheduler, state in zip(self.param_schedulers, sched_states):
                    if hasattr(scheduler, 'load_state_dict'):
                        scheduler.load_state_dict(state)
        return ckpt

    def save_checkpoint(self, filename):
        os.makedirs(osp.dirname(filename) or '.', exist_ok=True)
        data = {
            'state_dict': self.model.state_dict(),
            'iter': self.iter,
        }
        if self.train_loop is not None:
            data['loop'] = self.train_loop.state_dict()
        if self.optim_wrapper is not None:
            data['optimizer'] = self.optim_wrapper.state_dict()
        if self.param_schedulers:
            data['param_schedulers'] = [x.state_dict() for x in self.param_schedulers]
        jt.save(data, filename)
        return filename

    def train(self):
        self._ensure_train_components()
        self._ensure_val_components()
        if self.load_from:
            self.load_checkpoint(self.load_from, resume=self.resume)
        if self.resume_from:
            self.load_checkpoint(self.resume_from, resume=True)
        self.call_hook('before_run')
        self.train_loop.run()
        self.call_hook('after_run')

    def val(self):
        self._ensure_val_components()
        if self.load_from:
            self.load_checkpoint(self.load_from, resume=False)
        self.call_hook('before_run')
        self.val_loop.run()
        self.call_hook('after_run')

    def test(self):
        self._ensure_test_components()
        if self.load_from:
            self.load_checkpoint(self.load_from, resume=False)
        self.call_hook('before_run')
        self.test_loop.run()
        self.call_hook('after_run')


__all__ = ['Runner']
