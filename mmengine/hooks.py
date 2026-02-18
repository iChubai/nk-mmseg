"""Hook implementations compatible with mmengine defaults."""

from __future__ import annotations

import os
import os.path as osp
import time


class Hook:
    priority = 'NORMAL'

    def __init__(self, priority=None, **kwargs):
        del kwargs
        if priority is not None:
            self.priority = priority

    def before_run(self, runner):
        del runner

    def after_run(self, runner):
        del runner

    def before_train(self, runner):
        del runner

    def after_train(self, runner):
        del runner

    def before_val(self, runner):
        del runner

    def after_val(self, runner):
        del runner

    def before_test(self, runner):
        del runner

    def after_test(self, runner):
        del runner

    def before_train_epoch(self, runner):
        del runner

    def after_train_epoch(self, runner):
        del runner

    def before_val_epoch(self, runner):
        del runner

    def after_val_epoch(self, runner, metrics=None):
        del runner, metrics

    def before_test_epoch(self, runner):
        del runner

    def after_test_epoch(self, runner, metrics=None):
        del runner, metrics

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        del runner, batch_idx, data_batch

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del runner, batch_idx, data_batch, outputs

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        del runner, batch_idx, data_batch

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del runner, batch_idx, data_batch, outputs

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        del runner, batch_idx, data_batch

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del runner, batch_idx, data_batch, outputs

    @staticmethod
    def every_n_inner_iters(batch_idx, n):
        return n > 0 and (batch_idx + 1) % n == 0

    @staticmethod
    def every_n_train_iters(runner, n):
        cur_iter = getattr(runner.train_loop, 'cur_iter', 0)
        return n > 0 and (cur_iter + 1) % n == 0

    @staticmethod
    def every_n_epochs(runner, n):
        cur_epoch = getattr(runner.train_loop, 'cur_epoch', 0)
        return n > 0 and (cur_epoch + 1) % n == 0


class CheckpointHook(Hook):

    def __init__(self,
                 interval=1,
                 by_epoch=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.interval = int(interval)
        self.by_epoch = bool(by_epoch)
        self.out_dir = out_dir
        self.max_keep_ckpts = int(max_keep_ckpts)
        self.save_last = bool(save_last)
        self._saved = []

    def _save(self, runner, filename):
        out_dir = self.out_dir or runner.work_dir
        os.makedirs(out_dir, exist_ok=True)
        path = osp.join(out_dir, filename)
        runner.save_checkpoint(path)
        self._saved.append(path)
        if self.max_keep_ckpts > 0 and len(self._saved) > self.max_keep_ckpts:
            old = self._saved.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
        runner.logger.info(f'Saved checkpoint to {path}')

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        cur_epoch = getattr(runner.train_loop, 'cur_epoch', 0) + 1
        if self.every_n_epochs(runner, self.interval):
            self._save(runner, f'epoch_{cur_epoch}.pth')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del batch_idx, data_batch, outputs
        if self.by_epoch:
            return
        if self.every_n_train_iters(runner, self.interval):
            cur_iter = getattr(runner.train_loop, 'cur_iter', 0) + 1
            self._save(runner, f'iter_{cur_iter}.pth')


class DistSamplerSeedHook(Hook):

    def before_train_epoch(self, runner):
        dataloader = getattr(runner, 'train_dataloader', None)
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(getattr(runner.train_loop, 'cur_epoch', 0))


class IterTimerHook(Hook):

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        del runner, batch_idx, data_batch
        self._train_start = time.time()

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        del runner, batch_idx, data_batch
        self._val_start = time.time()

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        del runner, batch_idx, data_batch
        self._test_start = time.time()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del batch_idx, data_batch, outputs
        runner.set_log_var('time', time.time() - getattr(self, '_train_start', time.time()))

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del batch_idx, data_batch, outputs
        runner.set_log_var('time', time.time() - getattr(self, '_val_start', time.time()))

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del batch_idx, data_batch, outputs
        runner.set_log_var('time', time.time() - getattr(self, '_test_start', time.time()))


class LoggerHook(Hook):

    def __init__(self,
                 interval=50,
                 interval_exp_name=1000,
                 log_metric_by_epoch=False,
                 ignore_last=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.interval = int(interval)
        self.interval_exp_name = int(interval_exp_name)
        self.log_metric_by_epoch = bool(log_metric_by_epoch)
        self.ignore_last = bool(ignore_last)

    @staticmethod
    def _fmt_dict(data):
        items = []
        for k, v in data.items():
            try:
                fv = float(v)
                items.append(f'{k}: {fv:.4f}')
            except Exception:
                items.append(f'{k}: {v}')
        return '  '.join(items)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del data_batch
        if outputs:
            for k, v in outputs.items():
                runner.set_log_var(k, v)

        if not self.every_n_inner_iters(batch_idx, self.interval):
            return

        cur_iter = getattr(runner.train_loop, 'cur_iter', 0) + 1
        max_iters = getattr(runner.train_loop, 'max_iters', '?')
        lr = getattr(runner.optim_wrapper, 'lr', 0.0)
        logs = runner.get_log_vars()
        msg = f'(train) [{cur_iter}/{max_iters}] lr: {lr:.4e}'
        if logs:
            msg += '  ' + self._fmt_dict(logs)
        runner.logger.info(msg)

    def after_val_epoch(self, runner, metrics=None):
        if metrics is None:
            return
        runner.logger.info('(val) ' + self._fmt_dict(metrics))

    def after_test_epoch(self, runner, metrics=None):
        if metrics is None:
            return
        runner.logger.info('(test) ' + self._fmt_dict(metrics))


class ParamSchedulerHook(Hook):

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        del batch_idx, data_batch, outputs
        for scheduler in getattr(runner, 'param_schedulers', []):
            if not getattr(scheduler, 'by_epoch', False):
                scheduler.step()

    def after_train_epoch(self, runner):
        for scheduler in getattr(runner, 'param_schedulers', []):
            if getattr(scheduler, 'by_epoch', False):
                scheduler.step()


__all__ = [
    'Hook', 'CheckpointHook', 'DistSamplerSeedHook', 'IterTimerHook',
    'LoggerHook', 'ParamSchedulerHook'
]
