"""Runner loops compatible with mmengine iter-based training style."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import jittor as jt


class BaseLoop(metaclass=ABCMeta):

    def __init__(self, runner):
        self.runner = runner
        self._iter = 0
        self._epoch = 0

    @property
    def cur_iter(self):
        return self._iter

    @property
    def cur_epoch(self):
        return self._epoch

    def state_dict(self):
        return {'iter': self._iter, 'epoch': self._epoch}

    def load_state_dict(self, data):
        self._iter = int(data.get('iter', self._iter))
        self._epoch = int(data.get('epoch', self._epoch))

    @abstractmethod
    def run(self):
        raise NotImplementedError


class IterBasedTrainLoop(BaseLoop):

    def __init__(self, runner, max_iters, val_interval=0, **kwargs):
        super().__init__(runner)
        del kwargs
        self.max_iters = int(max_iters)
        self.val_interval = int(val_interval)

    def run(self):
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        data_iter = iter(self.runner.train_dataloader)
        while self._iter < self.max_iters:
            try:
                data_batch = next(data_iter)
            except StopIteration:
                self.runner.call_hook('after_train_epoch')
                self._epoch += 1
                self.runner.call_hook('before_train_epoch')
                data_iter = iter(self.runner.train_dataloader)
                data_batch = next(data_iter)

            self.run_iter(data_batch)

            if self.val_interval > 0 and self._iter > 0 and self._iter % self.val_interval == 0:
                if self.runner.val_loop is not None:
                    self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')

    def run_iter(self, data_batch):
        batch_idx = self._iter
        self.runner.iter = self._iter
        self.runner.call_hook(
            'before_train_iter', batch_idx=batch_idx, data_batch=data_batch)

        loss, log_vars = self.runner.run_train_step(data_batch)
        self.runner.optim_wrapper.update_params(loss)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=log_vars)

        self._iter += 1


class _BaseEvalLoop(BaseLoop):

    def __init__(self, runner, dataloader_attr, evaluator_attr):
        super().__init__(runner)
        self._dataloader_attr = dataloader_attr
        self._evaluator_attr = evaluator_attr

    def _iter_data(self):
        dataloader = getattr(self.runner, self._dataloader_attr)
        for idx, data_batch in enumerate(dataloader):
            yield idx, data_batch

    def _evaluate(self, dataset_len):
        evaluator = getattr(self.runner, self._evaluator_attr)
        return evaluator.evaluate(size=dataset_len)

    @staticmethod
    def _as_list(outputs):
        if outputs is None:
            return []
        if isinstance(outputs, list):
            return outputs
        return [outputs]


class ValLoop(_BaseEvalLoop):

    def __init__(self, runner, **kwargs):
        del kwargs
        super().__init__(runner, 'val_dataloader', 'val_evaluator')

    def run(self):
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        self._iter = 0

        for idx, data_batch in self._iter_data():
            self.run_iter(idx, data_batch)

        dataset_len = len(getattr(self.runner.val_dataloader, 'dataset', self.runner.val_dataloader))
        metrics = self._evaluate(dataset_len)
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    @jt.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.run_val_step(data_batch)
        outputs = self._as_list(outputs)
        self.runner.val_evaluator.process(data_batch, outputs)

        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


class TestLoop(_BaseEvalLoop):

    def __init__(self, runner, **kwargs):
        del kwargs
        super().__init__(runner, 'test_dataloader', 'test_evaluator')

    def run(self):
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        self._iter = 0

        for idx, data_batch in self._iter_data():
            self.run_iter(idx, data_batch)

        dataset_len = len(getattr(self.runner.test_dataloader, 'dataset', self.runner.test_dataloader))
        metrics = self._evaluate(dataset_len)
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

    @jt.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.run_test_step(data_batch)
        outputs = self._as_list(outputs)
        self.runner.test_evaluator.process(data_batch, outputs)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


__all__ = ['BaseLoop', 'IterBasedTrainLoop', 'ValLoop', 'TestLoop']
