"""Integration-oriented regressions for mmengine DFormer migration."""

import os
import sys
import unittest

import jittor as jt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmseg.datasets  # noqa: F401
import mmseg.evaluation  # noqa: F401
from mmengine.runner import Runner
from mmseg.registry import DATASETS, METRICS


@DATASETS.register_module(name='ToyRunnerDataset')
class ToyRunnerDataset:

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        value = float(idx)
        return dict(
            data=jt.array([[[value]]], dtype=jt.float32),
            modal_x=jt.array([[[value]]], dtype=jt.float32),
            label=jt.array([[0]], dtype=jt.int64),
        )


class ToyLossModel(jt.nn.Module):

    def execute(self, data, modal_x=None, label=None, mode='loss'):
        del modal_x
        logits = data
        if mode == 'loss':
            return logits.mean()
        return logits


class TestMMEngineDFormerIntegration(unittest.TestCase):

    def setUp(self):
        jt.flags.use_cuda = 0

    def test_runner_builds_dataset_from_cfg_dict(self):
        cfg = dict(
            model=ToyLossModel(),
            train_dataloader=dict(
                batch_size=1,
                sampler=dict(type='DefaultSampler', shuffle=False),
                dataset=dict(type='ToyRunnerDataset'),
            ),
            train_cfg=dict(type='IterBasedTrainLoop', max_iters=1, val_interval=0),
            optim_wrapper=dict(
                type='OptimWrapper',
                optimizer=dict(type='SGD', lr=0.01),
            ),
        )
        runner = Runner.from_cfg(cfg)
        dataloader = runner._build_dataloader(cfg['train_dataloader'])
        self.assertEqual(type(dataloader.dataset).__name__, 'ToyRunnerDataset')
        batch = next(iter(dataloader))
        self.assertEqual(tuple(batch['data'].shape), (1, 1, 1, 1))

    def test_rgbx_iou_metric_accepts_logits_batch(self):
        metric = METRICS.build(
            dict(type='RGBXIoUMetric', ignore_index=255, num_classes=2))
        metric.dataset_meta = {'classes': ('bg', 'fg')}

        gt = jt.array([[[0, 1], [1, 1]]], dtype=jt.int64)
        logits = jt.array(
            [[[[10.0, 0.0], [0.0, 0.0]], [[0.0, 10.0], [10.0, 10.0]]]],
            dtype=jt.float32,
        )

        metric.process(data_batch=dict(label=gt), data_samples=[logits])
        out = metric.evaluate(size=1)
        self.assertIn('mIoU', out)
        self.assertGreaterEqual(float(out['mIoU']), 99.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)

