# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import os.path as osp
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist

try:
    from prettytable import PrettyTable
except Exception:  # pragma: no cover
    PrettyTable = None

from mmseg.registry import METRICS


@METRICS.register_module()
class DepthMetric(BaseMetric):
    """Depth estimation metric implemented without torch dependency."""

    METRICS = ('d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog')

    def __init__(self,
                 depth_metrics: Optional[List[str]] = None,
                 min_depth_eval: float = 0.0,
                 max_depth_eval: float = float('inf'),
                 crop_type: Optional[str] = None,
                 depth_scale_factor: float = 1.0,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        del kwargs
        super().__init__(collect_device=collect_device, prefix=prefix)

        if depth_metrics is None:
            self.metrics = self.METRICS
        elif isinstance(depth_metrics, (tuple, list)):
            for metric in depth_metrics:
                assert metric in self.METRICS, (
                    f'the metric {metric} is not supported. '
                    f'Please use metrics in {self.METRICS}')
            self.metrics = depth_metrics
        else:
            raise TypeError('depth_metrics must be list/tuple or None')

        assert crop_type in [None, 'nyu_crop'], (
            f'Invalid crop_type {crop_type}, expected None or nyu_crop')

        self.crop_type = crop_type
        self.min_depth_eval = min_depth_eval
        self.max_depth_eval = max_depth_eval
        self.output_dir = output_dir
        self.format_only = format_only
        self.depth_scale_factor = depth_scale_factor
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, 'numpy'):
            x = x.numpy()
        return np.asarray(x)

    @staticmethod
    def _fetch(sample, key):
        if isinstance(sample, dict):
            return sample[key]
        return getattr(sample, key)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        del data_batch
        for data_sample in data_samples:
            pred_obj = self._fetch(data_sample, 'pred_depth_map')
            pred_data = pred_obj['data'] if isinstance(pred_obj, dict) else pred_obj.data
            pred_label = self._to_numpy(pred_data).squeeze()

            if not self.format_only:
                gt_obj = self._fetch(data_sample, 'gt_depth_map')
                gt_data = gt_obj['data'] if isinstance(gt_obj, dict) else gt_obj.data
                gt_depth = self._to_numpy(gt_data).squeeze()
                eval_mask = self._get_eval_mask(gt_depth)
                self.results.append((gt_depth[eval_mask], pred_label[eval_mask]))

            if self.output_dir is not None:
                img_path = self._fetch(data_sample, 'img_path') if not isinstance(data_sample, dict) else data_sample.get('img_path')
                basename = osp.splitext(osp.basename(img_path))[0] if img_path else 'pred'
                png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label * self.depth_scale_factor
                cv2.imwrite(png_filename, output_mask.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def _get_eval_mask(self, gt_depth: np.ndarray):
        valid_mask = np.logical_and(gt_depth > self.min_depth_eval,
                                    gt_depth < self.max_depth_eval)

        if self.crop_type == 'nyu_crop':
            crop_mask = np.zeros_like(valid_mask, dtype=bool)
            crop_mask[45:471, 41:601] = True
        else:
            crop_mask = np.ones_like(valid_mask, dtype=bool)

        return np.logical_and(valid_mask, crop_mask)

    @staticmethod
    def _calc_all_metrics(gt_depth: np.ndarray, pred_depth: np.ndarray):
        assert gt_depth.shape == pred_depth.shape

        eps = 1e-12
        pred_depth = np.maximum(pred_depth, eps)
        gt_depth = np.maximum(gt_depth, eps)

        thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
        diff = pred_depth - gt_depth
        diff_log = np.log(pred_depth) - np.log(gt_depth)

        d1 = np.mean(thresh < 1.25)
        d2 = np.mean(thresh < 1.25**2)
        d3 = np.mean(thresh < 1.25**3)

        abs_rel = np.mean(np.abs(diff) / gt_depth)
        sq_rel = np.mean(np.square(diff) / gt_depth)

        rmse = np.sqrt(np.mean(np.square(diff)))
        rmse_log = np.sqrt(np.mean(np.square(diff_log)))

        log10 = np.mean(np.abs(np.log10(pred_depth) - np.log10(gt_depth)))
        silog = np.sqrt(np.mean(np.square(diff_log)) -
                        0.5 * np.square(np.mean(diff_log)))

        return {
            'd1': float(d1),
            'd2': float(d2),
            'd3': float(d3),
            'abs_rel': float(abs_rel),
            'sq_rel': float(sq_rel),
            'rmse': float(rmse),
            'rmse_log': float(rmse_log),
            'log10': float(log10),
            'silog': float(silog),
        }

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        if not results:
            return {k: 0.0 for k in self.metrics}

        metrics = defaultdict(list)
        for gt_depth, pred_depth in results:
            for key, value in self._calc_all_metrics(gt_depth,
                                                     pred_depth).items():
                metrics[key].append(value)
        metrics = {k: sum(metrics[k]) / len(metrics[k]) for k in self.metrics}

        if PrettyTable is not None:
            table_data = PrettyTable()
            for key, val in metrics.items():
                table_data.add_column(key, [round(val, 5)])
            print_log('results:', logger)
            print_log('\n' + table_data.get_string(), logger=logger)

        return metrics
