# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image

try:
    from prettytable import PrettyTable
except Exception:  # pragma: no cover
    PrettyTable = None

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU metric implemented without torch dependency."""

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        del kwargs
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.format_only = format_only
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
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_obj = self._fetch(data_sample, 'pred_sem_seg')
            pred_data = pred_obj['data'] if isinstance(pred_obj, dict) else pred_obj.data
            pred_label = self._to_numpy(pred_data).squeeze()

            if not self.format_only:
                gt_obj = self._fetch(data_sample, 'gt_sem_seg')
                gt_data = gt_obj['data'] if isinstance(gt_obj, dict) else gt_obj.data
                label = self._to_numpy(gt_data).squeeze()
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))

            if self.output_dir is not None:
                img_path = self._fetch(data_sample, 'img_path') if not isinstance(data_sample, dict) else data_sample.get('img_path')
                basename = osp.splitext(osp.basename(img_path))[0] if img_path else 'pred'
                png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label
                reduce_zero = data_sample.get('reduce_zero_label', False) if isinstance(data_sample, dict) else getattr(data_sample, 'reduce_zero_label', False)
                if reduce_zero:
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        if not results:
            return {'mIoU': 0.0, 'mAcc': 0.0, 'aAcc': 0.0}

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = np.sum(np.stack(results[0], axis=0), axis=0)
        total_area_union = np.sum(np.stack(results[1], axis=0), axis=0)
        total_area_pred_label = np.sum(np.stack(results[2], axis=0), axis=0)
        total_area_label = np.sum(np.stack(results[3], axis=0), axis=0)

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.dataset_meta['classes']
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = {}
        for key, val in ret_metrics_summary.items():
            metrics[key] = val if key == 'aAcc' else val
            if key != 'aAcc':
                metrics['m' + key] = val

        if 'IoU' in metrics and 'mIoU' not in metrics:
            metrics['mIoU'] = metrics['IoU']
        if 'Acc' in metrics and 'mAcc' not in metrics:
            metrics['mAcc'] = metrics['Acc']

        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
            if ret_metric != 'aAcc'
        })

        if PrettyTable is not None:
            ret_metrics_class.update({'Class': class_names})
            ret_metrics_class.move_to_end('Class', last=False)
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)
            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: np.ndarray, label: np.ndarray,
                            num_classes: int, ignore_index: int):
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = np.bincount(intersect.astype(np.int64), minlength=num_classes).astype(np.float64)
        area_pred_label = np.bincount(pred_label.astype(np.int64), minlength=num_classes).astype(np.float64)
        area_label = np.bincount(label.astype(np.int64), minlength=num_classes).astype(np.float64)
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):

        def f_score(precision, recall, beta=1):
            denom = (beta**2 * precision) + recall
            denom = np.where(denom == 0, 1e-12, denom)
            return (1 + beta**2) * (precision * recall) / denom

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / np.maximum(total_area_label.sum(), 1e-12)
        ret_metrics = OrderedDict({'aAcc': all_acc})

        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / np.maximum(total_area_union, 1e-12)
                acc = total_area_intersect / np.maximum(total_area_label, 1e-12)
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / np.maximum(
                    total_area_pred_label + total_area_label, 1e-12)
                acc = total_area_intersect / np.maximum(total_area_label, 1e-12)
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / np.maximum(total_area_pred_label, 1e-12)
                recall = total_area_intersect / np.maximum(total_area_label, 1e-12)
                f_value = f_score(precision, recall, beta)
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
