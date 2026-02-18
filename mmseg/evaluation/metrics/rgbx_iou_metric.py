# Copyright (c) OpenMMLab. All rights reserved.
"""IoU metric compatible with legacy RGBX DFormer batch outputs."""

from __future__ import annotations

import numpy as np

from mmseg.registry import METRICS

from .iou_metric import IoUMetric


@METRICS.register_module()
class RGBXIoUMetric(IoUMetric):

    def __init__(self, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes) if num_classes is not None else None

    @staticmethod
    def _ensure_batch_dim(arr):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr[None, ...]
        return arr

    @staticmethod
    def _to_pred_label(arr):
        arr = np.asarray(arr)
        if arr.ndim == 4:
            # [N, C, H, W] logits
            return np.argmax(arr, axis=1)
        if arr.ndim == 3:
            # either [N, H, W] labels or [C, H, W] logits
            if arr.shape[0] <= 8 and arr.shape[1] > 16 and arr.shape[2] > 16:
                return np.argmax(arr, axis=0)[None, ...]
            return arr
        if arr.ndim == 2:
            return arr[None, ...]
        raise ValueError(f'Unsupported prediction shape: {arr.shape}')

    def _resolve_num_classes(self, pred_label, gt_label):
        if self.dataset_meta and self.dataset_meta.get('classes'):
            return len(self.dataset_meta['classes'])
        if self.num_classes is not None:
            return self.num_classes
        valid_pred = pred_label[pred_label != self.ignore_index]
        valid_gt = gt_label[gt_label != self.ignore_index]
        max_id = -1
        if valid_pred.size:
            max_id = max(max_id, int(valid_pred.max()))
        if valid_gt.size:
            max_id = max(max_id, int(valid_gt.max()))
        return max(max_id + 1, 1)

    def process(self, data_batch: dict, data_samples) -> None:
        # Standard mmseg data sample path: reuse parent behavior.
        if isinstance(data_samples, (list, tuple)) and data_samples:
            first = data_samples[0]
            if isinstance(first, dict) and 'pred_sem_seg' in first:
                return super().process(data_batch, data_samples)

        if isinstance(data_samples, (list, tuple)) and len(data_samples) == 1:
            pred_obj = data_samples[0]
        else:
            pred_obj = data_samples

        if isinstance(pred_obj, (list, tuple)) and pred_obj:
            pred_obj = pred_obj[0]

        pred_np = self._to_numpy(pred_obj)
        pred_label = self._to_pred_label(pred_np)

        if not isinstance(data_batch, dict) or 'label' not in data_batch:
            raise KeyError('RGBXIoUMetric expects data_batch["label"]')
        gt_np = self._to_numpy(data_batch['label'])
        gt_label = self._ensure_batch_dim(gt_np)
        pred_label = self._ensure_batch_dim(pred_label)

        batch_size = min(pred_label.shape[0], gt_label.shape[0])
        for i in range(batch_size):
            num_classes = self._resolve_num_classes(pred_label[i], gt_label[i])
            self.results.append(
                self.intersect_and_union(
                    pred_label[i].astype(np.int64),
                    gt_label[i].astype(np.int64),
                    num_classes,
                    self.ignore_index,
                ))

