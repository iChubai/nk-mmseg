"""Loading transforms compatibility."""

from __future__ import annotations

import mmengine.fileio as fileio
import numpy as np

from mmcv import imfrombytes
from .base import BaseTransform


class LoadImageFromFile(BaseTransform):

    def __init__(self, to_float32=False, color_type='color', backend_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args

    def transform(self, results):
        img_bytes = fileio.get(results['img_path'], backend_args=self.backend_args)
        img = imfrombytes(img_bytes, flag='color')
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


class LoadAnnotations(BaseTransform):

    def __init__(self,
                 with_bbox=False,
                 with_label=False,
                 with_seg=False,
                 with_keypoints=False,
                 imdecode_backend='cv2',
                 backend_args=None):
        del with_bbox, with_label, with_keypoints
        self.with_seg = with_seg
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args

    def _load_seg_map(self, results):
        img_bytes = fileio.get(results['seg_map_path'], backend_args=self.backend_args)
        gt = imfrombytes(img_bytes, flag='unchanged', backend=self.imdecode_backend)
        if gt.ndim == 3:
            gt = gt[..., 0]
        results['gt_seg_map'] = gt.astype(np.uint8)
        results.setdefault('seg_fields', []).append('gt_seg_map')

    def transform(self, results):
        if self.with_seg:
            self._load_seg_map(results)
        return results
