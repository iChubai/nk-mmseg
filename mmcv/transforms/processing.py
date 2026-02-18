"""Processing transforms compatibility."""

from __future__ import annotations

import copy
import random

import numpy as np

from mmcv import imflip, imresize, imrescale
from .base import BaseTransform


class RandomFlip(BaseTransform):

    def __init__(self, prob=None, direction='horizontal', swap_seg_labels=None):
        self.prob = prob
        self.direction = direction
        self.swap_seg_labels = swap_seg_labels

    def _choose_direction(self):
        if self.prob is None:
            return None
        direction = self.direction
        if isinstance(direction, str):
            direction = [direction]
        if isinstance(self.prob, (float, int)):
            p = float(self.prob)
            if random.random() >= p:
                return None
            return random.choice(direction)
        # list prob
        if len(self.prob) != len(direction):
            raise ValueError('len(prob) must equal len(direction)')
        r = random.random()
        cum = 0.0
        for p, d in zip(self.prob, direction):
            cum += float(p)
            if r < cum:
                return d
        return None

    def _flip_bbox(self, bboxes, img_shape, direction):
        h, w = img_shape
        out = bboxes.copy()
        if direction == 'horizontal':
            out[:, 0], out[:, 2] = w - bboxes[:, 2], w - bboxes[:, 0]
        elif direction == 'vertical':
            out[:, 1], out[:, 3] = h - bboxes[:, 3], h - bboxes[:, 1]
        elif direction == 'diagonal':
            out[:, 0], out[:, 2] = w - bboxes[:, 2], w - bboxes[:, 0]
            out[:, 1], out[:, 3] = h - bboxes[:, 3], h - bboxes[:, 1]
        return out

    def _flip_seg_map(self, seg, direction='horizontal'):
        return imflip(seg, direction=direction)

    def _flip(self, results):
        results['img'] = imflip(results['img'], direction=results['flip_direction'])

    def transform(self, results):
        flip_direction = self._choose_direction()
        results['flip'] = flip_direction is not None
        results['flip_direction'] = flip_direction
        if flip_direction is not None:
            self._flip(results)
        return results


class Resize(BaseTransform):

    def __init__(self,
                 scale=None,
                 scale_factor=None,
                 keep_ratio=False,
                 clip_object_border=True,
                 backend='cv2',
                 interpolation='bilinear'):
        del clip_object_border
        self.scale = scale
        self.scale_factor = scale_factor
        self.keep_ratio = keep_ratio
        self.backend = backend
        self.interpolation = interpolation

    def _target_scale(self, img):
        h, w = img.shape[:2]
        if self.scale is not None:
            return self.scale
        if self.scale_factor is None:
            return (w, h)
        if isinstance(self.scale_factor, (tuple, list)):
            sx, sy = self.scale_factor
            return (int(round(w * sx)), int(round(h * sy)))
        s = float(self.scale_factor)
        return (int(round(w * s)), int(round(h * s)))

    def _resize_img(self, results):
        target = self._target_scale(results['img'])
        if self.keep_ratio:
            img = imrescale(results['img'], target, interpolation=self.interpolation, backend=self.backend)
        else:
            img = imresize(results['img'], target, interpolation=self.interpolation, backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale'] = target
        ori_h, ori_w = results.get('ori_shape', img.shape[:2])[:2]
        results['scale_factor'] = (img.shape[1] / ori_w, img.shape[0] / ori_h)

    def _resize_seg(self, results):
        for seg_key in results.get('seg_fields', []):
            if seg_key not in results:
                continue
            if self.keep_ratio:
                gt = imrescale(results[seg_key], results['scale'], interpolation='nearest', backend=self.backend)
            else:
                gt = imresize(results[seg_key], results['scale'], interpolation='nearest', backend=self.backend)
            results[seg_key] = gt

    def transform(self, results):
        self._resize_img(results)
        self._resize_seg(results)
        return results


class RandomResize(Resize):

    def __init__(self, scale=None, ratio_range=None, keep_ratio=True, **kwargs):
        super().__init__(scale=scale, keep_ratio=keep_ratio, **kwargs)
        self.ratio_range = ratio_range

    def _target_scale(self, img):
        if self.scale is None:
            return super()._target_scale(img)
        if self.ratio_range is None:
            return self.scale
        ratio = random.uniform(float(self.ratio_range[0]),
                               float(self.ratio_range[1]))
        if isinstance(self.scale, (tuple, list)):
            return (int(round(self.scale[0] * ratio)),
                    int(round(self.scale[1] * ratio)))
        return (int(round(self.scale * ratio)), int(round(self.scale * ratio)))


class TestTimeAug(BaseTransform):

    def __init__(self, transforms):
        self._stages = []
        for stage in transforms:
            ops = []
            for t in stage:
                if isinstance(t, dict):
                    ops.append(self._build_transform(t))
                elif callable(t):
                    ops.append(t)
                else:
                    raise TypeError(
                        f'TestTimeAug transform must be dict/callable, got {type(t)}')
            self._stages.append(ops)

    @staticmethod
    def _build_transform(cfg):
        cfg = dict(cfg)
        t_type = cfg.pop('type')
        if t_type in ('Resize', Resize):
            return Resize(**cfg)
        if t_type in ('RandomResize', RandomResize):
            return RandomResize(**cfg)
        if t_type in ('RandomFlip', RandomFlip):
            return RandomFlip(**cfg)
        # Fallback to mmseg registry for dataset-specific transforms.
        from mmseg.registry import TRANSFORMS
        return TRANSFORMS.build(dict(type=t_type, **cfg))

    def transform(self, results):
        # Enumerate Cartesian product over stage branches.
        aug_data = [copy.deepcopy(results)]
        for stage in self._stages:
            if not stage:
                continue
            stage_data = []
            for data in aug_data:
                for op in stage:
                    out = op(copy.deepcopy(data))
                    if out is None:
                        continue
                    if isinstance(out, list):
                        stage_data.extend(out)
                    else:
                        stage_data.append(out)
            if not stage_data:
                return None
            aug_data = stage_data

        # Match mmcv TestTimeAug output shape: dict of lists.
        keys = list(aug_data[0].keys())
        packed = {k: [] for k in keys}
        for data in aug_data:
            for k in keys:
                packed[k].append(data.get(k))
        return packed
