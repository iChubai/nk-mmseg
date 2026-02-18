"""Visualization compatibility layer."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


class LocalVisBackend:

    def __init__(self, save_dir: Optional[str] = None, **kwargs):
        del kwargs
        self.save_dir = save_dir or 'vis_data'
        os.makedirs(self.save_dir, exist_ok=True)

    def add_image(self, name, image, step=0):
        if image is None:
            return None
        safe_name = str(name).replace('/', '_')
        out_file = os.path.join(self.save_dir, f'{safe_name}_{int(step)}.png')
        arr = np.array(image)
        if arr.ndim == 3:
            cv2.imwrite(out_file, arr[:, :, ::-1])
        else:
            cv2.imwrite(out_file, arr)
        return out_file


class Visualizer:
    _instance = None

    def __init__(self,
                 name='visualizer',
                 vis_backends=None,
                 save_dir: Optional[str] = None,
                 **kwargs):
        del kwargs
        self.name = name
        self.save_dir = save_dir
        self.dataset_meta = {'classes': None, 'palette': None}
        self._vis_backends = {}
        self._last_image = None

        if vis_backends is None:
            vis_backends = [dict(type=LocalVisBackend)]

        for i, cfg in enumerate(vis_backends):
            if isinstance(cfg, dict):
                cfg = dict(cfg)
                btype = cfg.pop('type', LocalVisBackend)
                if isinstance(btype, str):
                    if btype != 'LocalVisBackend':
                        continue
                    btype = LocalVisBackend
                backend = btype(
                    save_dir=cfg.pop('save_dir', save_dir), **cfg)
            elif callable(cfg):
                backend = cfg()
            else:
                continue
            self._vis_backends[f'backend_{i}'] = backend

        Visualizer._instance = self

    @classmethod
    def get_current_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_dataset_meta(self, classes=None, palette=None, dataset_name=None):
        del dataset_name
        if classes is not None:
            self.dataset_meta['classes'] = classes
        if palette is not None:
            self.dataset_meta['palette'] = palette

    def _draw_seg(self, image, data_sample):
        vis = np.array(image).copy()
        if data_sample is None:
            return vis

        pred = getattr(data_sample, 'pred_sem_seg', None)
        if pred is None:
            return vis
        seg = getattr(pred, 'data', pred)
        if hasattr(seg, 'numpy'):
            seg = seg.numpy()
        seg = np.array(seg)
        if seg.ndim == 3:
            seg = seg[0]

        palette = self.dataset_meta.get('palette')
        if palette is None:
            num_classes = int(seg.max()) + 1 if seg.size > 0 else 1
            palette = [[(37 * i) % 255, (17 * i) % 255, (97 * i) % 255]
                       for i in range(num_classes)]
        palette = np.array(palette, dtype=np.uint8)

        seg = seg.astype(np.int64)
        seg = np.clip(seg, 0, len(palette) - 1)
        seg_color = palette[seg]
        if seg_color.shape[:2] != vis.shape[:2]:
            seg_color = cv2.resize(
                seg_color,
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        return (vis * 0.5 + seg_color * 0.5).astype(np.uint8)

    def add_image(self, name, image, step=0):
        self._last_image = np.array(image) if image is not None else None
        for backend in self._vis_backends.values():
            if hasattr(backend, 'add_image'):
                backend.add_image(name, self._last_image, step=step)

    def add_datasample(self,
                       name,
                       image,
                       data_sample=None,
                       draw_gt=False,
                       draw_pred=True,
                       show=False,
                       wait_time=0,
                       step=0,
                       out_file=None,
                       **kwargs):
        del draw_gt, kwargs
        vis = np.array(image).copy()
        if draw_pred:
            vis = self._draw_seg(vis, data_sample)

        if out_file is not None:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            if vis.ndim == 3:
                cv2.imwrite(out_file, vis[:, :, ::-1])
            else:
                cv2.imwrite(out_file, vis)
        else:
            for backend in self._vis_backends.values():
                if hasattr(backend, 'add_image'):
                    backend.add_image(name, vis, step=step)

        if show:
            cv2.imshow(name, vis[:, :, ::-1] if vis.ndim == 3 else vis)
            cv2.waitKey(int(wait_time * 1000) if wait_time > 0 else 1)

        self._last_image = vis
        return vis

    def get_image(self):
        return self._last_image


__all__ = ['LocalVisBackend', 'Visualizer']
