import os
from typing import Optional

import cv2
import numpy as np


class SegLocalVisualizer:
    """A minimal local visualizer for semantic segmentation."""

    def __init__(self, save_dir: Optional[str] = None, alpha: float = 0.5):
        self.save_dir = save_dir
        self.alpha = alpha
        self.dataset_meta = {'classes': None, 'palette': None}
        self._last_image = None

    def set_dataset_meta(self, classes=None, palette=None, dataset_name=None):
        if classes is not None:
            self.dataset_meta['classes'] = classes
        if palette is not None:
            self.dataset_meta['palette'] = palette

    def add_datasample(self,
                       name,
                       image,
                       data_sample,
                       draw_gt=False,
                       draw_pred=True,
                       wait_time=0,
                       out_file=None,
                       show=False,
                       with_labels=True):
        vis = image.copy()
        if draw_pred and data_sample is not None and \
                getattr(data_sample, 'pred_sem_seg', None) is not None:
            seg = data_sample.pred_sem_seg.data
            if hasattr(seg, 'numpy'):
                seg = seg.numpy()
            if seg.ndim == 3:
                seg = seg[0]
            palette = self.dataset_meta.get('palette')
            if palette is None:
                num_classes = int(seg.max()) + 1
                palette = [[(37 * i) % 255, (17 * i) % 255, (97 * i) % 255]
                           for i in range(num_classes)]
            color_map = np.array(palette, dtype=np.uint8)
            seg_color = color_map[seg.astype(np.int32)]
            if seg_color.shape[:2] != vis.shape[:2]:
                seg_color = cv2.resize(seg_color, (vis.shape[1], vis.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            vis = (vis * (1 - self.alpha) + seg_color * self.alpha).astype(
                np.uint8)

        if out_file is None and self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            out_file = os.path.join(self.save_dir, name)
        if out_file is not None:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            cv2.imwrite(out_file, vis[:, :, ::-1] if vis.ndim == 3 else vis)
        if show:
            cv2.imshow(name, vis[:, :, ::-1] if vis.ndim == 3 else vis)
            cv2.waitKey(int(wait_time * 1000) if wait_time > 0 else 1)
        self._last_image = vis

    def get_image(self):
        return self._last_image

