# Copyright (c) OpenMMLab. All rights reserved.
"""Pure-Jittor remote sensing inferencer (lightweight implementation)."""

from __future__ import annotations

import os.path as osp
from typing import List, Optional, Tuple

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

from .inference import inference_model, init_model

try:
    from osgeo import gdal
except ImportError:
    gdal = None


class RSImage:

    def __init__(self, image):
        if gdal is None:
            raise ImportError('gdal is required for remote sensing inference.')
        self.dataset = gdal.Open(image, gdal.GA_ReadOnly) if isinstance(
            image, str) else image
        assert isinstance(self.dataset, gdal.Dataset), f'{image} is not a valid image'
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.channel = self.dataset.RasterCount
        self.trans = self.dataset.GetGeoTransform()
        self.proj = self.dataset.GetProjection()
        self.grids = []

    def read(self, grid: Optional[List] = None) -> np.ndarray:
        if grid is None:
            data = self.dataset.ReadAsArray()
        else:
            data = self.dataset.ReadAsArray(*grid[:4])
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        return np.einsum('ijk->jki', data)

    def create_seg_map(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = 'output_label.tif'
        driver = gdal.GetDriverByName('GTiff')
        seg_map = driver.Create(output_path, self.width, self.height, 1,
                                gdal.GDT_Byte)
        seg_map.SetGeoTransform(self.trans)
        seg_map.SetProjection(self.proj)
        return RSImage(seg_map)

    def write(self, data: np.ndarray, grid: Optional[List] = None):
        band = self.dataset.GetRasterBand(1)
        if grid is None:
            band.WriteArray(data)
        else:
            x_off, y_off, _, _, x_crop, y_crop, w_crop, h_crop = grid
            patch = data[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
            band.WriteArray(patch, x_off + x_crop, y_off + y_crop)

    def create_grids(self,
                     window_size: Tuple[int, int],
                     stride: Tuple[int, int] = (0, 0)):
        win_w, win_h = window_size
        stride_x = win_w if stride[0] == 0 else stride[0]
        stride_y = win_h if stride[1] == 0 else stride[1]

        self.grids = []
        for y in range(0, self.height, stride_y):
            for x in range(0, self.width, stride_x):
                x0 = min(x, max(0, self.width - win_w))
                y0 = min(y, max(0, self.height - win_h))
                x_size = min(win_w, self.width - x0)
                y_size = min(win_h, self.height - y0)
                self.grids.append([x0, y0, x_size, y_size, 0, 0, x_size, y_size])


class RSInferencer:

    def __init__(self, model, batch_size: int = 1, thread: int = 1):
        del batch_size, thread
        self.model = model

    @classmethod
    def from_config_path(cls,
                         config_path: str,
                         checkpoint_path: str,
                         batch_size: int = 1,
                         thread: int = 1,
                         device: Optional[str] = 'cuda:0'):
        init_default_scope('mmseg')
        _ = Config.fromfile(config_path)
        model = init_model(config_path, checkpoint=checkpoint_path, device=device)
        return cls(model, batch_size, thread)

    def run(self,
            image: str,
            output_path: Optional[str] = None,
            window_size: Tuple[int, int] = (512, 512),
            stride: Tuple[int, int] = (256, 256)):
        rs_img = RSImage(image)
        rs_img.create_grids(window_size=window_size, stride=stride)
        out = rs_img.create_seg_map(output_path)

        for grid in rs_img.grids:
            patch = rs_img.read(grid)
            pred = inference_model(self.model, patch)
            seg = pred.pred_sem_seg.data
            if hasattr(seg, 'numpy'):
                seg = seg.numpy()
            seg = seg.squeeze().astype(np.uint8)
            out.write(seg, grid)

        return getattr(out, 'path', output_path)
