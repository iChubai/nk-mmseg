"""Lightweight mmcv compatibility for pure Jittor runtime."""

from __future__ import annotations

import cv2
import numpy as np

__version__ = '2.0.0-jittor-lite'


_INTERP_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


def _interp_code(name: str):
    return _INTERP_MAP.get(name, cv2.INTER_LINEAR)


def imfrombytes(content: bytes, flag='color', backend='cv2'):
    del backend
    arr = np.frombuffer(content, dtype=np.uint8)
    if flag == 'unchanged':
        mode = cv2.IMREAD_UNCHANGED
    elif flag == 'grayscale':
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, mode)
    return img


def imresize(img, size, interpolation='bilinear', backend='cv2'):
    del backend
    if isinstance(size, (tuple, list)):
        w, h = int(size[0]), int(size[1])
    else:
        h, w = int(size), int(size)
    return cv2.resize(img, (w, h), interpolation=_interp_code(interpolation))


def imrescale(img, scale, interpolation='bilinear', backend='cv2'):
    del backend
    h, w = img.shape[:2]
    if isinstance(scale, (tuple, list)):
        target_w, target_h = float(scale[0]), float(scale[1])
        ratio = min(target_w / w, target_h / h)
    else:
        ratio = float(scale)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return imresize(img, (new_w, new_h), interpolation=interpolation)


def imresize_to_multiple(img,
                         divisor,
                         scale_factor=1,
                         interpolation='bilinear'):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h * scale_factor / divisor) * divisor)
    new_w = int(np.ceil(w * scale_factor / divisor) * divisor)
    return imresize(img, (new_w, new_h), interpolation=interpolation)


def imflip(img, direction='horizontal'):
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    if direction == 'vertical':
        return np.flip(img, axis=0)
    if direction == 'diagonal':
        return np.flip(np.flip(img, axis=0), axis=1)
    raise ValueError(f'Unsupported flip direction: {direction}')


def imrotate(img,
             angle,
             border_value=0,
             center=None,
             auto_bound=False,
             interpolation='bilinear'):
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    if auto_bound:
        cos = abs(mat[0, 0])
        sin = abs(mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        mat[0, 2] += (new_w / 2) - center[0]
        mat[1, 2] += (new_h / 2) - center[1]
        out_w, out_h = new_w, new_h
    else:
        out_w, out_h = w, h
    return cv2.warpAffine(
        img,
        mat,
        (out_w, out_h),
        flags=_interp_code(interpolation),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def clahe(channel, clip_limit=40.0, tile_grid_size=(8, 8)):
    clahe_op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe_op.apply(channel)


def lut_transform(img, table):
    return cv2.LUT(img, table)


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
