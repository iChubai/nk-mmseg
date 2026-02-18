from __future__ import annotations

from typing import Any, Optional


class PixelData(dict):

    def __init__(self, data=None, **kwargs):
        super().__init__()
        if data is not None:
            self['data'] = data
        for k, v in kwargs.items():
            self[k] = v

    @property
    def data(self):
        return self.get('data')


class SegDataSample(dict):
    """A lightweight but mmengine-like segmentation data sample."""

    def __init__(self, metainfo: Optional[dict] = None, **kwargs):
        super().__init__()
        self.metainfo = {}
        data = kwargs.pop('data', None)
        if metainfo is not None:
            self.set_metainfo(dict(metainfo))
        if isinstance(data, dict):
            self.set_data(data)
        if kwargs:
            self.set_data(kwargs)

    def set_pred_sem_seg(self, seg):
        self['pred_sem_seg'] = PixelData(data=seg)
        return self

    def set_gt_sem_seg(self, seg):
        self['gt_sem_seg'] = PixelData(data=seg)
        return self

    def set_metainfo(self, meta: dict):
        self.metainfo.update(meta)
        return self

    def set_data(self, data: dict):
        for k, v in data.items():
            self[k] = v
        return self

    def get(self, key: str, default: Any = None):
        if key == 'metainfo':
            return self.metainfo
        return super().get(key, default)

    def __getattr__(self, key: str):
        if key in ('metainfo',):
            return super().__getattribute__(key)
        if key in self:
            return self[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any):
        if key in ('metainfo',):
            super().__setattr__(key, value)
        else:
            self[key] = value
