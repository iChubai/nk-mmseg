"""Minimal dataset abstractions for mmseg compatibility."""

from __future__ import annotations

import copy
import functools
import os.path as osp
from collections.abc import Callable, Sequence
from typing import Any, Optional


class Compose:

    def __init__(self, transforms=None):
        self.transforms = []
        if transforms is None:
            return
        if isinstance(transforms, dict):
            transforms = [transforms]
        for transform in transforms:
            if isinstance(transform, dict):
                # Lazy import to avoid circular import during package bootstrap.
                from mmseg.registry import TRANSFORMS
                transform = TRANSFORMS.build(transform)
            if not callable(transform):
                raise TypeError(f'transform should be callable, got {type(transform)}')
            self.transforms.append(transform)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


class BaseDataset:
    METAINFO: dict = {}

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: Optional[dict] = None,
                 filter_cfg: Optional[dict] = None,
                 indices=None,
                 serialize_data: bool = True,
                 pipeline=None,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 **kwargs):
        del kwargs
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix or {})
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list = []
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.pipeline = Compose(pipeline)
        self._fully_initialized = False

        if self.data_root is not None:
            self._join_prefix()
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self):
        return copy.deepcopy(self._metainfo)

    @classmethod
    def _load_metainfo(cls, metainfo: Optional[dict] = None):
        out = copy.deepcopy(getattr(cls, 'METAINFO', {}))
        if metainfo:
            out.update(metainfo)
        return out

    def _join_prefix(self):
        if self.ann_file and not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.data_root, self.ann_file)
        for k, v in list(self.data_prefix.items()):
            if isinstance(v, str) and v and not osp.isabs(v):
                self.data_prefix[k] = osp.join(self.data_root, v)

    def load_data_list(self):
        raise NotImplementedError

    def filter_data(self):
        return self.data_list

    def full_init(self):
        if self._fully_initialized:
            return
        self.data_list = self.load_data_list()
        if hasattr(self, 'filter_data'):
            self.data_list = self.filter_data()
        if isinstance(self._indices, int):
            self.data_list = self.data_list[:self._indices]
        elif isinstance(self._indices, (list, tuple)):
            self.data_list = [self.data_list[i] for i in self._indices]
        self._fully_initialized = True

    def get_data_info(self, idx: int) -> dict:
        return self.data_list[idx]

    def prepare_data(self, idx: int):
        return self.pipeline(copy.deepcopy(self.get_data_info(idx)))

    def __getitem__(self, idx: int):
        if not self._fully_initialized:
            self.full_init()
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise RuntimeError('Test pipeline should not return None')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is not None:
                return data
            idx = (idx + 1) % len(self.data_list)
        raise RuntimeError(
            f'Cannot find valid data after {self.max_refetch} retries')

    def __len__(self):
        if not self._fully_initialized:
            self.full_init()
        return len(self.data_list)


class ConcatDataset(BaseDataset):

    def __init__(self, datasets, lazy_init=False, **kwargs):
        del kwargs
        self.datasets = []
        from mmseg.registry import DATASETS
        for ds in datasets:
            if isinstance(ds, dict):
                ds = DATASETS.build(ds)
            self.datasets.append(ds)
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self):
        if not self.datasets:
            return {}
        return self.datasets[0].metainfo

    def full_init(self):
        if self._fully_initialized:
            return
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            ds.full_init()
            total += len(ds)
            self.cumulative_sizes.append(total)
        self._fully_initialized = True

    def __len__(self):
        if not self._fully_initialized:
            self.full_init()
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _locate(self, idx):
        for ds_idx, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                prev = 0 if ds_idx == 0 else self.cumulative_sizes[ds_idx - 1]
                return ds_idx, idx - prev
        raise IndexError(idx)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self._locate(idx)
        return self.datasets[ds_idx][sample_idx]

    def get_data_info(self, idx):
        ds_idx, sample_idx = self._locate(idx)
        return self.datasets[ds_idx].get_data_info(sample_idx)


def force_full_init(func: Callable):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'full_init') and not getattr(self, '_fully_initialized', False):
            self.full_init()
        return func(self, *args, **kwargs)

    return wrapper
