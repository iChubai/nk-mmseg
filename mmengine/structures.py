"""Data structure compatibility layer."""

from __future__ import annotations

import copy

try:
    import jittor as jt
    _JT_VAR = (jt.Var, )
except Exception:  # pragma: no cover
    jt = None
    _JT_VAR = ()


def _apply_value(value, fn):
    if isinstance(value, BaseDataElement):
        return value.apply(fn)
    if isinstance(value, dict):
        return {k: _apply_value(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_apply_value(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_apply_value(v, fn) for v in value)
    return fn(value)


class BaseDataElement(dict):

    def __init__(self, metainfo=None, **kwargs):
        super().__init__()
        super().__setattr__('_metainfo', {})
        if metainfo is not None:
            self.set_metainfo(metainfo)
        if kwargs:
            self.set_data(kwargs)

    def __getattr__(self, name):
        if name == 'metainfo':
            return self._metainfo
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name == 'metainfo':
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
            return None
        raise AttributeError(name)

    def set_data(self, data: dict):
        for k, v in data.items():
            self[k] = v
        return self

    def set_metainfo(self, meta: dict):
        if meta:
            self._metainfo.update(meta)
        return self

    @property
    def metainfo(self):
        return self._metainfo

    def get(self, key, default=None):
        if key == 'metainfo':
            return self._metainfo
        return super().get(key, default)

    def new(self, metainfo=None, **kwargs):
        inst = self.__class__()
        inst.set_metainfo(copy.deepcopy(self._metainfo))
        if metainfo:
            inst.set_metainfo(metainfo)
        if kwargs:
            inst.set_data(kwargs)
        return inst

    def clone(self):
        return self.new(**{k: copy.deepcopy(v) for k, v in self.items()})

    def apply(self, fn):
        out = self.__class__()
        out.set_metainfo(copy.deepcopy(self._metainfo))
        for k, v in self.items():
            out[k] = _apply_value(v, fn)
        return out

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def cpu(self):
        return self

    def numpy(self):
        def _to_numpy(x):
            if _JT_VAR and isinstance(x, _JT_VAR):
                return x.numpy()
            return x

        return self.apply(_to_numpy)


class InstanceData(BaseDataElement):

    def __len__(self):
        if not self:
            return 0
        for value in self.values():
            if hasattr(value, '__len__'):
                try:
                    return len(value)
                except Exception:
                    continue
        return 0


class PixelData(BaseDataElement):
    pass
