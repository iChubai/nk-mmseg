"""Model compatibility layer."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, List, Sequence

import jittor as jt
import jittor.nn as nn

from .weight_init import (bias_init_with_prob, caffe2_xavier_init,
                          constant_init, kaiming_init, normal_init,
                          trunc_normal_, trunc_normal_init, uniform_init,
                          xavier_init)


def _normalize_index(index: int, length: int) -> int:
    idx = int(index)
    if idx < 0:
        idx += int(length)
    if idx < 0 or idx >= int(length):
        raise IndexError(f'index {index} is out of range')
    return idx


def _patch_jittor_container_setitem() -> None:
    # Some mmseg modules mutate ModuleList/Sequential by index during init.
    # Native Jittor containers expose __getitem__ but not __setitem__.
    if not hasattr(nn.ModuleList, '__setitem__'):

        def _module_list_setitem(self, index, module):
            idx = _normalize_index(index, len(self.layers))
            self.layers[idx] = module
            setattr(self, str(idx), module)

        nn.ModuleList.__setitem__ = _module_list_setitem

    if not hasattr(nn.Sequential, '__setitem__'):

        def _sequential_setitem(self, index, module):
            idx = _normalize_index(index, len(self.layers))
            self.layers[idx] = module
            setattr(self, str(idx), module)

        nn.Sequential.__setitem__ = _sequential_setitem


_patch_jittor_container_setitem()


def _patch_jittor_activation_fns() -> None:
    # Jittor ReLU/LeakyReLU modules may pass inplace=... into functional ops,
    # while some Jittor builds expose relu/leaky_relu without inplace kwarg.
    relu_fn = getattr(nn, 'relu', None)
    if callable(relu_fn):

        def _relu_compat(x, *args, **kwargs):
            kwargs.pop('inplace', None)
            return relu_fn(x, *args, **kwargs)

        nn.relu = _relu_compat

    jt_relu_fn = getattr(jt, 'relu', None)
    if callable(jt_relu_fn):

        def _jt_relu_compat(x, *args, **kwargs):
            kwargs.pop('inplace', None)
            return jt_relu_fn(x, *args, **kwargs)

        jt.relu = _jt_relu_compat

    lrelu_fn = getattr(nn, 'leaky_relu', None)
    if callable(lrelu_fn):

        def _lrelu_compat(x, *args, **kwargs):
            kwargs.pop('inplace', None)
            return lrelu_fn(x, *args, **kwargs)

        nn.leaky_relu = _lrelu_compat

    jt_lrelu_fn = getattr(jt, 'leaky_relu', None)
    if callable(jt_lrelu_fn):

        def _jt_lrelu_compat(x, *args, **kwargs):
            kwargs.pop('inplace', None)
            return jt_lrelu_fn(x, *args, **kwargs)

        jt.leaky_relu = _jt_lrelu_compat

    relu6_fn = getattr(nn, 'relu6', None)
    if callable(relu6_fn):

        def _relu6_compat(x, *args, **kwargs):
            kwargs.pop('inplace', None)
            return relu6_fn(x, *args, **kwargs)

        nn.relu6 = _relu6_compat

    relu_cls = getattr(nn, 'ReLU', None)
    if relu_cls is not None and hasattr(relu_cls, '__init__'):
        relu_init = relu_cls.__init__

        def _relu_init_compat(self, *args, **kwargs):
            kwargs.pop('inplace', None)
            relu_init(self, *args, **kwargs)

        relu_cls.__init__ = _relu_init_compat

    lrelu_cls = getattr(nn, 'LeakyReLU', None)
    if lrelu_cls is not None and hasattr(lrelu_cls, '__init__'):
        lrelu_init = lrelu_cls.__init__

        def _lrelu_init_compat(self, *args, **kwargs):
            kwargs.pop('inplace', None)
            lrelu_init(self, *args, **kwargs)

        lrelu_cls.__init__ = _lrelu_init_compat

    relu6_cls = getattr(nn, 'ReLU6', None)
    if relu6_cls is not None and hasattr(relu6_cls, '__init__'):
        relu6_init = relu6_cls.__init__

        def _relu6_init_compat(self, *args, **kwargs):
            kwargs.pop('inplace', None)
            relu6_init(self, *args, **kwargs)

        relu6_cls.__init__ = _relu6_init_compat


_patch_jittor_activation_fns()


ModuleList = nn.ModuleList
Sequential = nn.Sequential


if hasattr(nn, 'ModuleDict'):
    ModuleDict = nn.ModuleDict
else:

    class ModuleDict(nn.Module):

        def __init__(self, modules=None):
            super().__init__()
            self._module_names = []
            if modules is not None:
                self.update(modules)

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, module):
            if key not in self._module_names:
                self._module_names.append(key)
            setattr(self, key, module)

        def __contains__(self, key):
            return key in self._module_names

        def keys(self):
            return list(self._module_names)

        def values(self):
            return [getattr(self, k) for k in self._module_names]

        def items(self):
            return [(k, getattr(self, k)) for k in self._module_names]

        def update(self, modules):
            if hasattr(modules, 'items'):
                iterable = modules.items()
            else:
                iterable = modules
            for k, v in iterable:
                self[k] = v


def _iter_init_cfg(init_cfg) -> List[dict]:
    if init_cfg is None:
        return []
    if isinstance(init_cfg, (list, tuple)):
        return [cfg for cfg in init_cfg if isinstance(cfg, dict)]
    if isinstance(init_cfg, dict):
        return [init_cfg]
    return []


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _module_layer_aliases(module: nn.Module) -> set[str]:
    names = {
        cls.__name__
        for cls in type(module).mro()
        if cls.__name__ not in ('object', )
    }

    if 'Conv' in names:
        names.update({'Conv2d'})
    if 'ConvTranspose' in names:
        names.update({'ConvTranspose2d'})
    if 'BatchNorm' in names:
        names.update(
            {'BatchNorm', '_BatchNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d'})
    if 'LayerNorm' in names:
        names.update({'LayerNorm'})
    if 'GroupNorm' in names:
        names.update({'GroupNorm'})
    if 'Linear' in names:
        names.update({'Linear'})
    if 'PReLU' in names:
        names.update({'PReLU'})

    normalized = set(names)
    normalized.update({name.lstrip('_') for name in names})
    return normalized


def _match_layer(module: nn.Module, layer: Sequence[str] | str | None) -> bool:
    layer_names = _as_list(layer)
    if not layer_names:
        return True
    aliases = _module_layer_aliases(module)
    for name in layer_names:
        if name in aliases or name.lstrip('_') in aliases:
            return True
    return False


def _apply_single_initializer(module: nn.Module, cfg: dict) -> bool:
    init_type = str(cfg.get('type', '')).lower()
    if init_type == 'constant':
        constant_init(module, val=cfg.get('val', 0), bias=cfg.get('bias', 0))
        return True
    if init_type == 'normal':
        normal_init(
            module,
            mean=cfg.get('mean', 0),
            std=cfg.get('std', 1),
            bias=cfg.get('bias', 0))
        return True
    if init_type == 'uniform':
        uniform_init(
            module,
            a=cfg.get('a', 0),
            b=cfg.get('b', 1),
            bias=cfg.get('bias', 0))
        return True
    if init_type == 'xavier':
        xavier_init(
            module,
            gain=cfg.get('gain', 1),
            bias=cfg.get('bias', 0),
            distribution=cfg.get('distribution', 'normal'))
        return True
    if init_type == 'kaiming':
        kaiming_init(
            module,
            a=cfg.get('a', 0),
            mode=cfg.get('mode', 'fan_out'),
            nonlinearity=cfg.get('nonlinearity', 'relu'),
            bias=cfg.get('bias', 0),
            distribution=cfg.get('distribution', 'normal'))
        return True
    if init_type in ('truncnormal', 'trunc_normal'):
        trunc_normal_init(
            module,
            mean=cfg.get('mean', 0),
            std=cfg.get('std', 1),
            a=cfg.get('a', -2),
            b=cfg.get('b', 2),
            bias=cfg.get('bias', 0))
        return True
    if init_type in ('caffe2xavier', 'caffe2_xavier'):
        caffe2_xavier_init(module, bias=cfg.get('bias', 0))
        return True
    return False


def _try_load_pretrained(module: nn.Module, cfg: dict) -> bool:
    init_type = str(cfg.get('type', '')).lower()
    if init_type not in ('pretrained', 'pretrained_part'):
        return False

    checkpoint = cfg.get('checkpoint')
    if not checkpoint:
        return True

    strict = bool(cfg.get('strict', False))
    map_location = cfg.get('map_location', 'cpu')
    prefix = cfg.get('prefix')

    from mmengine.runner.checkpoint import (_extract_state_dict, _load_checkpoint,
                                            load_state_dict)

    ckpt = _load_checkpoint(checkpoint, map_location=map_location, logger=None)

    if prefix is None:
        load_state_dict(module, ckpt, strict=strict, logger=None)
        return True

    state_dict = _extract_state_dict(ckpt)
    if not isinstance(state_dict, dict):
        return True

    prefix = str(prefix)
    filtered = {}
    for key, value in state_dict.items():
        key_str = str(key)
        if key_str.startswith(prefix):
            filtered[key_str[len(prefix):]] = value

    if not filtered and not strict:
        return True
    load_state_dict(module, filtered, strict=strict, logger=None)
    return True


def _apply_init_cfg(module: nn.Module, cfg: dict) -> None:
    if _try_load_pretrained(module, cfg):
        return

    layer = cfg.get('layer')
    for _, child in module.named_modules():
        if _match_layer(child, layer):
            _apply_single_initializer(child, cfg)

    override = cfg.get('override')
    if override is None:
        return
    override_list = override if isinstance(override, (list, tuple)) else [override]
    named_modules = dict(module.named_modules())

    for override_cfg in override_list:
        if not isinstance(override_cfg, dict):
            continue
        target_name = override_cfg.get('name')
        if not target_name or target_name not in named_modules:
            continue

        target_cfg = copy.deepcopy(cfg)
        target_cfg.pop('override', None)
        target_cfg.pop('layer', None)
        target_cfg.update(override_cfg)
        target_cfg.pop('name', None)

        target_module = named_modules[target_name]
        target_layer = target_cfg.pop('layer', None)
        if _match_layer(target_module, target_layer):
            _apply_single_initializer(target_module, target_cfg)


def _can_call_init_weights_no_args(init_fn) -> bool:
    try:
        sig = inspect.signature(init_fn)
    except Exception:
        return True
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect._empty:
            return False
    return True


class BaseModule(nn.Module):

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.init_cfg = init_cfg
        self._is_init = bool(getattr(self, '_is_init', False))

    @property
    def is_init(self):
        return bool(getattr(self, '_is_init', False))

    def init_weights(self):
        if self.is_init:
            return

        for cfg in _iter_init_cfg(self.init_cfg):
            _apply_init_cfg(self, cfg)

        for child in self.children():
            child_init = getattr(child, 'init_weights', None)
            if child_init is None or not callable(child_init):
                continue
            if getattr(child, '_is_init', False):
                continue
            if not _can_call_init_weights_no_args(child_init):
                continue
            child_init()

        self._is_init = True

    def execute(self, *args, **kwargs):
        # Upstream mmseg/mmengine modules are mostly torch-style and define
        # `forward` but not Jittor `execute`.
        if hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement execute/forward.')


class BaseDataPreprocessor(BaseModule):

    def cast_data(self, data):
        return data

    def forward(self, data: Dict[str, Any], training: bool = False):
        del training
        return data


class BaseModel(BaseModule):

    def __init__(self, data_preprocessor=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.data_preprocessor = data_preprocessor

    def parse_losses(self, losses):
        log_vars = {}
        total_loss = 0.0
        for name, value in losses.items():
            if isinstance(value, (list, tuple)):
                v = sum([x.mean() for x in value])
            elif hasattr(value, 'mean'):
                v = value.mean()
            else:
                v = jt.array(float(value))
            log_vars[name] = v
            if 'loss' in name:
                total_loss = total_loss + v
        log_vars['loss'] = total_loss
        return total_loss, log_vars


class BaseTTAModel(BaseModel):

    def __init__(self, module=None, **kwargs):
        super().__init__(**kwargs)
        self.module = module

    def forward(self, *args, **kwargs):
        if self.module is None:
            raise RuntimeError('BaseTTAModel.module is not set')
        if hasattr(self.module, 'forward'):
            return self.module.forward(*args, **kwargs)
        return self.module(*args, **kwargs)

    def merge_preds(self, data_samples_list):
        raise NotImplementedError


def revert_sync_batchnorm(model):
    return model


__all__ = [
    'BaseModule',
    'BaseDataPreprocessor',
    'BaseModel',
    'BaseTTAModel',
    'ModuleList',
    'ModuleDict',
    'Sequential',
    'revert_sync_batchnorm',
    'normal_init',
    'trunc_normal_init',
    'trunc_normal_',
    'uniform_init',
    'constant_init',
    'kaiming_init',
    'caffe2_xavier_init',
    'xavier_init',
    'bias_init_with_prob',
]
