"""Checkpoint loading compatibility."""

from __future__ import annotations

import jittor as jt


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ('state_dict', 'model', 'model_state_dict', 'state_dict_ema'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _shape_of(value):
    shape = getattr(value, 'shape', None)
    if shape is None:
        return None
    try:
        return tuple(shape)
    except Exception:
        return None


def _load_checkpoint_any(filename):
    # Prefer the project-level robust loader (supports torch zip checkpoints
    # without importing torch), fallback to jt.load when unavailable.
    try:
        from utils.jt_utils import _load_checkpoint_any as _external_loader
        checkpoint, _ = _external_loader(filename)
        return checkpoint
    except Exception:
        return jt.load(filename)


def _try_load_state_dict(model, state, strict):
    if hasattr(model, 'load_state_dict'):
        try:
            return model.load_state_dict(state, strict=strict)
        except TypeError:
            # Some wrappers do not support strict argument.
            return model.load_state_dict(state)
    if hasattr(model, 'load_parameters'):
        model.load_parameters(state)
        return None
    raise RuntimeError(f'{type(model).__name__} has no load_state_dict/load_parameters.')


def _load_checkpoint_to_model(model, checkpoint, strict=False):
    state = _extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f'checkpoint state_dict must be dict, got {type(state)}')

    # First try direct loading.
    try:
        _try_load_state_dict(model, state, strict=strict)
        return model
    except Exception:
        pass

    # Common fallback: DataParallel prefix.
    if any(str(k).startswith('module.') for k in state.keys()):
        stripped = {
            (k[7:] if str(k).startswith('module.') else k): v
            for k, v in state.items()
        }
        try:
            _try_load_state_dict(model, stripped, strict=strict)
            return model
        except Exception:
            pass

    # Best-effort partial load by key and shape.
    if not hasattr(model, 'state_dict') or not hasattr(model, 'load_parameters'):
        raise RuntimeError(
            f'Failed to load checkpoint into {type(model).__name__}: '
            'model does not support partial loading.')

    model_state = model.state_dict()
    filtered = {}
    for key, value in state.items():
        dst_key = key
        if dst_key not in model_state and str(dst_key).startswith('module.'):
            dst_key = dst_key[7:]
        if dst_key not in model_state:
            continue
        src_shape = _shape_of(value)
        dst_shape = _shape_of(model_state[dst_key])
        if src_shape is not None and dst_shape is not None and src_shape != dst_shape:
            continue
        filtered[dst_key] = value

    if not filtered:
        raise RuntimeError('No checkpoint parameters matched current model state_dict.')

    model.load_parameters(filtered)

    if strict:
        model_keys = set(model_state.keys())
        load_keys = set(filtered.keys())
        missing = model_keys - load_keys
        unexpected = set(state.keys()) - model_keys
        if missing or unexpected:
            raise RuntimeError(
                f'Strict checkpoint load failed. missing={len(missing)} '
                f'unexpected={len(unexpected)}')
    return model


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    del map_location, logger
    ckpt = _load_checkpoint(filename)
    _load_checkpoint_to_model(model, ckpt, strict=strict)
    return ckpt


def _load_checkpoint(filename, map_location='cpu', logger=None):
    del map_location, logger
    return _load_checkpoint_any(filename)


def load_state_dict(module, state_dict, strict=False, logger=None):
    del logger
    return _load_checkpoint_to_model(module, state_dict, strict=strict)


class CheckpointLoader:

    @staticmethod
    def load_checkpoint(filename, map_location='cpu', logger=None, **kwargs):
        del map_location, logger, kwargs
        return _load_checkpoint(filename)
