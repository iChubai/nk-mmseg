"""
Jittor utilities for DFormer implementation.

This version supports loading both native Jittor checkpoints and PyTorch-style
`.pth` zip checkpoints without requiring the torch package at runtime.
"""

import argparse
import io
import os
import pickle
import random
import time
import zipfile
from collections import OrderedDict

import jittor as jt
import numpy as np

_RUNTIME_CHECK_DONE = False


def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def link_file(src, target):
    """Create symbolic link."""
    if os.path.islink(target) or os.path.exists(target):
        os.remove(target)
    os.symlink(src, target)


def extant_file(x):
    """Check if file exists."""
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError(f'{x} does not exist')
    return x


def parse_devices(input_devices):
    """Parse device string."""
    if input_devices.endswith('*'):
        return list(range(jt.flags.use_cuda))
    return [int(d) for d in input_devices.split(',')]


def _parse_major_version(version_str):
    try:
        return int(str(version_str).split('.')[0])
    except Exception:
        return -1


def check_runtime_compatibility(raise_on_error=True):
    """Validate critical runtime constraints for stable migration behavior."""
    global _RUNTIME_CHECK_DONE
    if _RUNTIME_CHECK_DONE:
        return True

    np_major = _parse_major_version(np.__version__)
    allow_numpy2 = os.environ.get('NKMMSEG_ALLOW_NUMPY2', '0') == '1'
    if np_major >= 2 and not allow_numpy2:
        msg = (
            'Incompatible runtime detected: numpy>=2 can corrupt Jittor tensor '
            f'conversion in this environment (numpy={np.__version__}, '
            f'jittor={getattr(jt, "__version__", "unknown")}). '
            'Please use numpy<2 (recommended 1.26.4), or set '
            'NKMMSEG_ALLOW_NUMPY2=1 to bypass this guard.'
        )
        if raise_on_error:
            raise RuntimeError(msg)
        print(f'Warning: {msg}')
        return False

    _RUNTIME_CHECK_DONE = True
    return True


def _default_param_mapping():
    """Explicit PyTorch->Jittor key mapping."""
    return {
        'decode_head.conv_seg.weight': 'decode_head.cls_seg.weight',
        'decode_head.conv_seg.bias': 'decode_head.cls_seg.bias',
        'decode_head.squeeze.bn.weight': 'decode_head.squeeze.norm.weight',
        'decode_head.squeeze.bn.bias': 'decode_head.squeeze.norm.bias',
        'decode_head.squeeze.bn.running_mean':
        'decode_head.squeeze.norm.running_mean',
        'decode_head.squeeze.bn.running_var':
        'decode_head.squeeze.norm.running_var',
        'decode_head.hamburger.ham_out.bn.weight':
        'decode_head.hamburger.ham_out.norm.weight',
        'decode_head.hamburger.ham_out.bn.bias':
        'decode_head.hamburger.ham_out.norm.bias',
        'decode_head.hamburger.ham_out.bn.running_mean':
        'decode_head.hamburger.ham_out.norm.running_mean',
        'decode_head.hamburger.ham_out.bn.running_var':
        'decode_head.hamburger.ham_out.norm.running_var',
        'decode_head.align.bn.weight': 'decode_head.align.norm.weight',
        'decode_head.align.bn.bias': 'decode_head.align.norm.bias',
        'decode_head.align.bn.running_mean':
        'decode_head.align.norm.running_mean',
        'decode_head.align.bn.running_var': 'decode_head.align.norm.running_var',
    }


def _convert_param_name(src_key, jittor_state_dict):
    """Convert checkpoint key to model key."""
    if 'num_batches_tracked' in src_key:
        return None
    if any(x in src_key
           for x in ['backbone.norm0', 'backbone.norm1', 'backbone.norm2',
                     'backbone.norm3']):
        return None

    mapping = _default_param_mapping()
    base_keys = [src_key]
    if src_key.startswith('module.'):
        base_keys.append(src_key[7:])
    if src_key.startswith('backbone.'):
        base_keys.append(src_key[9:])
    if src_key.startswith('module.backbone.'):
        base_keys.append(src_key[16:])
    if src_key.startswith('model.'):
        base_keys.append(src_key[6:])
    if src_key.startswith('model.backbone.'):
        base_keys.append(src_key[15:])

    candidates = []
    for key in base_keys:
        if key in mapping:
            candidates.append(mapping[key])
        candidates.append(key)
        if 'gamma_1' in key:
            candidates.append(key.replace('gamma_1', 'gamma1'))
        if 'gamma_2' in key:
            candidates.append(key.replace('gamma_2', 'gamma2'))
        if '.bn.' in key:
            candidates.append(key.replace('.bn.', '.norm.'))

    dedup = []
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)

    for key in dedup:
        if key in jittor_state_dict:
            return key
    return None


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, jt.Var):
        return value.numpy()
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.array(value)


def _to_jt_var(np_val, ref_var=None):
    """Create a Jittor Var from numpy with safe copy and dtype alignment."""
    arr = np.array(np_val, copy=True, order='C')

    target_dtype = None
    if ref_var is not None and hasattr(ref_var, 'dtype'):
        target_dtype = str(ref_var.dtype)

    if target_dtype in ('float32', 'float'):
        arr = arr.astype(np.float32, copy=False)
        return jt.float32(arr.copy())
    if target_dtype == 'float64':
        arr = arr.astype(np.float64, copy=False)
        return jt.float64(arr.copy())
    if target_dtype in ('float16', 'half'):
        arr = arr.astype(np.float16, copy=False)
        return jt.float16(arr.copy())
    if target_dtype == 'int64':
        arr = arr.astype(np.int64, copy=False)
        return jt.int64(arr.copy())
    if target_dtype == 'int32':
        arr = arr.astype(np.int32, copy=False)
        return jt.int32(arr.copy())
    if target_dtype == 'int16':
        arr = arr.astype(np.int16, copy=False)
        return jt.int16(arr.copy())
    if target_dtype == 'int8':
        arr = arr.astype(np.int8, copy=False)
        return jt.int8(arr.copy())
    if target_dtype == 'uint8':
        arr = arr.astype(np.uint8, copy=False)
        return jt.uint8(arr.copy())
    if target_dtype == 'bool':
        arr = arr.astype(np.bool_, copy=False)
        return jt.bool(arr.copy())

    # Fallback path.
    return jt.array(arr.copy())


def _load_state_dict_arrays(model, state_dict, verbose=True, return_stats=False):
    """Load checkpoint arrays into model with key conversion."""
    model_state = model.state_dict()
    converted_count = 0
    skipped_count = 0
    shape_mismatch = 0
    mapped_dst_keys = set()
    loaded_dst_keys = set()
    unmapped_src_keys = []
    shape_mismatch_keys = []
    failed_assign_keys = []

    for src_key, src_val in state_dict.items():
        dst_key = _convert_param_name(src_key, model_state)
        if dst_key is None:
            skipped_count += 1
            unmapped_src_keys.append(src_key)
            continue

        mapped_dst_keys.add(dst_key)
        np_val = _to_numpy(src_val)
        expected_shape = tuple(model_state[dst_key].shape)
        if tuple(np_val.shape) != expected_shape:
            shape_mismatch += 1
            skipped_count += 1
            shape_mismatch_keys.append((src_key, dst_key))
            continue

        try:
            target_var = model_state[dst_key]
            target_var.update(_to_jt_var(np_val, target_var))
            target_var.sync(False, False)
        except Exception as e:
            skipped_count += 1
            failed_assign_keys.append((src_key, dst_key, str(e)))
            continue
        converted_count += 1
        loaded_dst_keys.add(dst_key)

    jt.sync_all()
    stats = {
        'total_src': len(state_dict),
        'loaded_count': converted_count,
        'skipped_count': skipped_count,
        'shape_mismatch_count': shape_mismatch,
        'mapped_dst_keys': mapped_dst_keys,
        'loaded_dst_keys': loaded_dst_keys,
        'unmapped_src_keys': unmapped_src_keys,
        'shape_mismatch_keys': shape_mismatch_keys,
        'failed_assign_keys': failed_assign_keys,
    }

    if verbose:
        print(
            f'Loaded {converted_count} params, skipped {skipped_count}, '
            f'shape_mismatch {shape_mismatch}.')
    if return_stats:
        return model, stats
    return model


_TORCH_STORAGE_DTYPE = {
    'HalfStorage': np.float16,
    'BFloat16Storage': np.uint16,
    'FloatStorage': np.float32,
    'DoubleStorage': np.float64,
    'LongStorage': np.int64,
    'IntStorage': np.int32,
    'ShortStorage': np.int16,
    'CharStorage': np.int8,
    'ByteStorage': np.uint8,
    'BoolStorage': np.bool_,
}


class _TorchStorageType:
    def __init__(self, name):
        if name not in _TORCH_STORAGE_DTYPE:
            raise KeyError(name)
        self.dtype = _TORCH_STORAGE_DTYPE[name]


class _TorchDummy:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __call__(self, *args, **kwargs):
        return self


def _torch_rebuild_tensor_v2(storage, storage_offset, size, stride,
                             requires_grad, backward_hooks):
    if len(size) == 0:
        return np.array(storage[storage_offset:storage_offset + 1])[0]

    numel = int(np.prod(size))
    base = storage[storage_offset:storage_offset + numel]

    # Fast path for contiguous tensors.
    contiguous_stride = []
    step = 1
    for s in reversed(size):
        contiguous_stride.append(step)
        step *= s
    contiguous_stride = tuple(reversed(contiguous_stride))
    if stride is None or tuple(stride) == contiguous_stride:
        return np.array(base, copy=True).reshape(size)

    try:
        view = np.lib.stride_tricks.as_strided(
            storage[storage_offset:],
            shape=tuple(size),
            strides=tuple(int(s) * storage.dtype.itemsize for s in stride))
        return np.array(view, copy=True)
    except Exception:
        return np.array(base, copy=True).reshape(size)


def _torch_rebuild_tensor(storage, storage_offset, size, stride):
    return _torch_rebuild_tensor_v2(storage, storage_offset, size, stride, None,
                                    None)


def _torch_rebuild_parameter(data, requires_grad, backward_hooks):
    return data


def _torch_device(*args, **kwargs):
    return str(args[0]) if args else 'cpu'


class _TorchZipUnpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if isinstance(name, str) and 'Storage' in name:
            return _TorchStorageType(name)
        if name == '_rebuild_tensor_v2':
            return _torch_rebuild_tensor_v2
        if name == '_rebuild_tensor':
            return _torch_rebuild_tensor
        if name == '_rebuild_parameter':
            return _torch_rebuild_parameter
        if mod_name == 'collections' and name == 'OrderedDict':
            return OrderedDict
        if mod_name == 'torch' and name == 'device':
            return _torch_device
        if mod_name.startswith('torch'):
            return _TorchDummy
        return super().find_class(mod_name, name)


def _load_torch_zip_checkpoint(path):
    """Load torch.save zip checkpoint without importing torch."""
    with zipfile.ZipFile(path, 'r') as zf:
        data_pkl_candidates = [n for n in zf.namelist() if n.endswith('data.pkl')]
        if not data_pkl_candidates:
            raise RuntimeError(f'Unsupported torch checkpoint format: {path}')
        data_pkl_name = data_pkl_candidates[0]
        prefix = data_pkl_name[:-8]
        storages = {}

        def persistent_load(saved_id):
            typename = saved_id[0]
            if isinstance(typename, bytes):
                typename = typename.decode('ascii')
            if typename != 'storage':
                raise RuntimeError(
                    f'Unknown persistent typename: {typename} in {path}')
            storage_type, key, location, numel = saved_id[1:]
            if key not in storages:
                data_name = f'{prefix}data/{key}'
                raw = zf.read(data_name)
                dtype = storage_type.dtype
                storages[key] = np.frombuffer(raw, dtype=dtype).copy()
            return storages[key]

        data_buf = io.BytesIO(zf.read(data_pkl_name))
        unpickler = _TorchZipUnpickler(data_buf)
        unpickler.persistent_load = persistent_load
        return unpickler.load()


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'],
                                                     dict):
            return checkpoint['state_dict']
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            return checkpoint['model']
        if 'state_dict_ema' in checkpoint and isinstance(
                checkpoint['state_dict_ema'], dict):
            return checkpoint['state_dict_ema']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f'Unsupported checkpoint object type: {type(checkpoint)}')


def _load_checkpoint_any(model_file):
    """Load checkpoint object from path with Jittor first, torch-zip fallback."""
    if not isinstance(model_file, str):
        return model_file, 'in-memory'

    suffix = model_file.lower()
    if suffix.endswith(('.pth', '.pt', '.pth.tar', '.bin')):
        try:
            return _load_torch_zip_checkpoint(model_file), 'torch-zip'
        except Exception as zip_err:
            try:
                return jt.load(model_file), 'jittor'
            except Exception as jt_err:
                raise RuntimeError(
                    f'Failed to load checkpoint: {model_file}\n'
                    f'torch-zip parser error: {zip_err}\n'
                    f'jt.load error: {jt_err}') from jt_err
    try:
        return jt.load(model_file), 'jittor'
    except Exception as jt_err:
        raise RuntimeError(
            f'Failed to load checkpoint with jt.load: {model_file}') from jt_err


def _verbose_key_logs_enabled():
    return os.environ.get('NKMMSEG_LOAD_VERBOSE_KEYS', '0') == '1'


def _print_key_summary(prefix, keys):
    keys = list(keys)
    if not keys:
        return
    if _verbose_key_logs_enabled():
        print(f'{prefix}: {", ".join(sorted(keys)[:50])}')
    else:
        print(f'{prefix}: {len(keys)} '
              f'(set NKMMSEG_LOAD_VERBOSE_KEYS=1 to print key names)')


def load_model(model,
               model_file,
               is_restore=False,
               strict=False,
               return_stats=False):
    """Load model from checkpoint file."""
    check_runtime_compatibility(raise_on_error=True)
    t_start = time.time()
    if model_file is None:
        if return_stats:
            return model, {}
        return model

    checkpoint_obj, source = _load_checkpoint_any(model_file)
    state_dict = _extract_state_dict(checkpoint_obj)
    t_ioend = time.time()

    if is_restore:
        restored = OrderedDict()
        for key, val in state_dict.items():
            # Resume checkpoints may already contain "module." prefixed keys.
            # Avoid creating "module.module.*" entries.
            norm_key = key if str(key).startswith('module.') else f'module.{key}'
            restored[norm_key] = val
        state_dict = restored

    model, load_stats = _load_state_dict_arrays(
        model, state_dict, verbose=True, return_stats=True)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    direct_missing_keys = own_keys - ckpt_keys
    direct_unexpected_keys = ckpt_keys - own_keys
    # For resume flows we may intentionally prefix keys with "module."; the
    # pre-mapping key diff is expected and not actionable.
    if not is_restore:
        _print_key_summary('Missing key(s) in checkpoint(before mapping)',
                           direct_missing_keys)
        _print_key_summary('Unexpected key(s) in checkpoint(before mapping)',
                           direct_unexpected_keys)

    missing_after_mapping = own_keys - load_stats['loaded_dst_keys']
    _print_key_summary('Missing key(s) after mapping/load', missing_after_mapping)
    _print_key_summary('Unmapped checkpoint key(s)', load_stats['unmapped_src_keys'])
    if load_stats['shape_mismatch_keys']:
        if _verbose_key_logs_enabled():
            brief = [f'{s}->{d}' for s, d in load_stats['shape_mismatch_keys'][:20]]
            print('Shape mismatch key(s): ' + ', '.join(brief))
        else:
            print('Shape mismatch key(s): '
                  f'{len(load_stats["shape_mismatch_keys"])}')
    if load_stats['failed_assign_keys']:
        if _verbose_key_logs_enabled():
            brief = [f'{s}->{d}' for s, d, _ in load_stats['failed_assign_keys'][:20]]
            print('Failed assign key(s): ' + ', '.join(brief))
        else:
            print('Failed assign key(s): '
                  f'{len(load_stats["failed_assign_keys"])}')

    if strict:
        strict_errors = []
        if missing_after_mapping:
            strict_errors.append(
                f'missing_after_mapping={len(missing_after_mapping)}')
        if load_stats['shape_mismatch_count'] > 0:
            strict_errors.append(
                f'shape_mismatch={load_stats["shape_mismatch_count"]}')
        if len(load_stats['failed_assign_keys']) > 0:
            strict_errors.append(
                f'failed_assign={len(load_stats["failed_assign_keys"])}')
        if strict_errors:
            raise RuntimeError(
                'Strict checkpoint load failed: ' + ', '.join(strict_errors))

    t_end = time.time()
    print(
        f'Load model from {source}, Time usage:\n\tIO: {t_ioend - t_start}, '
        f'initialize parameters: {t_end - t_ioend}')

    summary = {
        'source': source,
        'direct_missing_keys': direct_missing_keys,
        'direct_unexpected_keys': direct_unexpected_keys,
        'missing_after_mapping': missing_after_mapping,
        'load_stats': load_stats,
        'io_time': t_ioend - t_start,
        'assign_time': t_end - t_ioend,
    }
    if return_stats:
        return model, summary
    return model


def save_model(model, model_file):
    """Save model to checkpoint file."""
    ensure_dir(os.path.dirname(model_file))
    jt.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')


def all_reduce_tensor(tensor, op=None, world_size=1):
    """All reduce tensor across processes."""
    if jt.world_size > 1:
        return jt.distributed.all_reduce(tensor)
    return tensor


def reduce_tensor(tensor):
    """Reduce tensor across processes."""
    return tensor


def get_world_size():
    return 1


def get_rank():
    return 0


def is_main_process():
    return True


def synchronize():
    pass


def time_synchronized():
    jt.sync_all()
    return time.time()


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Timer utility."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self, average=True):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    out = ''
    i = 1
    if days > 0:
        out += f'{days}D'
        i += 1
    if hours > 0 and i <= 2:
        out += f'{hours}h'
        i += 1
    if minutes > 0 and i <= 2:
        out += f'{minutes}m'
        i += 1
    if secondsf > 0 and i <= 2:
        out += f'{secondsf}s'
        i += 1
    if millis > 0 and i <= 2:
        out += f'{millis}ms'
        i += 1
    if out == '':
        out = '0ms'
    return out
