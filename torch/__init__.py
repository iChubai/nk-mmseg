"""Torch compatibility shim backed by Jittor.

This shim is intentionally minimal and only covers APIs used by this repo.
"""

from __future__ import annotations

import contextlib
import builtins as _builtins
from types import SimpleNamespace

import jittor as jt

from . import nn
from . import _utils as _utils  # ensure torch._utils is importable for jt.save compatibility

Tensor = jt.Var
LongTensor = jt.Var
FloatTensor = jt.Var
Size = tuple

float16 = jt.float16
float32 = jt.float32
float64 = jt.float64
float = jt.float32
int64 = jt.int64
long = jt.int64
int32 = jt.int32
uint8 = jt.uint8
bool = jt.bool


if not hasattr(jt.Var, 'T'):

    @property
    def _var_T(self):
        dims = list(range(len(self.shape)))
        dims.reverse()
        return self.permute(*dims)

    jt.Var.T = _var_T


if not hasattr(jt.Var, 'device'):

    @property
    def _var_device(self):
        return device('cuda' if jt.has_cuda else 'cpu')

    jt.Var.device = _var_device


if not hasattr(jt.Var, 'is_cuda'):

    @property
    def _var_is_cuda(self):
        return _builtins.bool(getattr(jt.flags, 'use_cuda', 0))

    jt.Var.is_cuda = _var_is_cuda


if not hasattr(jt.Var, 'new_tensor'):

    def _var_new_tensor(self, data, dtype=None, device=None):
        del device
        out = jt.array(data)
        if dtype is None:
            dtype = self.dtype
        return out.astype(dtype)

    jt.Var.new_tensor = _var_new_tensor


def _normalize_shape_args(shape_args, size=None):
    if size is not None:
        if len(shape_args) != 0:
            raise TypeError('cannot set both positional shape and size=')
        shape_args = (size, )
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
        return tuple(shape_args[0])
    return tuple(shape_args)


def _var_new_zeros(self, *shape, dtype=None, device=None, size=None):
    del device
    shape = _normalize_shape_args(shape, size=size)
    return jt.zeros(shape, dtype=dtype or self.dtype)


def _var_new_ones(self, *shape, dtype=None, device=None, size=None):
    del device
    shape = _normalize_shape_args(shape, size=size)
    return jt.ones(shape, dtype=dtype or self.dtype)


jt.Var.new_zeros = _var_new_zeros
jt.Var.new_ones = _var_new_ones


def _var_argmax(self, dim=None, keepdim=False, axis=None):
    if axis is not None and dim is None:
        dim = axis
    if dim is None:
        flat = self.reshape((-1, ))
        indices, _ = jt.argmax(flat, dim=0, keepdims=keepdim)
        return indices
    indices, _ = jt.argmax(self, dim=dim, keepdims=keepdim)
    return indices


jt.Var.argmax = _var_argmax


if not hasattr(jt.Var, 'bmm'):

    def _var_bmm(self, other):
        return jt.matmul(self, other)

    jt.Var.bmm = _var_bmm


def is_tensor(x):
    return isinstance(x, jt.Var)


def tensor(data, dtype=None, device=None, requires_grad=False, **kwargs):
    del device, kwargs
    arr = jt.array(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    arr.requires_grad = _builtins.bool(requires_grad)
    return arr


def from_numpy(arr):
    return jt.array(arr)


def zeros(*shape, dtype=None, device=None, size=None, requires_grad=False, **kwargs):
    del device, kwargs
    shape = _normalize_shape_args(shape, size=size)
    out = jt.zeros(shape, dtype=dtype or jt.float32)
    out.requires_grad = _builtins.bool(requires_grad)
    return out


def ones(*shape, dtype=None, device=None, size=None, requires_grad=False, **kwargs):
    del device, kwargs
    shape = _normalize_shape_args(shape, size=size)
    out = jt.ones(shape, dtype=dtype or jt.float32)
    out.requires_grad = _builtins.bool(requires_grad)
    return out


def empty(*shape, dtype=None, device=None, size=None, requires_grad=False, **kwargs):
    del device, kwargs
    shape = _normalize_shape_args(shape, size=size)
    # Jittor does not expose an uninitialized tensor API; randn is sufficient
    # for compatibility consumers that only need shape/dtype allocation.
    out = jt.randn(shape)
    out = out if dtype is None else out.astype(dtype)
    out.requires_grad = _builtins.bool(requires_grad)
    return out


def full(size=None,
         fill_value=0,
         dtype=None,
         device=None,
         requires_grad=False,
         **kwargs):
    del device
    if size is None:
        size = kwargs.get('shape', None)
    if size is None:
        raise TypeError('full() missing required argument: size')
    out = jt.full(size, fill_value, dtype=dtype or jt.float32)
    out.requires_grad = _builtins.bool(requires_grad)
    return out


def rand(*shape, dtype=None, device=None, size=None, requires_grad=False, **kwargs):
    del device, kwargs
    shape = _normalize_shape_args(shape, size=size)
    x = jt.rand(shape)
    if dtype is not None:
        x = x.astype(dtype)
    x.requires_grad = _builtins.bool(requires_grad)
    return x


def randn(*shape, dtype=None, device=None, size=None, requires_grad=False, **kwargs):
    del device, kwargs
    shape = _normalize_shape_args(shape, size=size)
    x = jt.randn(shape)
    if dtype is not None:
        x = x.astype(dtype)
    x.requires_grad = _builtins.bool(requires_grad)
    return x


def arange(*args, **kwargs):
    return jt.arange(*args, **kwargs)


def linspace(start, end, steps=100, dtype=None, device=None):
    del device
    x = jt.linspace(start, end, steps)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def cat(tensors, dim=0):
    return jt.concat(list(tensors), dim=dim)


def stack(tensors, dim=0):
    return jt.stack(list(tensors), dim=dim)


def matmul(a, b):
    return jt.matmul(a, b)


def bmm(a, b):
    return jt.matmul(a, b)


def einsum(expr, *ops):
    return jt.linalg.einsum(expr, *ops)


def sum(x, dim=None, keepdim=False):
    if dim is None:
        return x.sum()
    return x.sum(dim=dim, keepdims=keepdim)


def mean(x, dim=None, keepdim=False):
    if dim is None:
        return x.mean()
    return x.mean(dim=dim, keepdims=keepdim)


def max(x, dim=None, keepdim=False):
    if dim is None:
        return jt.max(x)
    indices, values = jt.argmax(x, dim=dim, keepdims=keepdim)
    return values, indices


def abs(x):
    return jt.abs(x)


def mul(x, y):
    return x * y


def pow(x, y):
    return jt.pow(x, y)


def sqrt(x):
    return jt.sqrt(x)


def exp(x):
    return jt.exp(x)


def log(x):
    return jt.log(x)


def log10(x):
    return jt.log(x) / jt.log(jt.array(10.0))


def unique(x):
    return jt.unique(x)


def where(cond, x, y):
    return jt.where(cond, x, y)


def clamp(input, min=None, max=None):
    if min is None and max is None:
        return input
    if min is None:
        min = -1e30
    if max is None:
        max = 1e30
    return jt.clamp(input, min_v=min, max_v=max)


def clip(input, min=None, max=None):
    return clamp(input, min=min, max=max)


def topk(input, k, dim=-1, largest=True, sorted=True):
    del sorted
    if largest:
        return jt.topk(input, int(k), dim=dim)
    values, indices = jt.topk(-input, int(k), dim=dim)
    return -values, indices


def sort(input, dim=-1, descending=False):
    idx, values = jt.argsort(input, dim=dim, descending=descending)
    return values, idx


def nonzero(input, as_tuple=False):
    out = jt.nonzero(input)
    if not as_tuple:
        return out
    if out.ndim == 1:
        return (out, )
    return tuple([out[:, i] for i in range(out.shape[1])])


def roll(input, shifts, dims=None):
    return jt.roll(input, shifts=shifts, dims=dims)


def dot(input, tensor):
    return (input.reshape((-1, )) * tensor.reshape((-1, ))).sum()


def equal(input, other):
    if tuple(input.shape) != tuple(other.shape):
        return False
    return _builtins.bool(jt.equal(input, other))


def isnan(input):
    return jt.isnan(input)


def finfo(dtype):
    return jt.finfo(dtype)


def flatten(input, start_dim=0, end_dim=-1):
    ndim = len(input.shape)
    if end_dim < 0:
        end_dim = ndim + end_dim
    if start_dim < 0:
        start_dim = ndim + start_dim
    if start_dim == 0 and end_dim == ndim - 1:
        return input.reshape((-1, ))
    pre = list(input.shape[:start_dim])
    mid = 1
    for d in input.shape[start_dim:end_dim + 1]:
        mid *= int(d)
    post = list(input.shape[end_dim + 1:])
    return input.reshape(tuple(pre + [mid] + post))


def meshgrid(tensors, indexing='ij'):
    del indexing
    if isinstance(tensors, (list, tuple)):
        return jt.meshgrid(*list(tensors))
    return jt.meshgrid(tensors)


def zeros_like(x, dtype=None, **kwargs):
    del kwargs
    out = jt.zeros_like(x)
    return out if dtype is None else out.astype(dtype)


def ones_like(x, dtype=None, **kwargs):
    del kwargs
    out = jt.ones_like(x)
    return out if dtype is None else out.astype(dtype)


def sigmoid(x):
    return x.sigmoid()


def no_grad():
    return jt.no_grad()


class device:

    def __init__(self, name='cpu'):
        self.type = name

    def __str__(self):
        return self.type


class _CudaNS:

    @staticmethod
    def is_available():
        return _builtins.bool(jt.has_cuda)

    @staticmethod
    def device_count():
        return 1 if _builtins.bool(jt.has_cuda) else 0

    @staticmethod
    def current_device():
        return 0


cuda = _CudaNS()


from . import jit  # noqa: E402
from . import optim  # noqa: E402

__all__ = [
    'Tensor', 'LongTensor', 'FloatTensor', 'Size', 'tensor', 'from_numpy',
    'zeros', 'ones', 'empty', 'full', 'rand', 'randn', 'arange', 'linspace',
    'cat', 'stack', 'matmul', 'bmm', 'einsum', 'sum', 'mean', 'max', 'abs',
    'mul', 'pow', 'sqrt', 'exp', 'log', 'log10', 'unique', 'where', 'clamp',
    'clip', 'topk', 'sort', 'nonzero', 'roll', 'dot', 'equal', 'isnan',
    'finfo', 'flatten', 'meshgrid', 'zeros_like', 'ones_like', 'sigmoid',
    'no_grad', 'device', 'cuda', 'nn', 'jit', 'optim', 'float16', 'float32',
    'float64', 'float', 'int64', 'long', 'int32', 'uint8', 'bool',
    'is_tensor'
]
