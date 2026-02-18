"""Utility functions for mmengine compatibility."""

from __future__ import annotations

import hashlib
import os
import os.path as osp
import subprocess
import sys
from typing import Iterable


def mkdir_or_exist(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def is_str(x):
    return isinstance(x, str)


def is_tuple_of(seq, expected_type) -> bool:
    return isinstance(seq, tuple) and all(isinstance(x, expected_type) for x in seq)


def is_list_of(seq, expected_type) -> bool:
    return isinstance(seq, list) and all(isinstance(x, expected_type) for x in seq)


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def list_from_file(filename: str, encoding='utf-8', backend_args=None):
    del backend_args
    with open(filename, 'r', encoding=encoding) as f:
        return [line.rstrip('\n') for line in f]


def get_git_hash(fallback='unknown'):
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return fallback


def collect_env():
    return {
        'python': sys.version,
    }
