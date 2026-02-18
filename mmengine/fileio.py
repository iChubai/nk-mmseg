"""Minimal file I/O helpers."""

from __future__ import annotations

import os
import os.path as osp
import json
import pickle
from typing import Generator, Iterable, Optional


def get(path: str, backend_args: Optional[dict] = None) -> bytes:
    del backend_args
    with open(path, 'rb') as f:
        return f.read()


def exists(path: str, backend_args: Optional[dict] = None) -> bool:
    del backend_args
    return osp.exists(path)


def list_dir_or_file(dir_path: str,
                     list_dir: bool = False,
                     suffix: str | tuple[str, ...] | None = None,
                     recursive: bool = False,
                     backend_args: Optional[dict] = None) -> Iterable[str]:
    del backend_args
    if not osp.isdir(dir_path):
        return []

    if recursive:
        for root, dirs, files in os.walk(dir_path):
            rel_root = osp.relpath(root, dir_path)
            if list_dir:
                for d in dirs:
                    yield d if rel_root == '.' else osp.join(rel_root, d)
            else:
                for fn in files:
                    if suffix is None or fn.endswith(suffix):
                        yield fn if rel_root == '.' else osp.join(rel_root, fn)
    else:
        entries = sorted(os.listdir(dir_path))
        for name in entries:
            full = osp.join(dir_path, name)
            if list_dir:
                if osp.isdir(full):
                    yield name
            else:
                if osp.isfile(full):
                    if suffix is None or name.endswith(suffix):
                        yield name


def load(path: str, backend_args: Optional[dict] = None):
    del backend_args
    ext = osp.splitext(path)[-1].lower()
    if ext in ('.json',):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if ext in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    if ext in ('.yml', '.yaml'):
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError('pyyaml is required to load yaml files') from exc
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
