"""Distributed helpers (Jittor-aware, single-process friendly)."""

from __future__ import annotations

import functools

import jittor as jt


def is_main_process() -> bool:
    return (not jt.in_mpi) or jt.rank == 0


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return wrapper


def all_reduce(value, op='mean'):
    if not jt.in_mpi:
        return value
    if hasattr(value, 'mpi_all_reduce'):
        return value.mpi_all_reduce(op)
    arr = jt.array(value)
    return arr.mpi_all_reduce(op)


def get_dist_info():
    if not jt.in_mpi:
        return 0, 1
    return int(jt.rank), int(jt.world_size)


def barrier():
    if jt.in_mpi and hasattr(jt, 'sync_all'):
        jt.sync_all()
