"""Lightweight mmengine compatibility layer for pure Jittor runtime."""

from .config import Config, ConfigDict
from .dataset import BaseDataset, Compose, ConcatDataset, force_full_init
from .dist import all_reduce, is_main_process, master_only
from .evaluator import BaseMetric
from .hooks import (CheckpointHook, DistSamplerSeedHook, Hook, IterTimerHook,
                    LoggerHook, ParamSchedulerHook)
from .logging import MMLogger, print_log
from .runner import Runner
from .utils import (get_git_hash, is_str, is_tuple_of, list_from_file,
                    mkdir_or_exist)


class DefaultScope:
    _instances = {}
    _current = None

    def __init__(self, instance_name: str, scope_name: str = 'mmengine'):
        self.instance_name = instance_name
        self.scope_name = scope_name

    @classmethod
    def get_instance(cls, instance_name: str, scope_name: str = 'mmengine'):
        inst = cls(instance_name, scope_name=scope_name)
        cls._instances[instance_name] = inst
        cls._current = inst
        return inst

    @classmethod
    def get_current_instance(cls):
        return cls._current

    @classmethod
    def check_instance_created(cls, instance_name: str):
        return instance_name in cls._instances


__all__ = [
    'Config',
    'ConfigDict',
    'BaseDataset',
    'Compose',
    'ConcatDataset',
    'force_full_init',
    'BaseMetric',
    'Runner',
    'Hook',
    'CheckpointHook',
    'DistSamplerSeedHook',
    'IterTimerHook',
    'LoggerHook',
    'ParamSchedulerHook',
    'MMLogger',
    'print_log',
    'is_main_process',
    'master_only',
    'all_reduce',
    'mkdir_or_exist',
    'is_str',
    'is_tuple_of',
    'list_from_file',
    'get_git_hash',
    'DefaultScope',
]
