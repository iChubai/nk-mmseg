# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .class_names import (ade_classes, ade_palette, bdd100k_classes,
                          bdd100k_palette, cityscapes_classes,
                          cityscapes_palette, cocostuff_classes,
                          cocostuff_palette, dataset_aliases as _dataset_aliases,
                          get_classes as _base_get_classes,
                          get_palette as _base_get_palette, isaid_classes,
                          isaid_palette, loveda_classes, loveda_palette,
                          potsdam_classes, potsdam_palette, stare_classes,
                          stare_palette, synapse_classes, synapse_palette,
                          vaihingen_classes, vaihingen_palette, voc_classes,
                          voc_palette)
# yapf: enable
from .collect_env import collect_env
from .get_templates import get_predefined_templates
from .io import datafrombytes
from .misc import add_prefix, stack_batch
from .set_env import register_all_modules
from .typing_utils import (ConfigType, ForwardResults, MultiConfig,
                           OptConfigType, OptMultiConfig, OptSampleList,
                           SampleList, TensorDict, TensorList)

try:
    from . import tokenizer
    tokenize = tokenizer.tokenize
except Exception:  # pragma: no cover

    class _TokenizerFallback:

        @staticmethod
        def tokenize(*args, **kwargs):
            raise ImportError('tokenizer dependencies are not available')

    tokenizer = _TokenizerFallback()

    def tokenize(*args, **kwargs):
        raise ImportError('tokenizer dependencies are not available')


_extra_dataset_aliases = {
    'nyudepthv2': ['nyudepthv2', 'nyu', 'nyud'],
    'sunrgbd': ['sunrgbd', 'sun'],
}


def _default_palette(num_classes: int):
    palette = []
    for i in range(num_classes):
        palette.append([(37 * i) % 255, (17 * i) % 255, (97 * i) % 255])
    return palette


dataset_aliases = dict(_dataset_aliases)
dataset_aliases.update(_extra_dataset_aliases)


def get_classes(dataset):
    if isinstance(dataset, str):
        name = dataset.lower()
        if name in _extra_dataset_aliases['nyudepthv2']:
            from local_configs._base_.datasets.NYUDepthv2 import C
            return list(C.class_names)
        if name in _extra_dataset_aliases['sunrgbd']:
            from local_configs._base_.datasets.SUNRGBD import C
            return list(C.class_names)
    return _base_get_classes(dataset)


def get_palette(dataset):
    if isinstance(dataset, str):
        name = dataset.lower()
        if name in _extra_dataset_aliases['nyudepthv2']:
            return _default_palette(len(get_classes(dataset)))
        if name in _extra_dataset_aliases['sunrgbd']:
            return _default_palette(len(get_classes(dataset)))
    return _base_get_palette(dataset)


# isort: off
from .mask_classification import MatchMasks, seg_data_to_instance_data

__all__ = [
    'collect_env',
    'register_all_modules',
    'stack_batch',
    'add_prefix',
    'ConfigType',
    'OptConfigType',
    'MultiConfig',
    'OptMultiConfig',
    'SampleList',
    'OptSampleList',
    'TensorDict',
    'TensorList',
    'ForwardResults',
    'cityscapes_classes',
    'ade_classes',
    'voc_classes',
    'cocostuff_classes',
    'loveda_classes',
    'potsdam_classes',
    'vaihingen_classes',
    'isaid_classes',
    'stare_classes',
    'cityscapes_palette',
    'ade_palette',
    'voc_palette',
    'cocostuff_palette',
    'loveda_palette',
    'potsdam_palette',
    'vaihingen_palette',
    'isaid_palette',
    'stare_palette',
    'dataset_aliases',
    'get_classes',
    'get_palette',
    'datafrombytes',
    'synapse_palette',
    'synapse_classes',
    'get_predefined_templates',
    'tokenizer',
    'tokenize',
    'seg_data_to_instance_data',
    'MatchMasks',
    'bdd100k_classes',
    'bdd100k_palette',
]
