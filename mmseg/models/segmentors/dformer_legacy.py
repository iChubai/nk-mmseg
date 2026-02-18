# Copyright (c) OpenMMLab. All rights reserved.
"""Legacy DFormer segmentor wrapper registered into mmseg MODELS."""

from __future__ import annotations

from typing import Any

import jittor.nn as nn

from mmseg.registry import MODELS
from models.builder import Config as LegacyConfig
from models.builder import EncoderDecoder as LegacyEncoderDecoder


def _to_legacy_cfg(cfg: Any, **overrides: Any):
    if cfg is None:
        cfg_obj = LegacyConfig()
    elif isinstance(cfg, dict):
        cfg_obj = LegacyConfig(**cfg)
    else:
        cfg_obj = cfg

    for key, value in overrides.items():
        if value is None:
            continue
        setattr(cfg_obj, key, value)
    return cfg_obj


@MODELS.register_module(name=['DFormerLegacySegmentor', 'DFormerRGBXSegmentor'])
class DFormerLegacySegmentor(LegacyEncoderDecoder):
    """Adapter that exposes legacy DFormer builder via mmseg registry."""

    def __init__(self,
                 cfg=None,
                 backbone=None,
                 decoder=None,
                 num_classes=None,
                 decoder_embed_dim=None,
                 drop_path_rate=None,
                 aux_rate=None,
                 pretrained_model=None,
                 criterion=None,
                 norm_layer='BN',
                 syncbn=False,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        cfg_obj = _to_legacy_cfg(
            cfg,
            backbone=backbone,
            decoder=decoder,
            num_classes=num_classes,
            decoder_embed_dim=decoder_embed_dim,
            drop_path_rate=drop_path_rate,
            aux_rate=aux_rate,
            pretrained_model=pretrained_model,
            **kwargs)

        norm = self._resolve_norm_layer(norm_layer, syncbn)
        if criterion is None:
            ignore_index = int(getattr(cfg_obj, 'background', 255))
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        super().__init__(
            cfg=cfg_obj,
            criterion=criterion,
            norm_layer=norm,
            syncbn=syncbn)

        # Keep mmengine-style attributes for compatibility.
        self.data_preprocessor = data_preprocessor
        self.init_cfg = init_cfg

    @staticmethod
    def _resolve_norm_layer(norm_layer, syncbn):
        if norm_layer is None:
            return nn.BatchNorm2d
        if isinstance(norm_layer, str):
            if syncbn or norm_layer.lower() in ('syncbn', 'sync_batchnorm'):
                return getattr(nn, 'SyncBatchNorm', nn.BatchNorm2d)
            return nn.BatchNorm2d
        return norm_layer

