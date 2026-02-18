# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors."""

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: OptConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if pretrained is not None:
            if isinstance(backbone, dict):
                assert backbone.get('pretrained') is None, \
                    'both backbone and segmentor set pretrained weight'
                backbone = dict(backbone)
                backbone['pretrained'] = pretrained
            else:
                assert getattr(backbone, 'pretrained', None) is None, \
                    'both backbone and segmentor set pretrained weight'
                backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if decode_head is not None:
            self._init_decode_head(decode_head)
        else:
            self.decode_head = None
            self.align_corners = False
            self.num_classes = None
            self.out_channels = None
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _test_cfg_get(self, key, default=None):
        if self.test_cfg is None:
            return default
        if isinstance(self.test_cfg, dict):
            return self.test_cfg.get(key, default)
        return getattr(self.test_cfg, key, default)

    def _require_decode_head(self):
        if not self.with_decode_head:
            raise RuntimeError(
                'decode_head is required for train/inference, but got None.')

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        if auxiliary_head is None:
            return
        if isinstance(auxiliary_head, list):
            self.auxiliary_head = nn.ModuleList()
            for head_cfg in auxiliary_head:
                self.auxiliary_head.append(MODELS.build(head_cfg))
        else:
            self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        self._require_decode_head()
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        self._require_decode_head()
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0]) for _ in range(inputs.shape[0])
            ]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        self._require_decode_head()
        x = self.extract_feat(inputs)
        decode_forward = getattr(self.decode_head, 'forward', None)
        if callable(decode_forward):
            try:
                return decode_forward(x)
            except (AttributeError, NotImplementedError, TypeError):
                # Some heads require test-time metadata in forward/predict path.
                pass
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0]) for _ in range(inputs.shape[0])
            ]
        return self.decode_head.predict(x, batch_img_metas, self.test_cfg)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        self._require_decode_head()
        stride = self._test_cfg_get('stride')
        crop_size = self._test_cfg_get('crop_size')
        if stride is None or crop_size is None:
            raise RuntimeError(
                'slide test mode requires test_cfg.stride and test_cfg.crop_size')
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert int((count_mat == 0).sum().item()) == 0
        seg_logits = preds / count_mat
        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        mode = self._test_cfg_get('mode', 'whole')
        assert mode in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got {mode}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        # aug_test rescales all imgs back to ori_shape for now.
        assert rescale
        seg_logit = self.inference(inputs[0], batch_img_metas[0])
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i])
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        return list(seg_pred)
