# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class Mask2FormerHead(MMDET_Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        super().__init__(**kwargs)

        self._mmdet_available = MMDET_Mask2FormerHead is not BaseModule
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        self.in_index = kwargs.get('in_index', -1)
        self.feat_channels = kwargs['feat_channels']

        self.cls_embed = nn.Linear(self.feat_channels, self.num_classes + 1)
        if not self._mmdet_available:
            in_channels = kwargs.get('in_channels', self.feat_channels)
            fallback_in_channels = self._resolve_fallback_in_channels(
                in_channels, self.in_index)
            self.fallback_proj = nn.Conv2d(
                fallback_in_channels, self.feat_channels, kernel_size=1)
            self.fallback_act = nn.ReLU()
            self.fallback_seg = nn.Conv2d(
                self.feat_channels, self.num_classes, kernel_size=1)

    @staticmethod
    def _resolve_fallback_in_channels(in_channels, in_index) -> int:
        if isinstance(in_channels, (list, tuple)):
            if isinstance(in_index, (list, tuple)) and len(in_index) > 0:
                idx = int(in_index[-1])
            elif isinstance(in_index, int):
                idx = int(in_index)
            else:
                idx = -1
            if idx < 0:
                idx += len(in_channels)
            idx = max(0, min(idx, len(in_channels) - 1))
            return int(in_channels[idx])
        return int(in_channels)

    def _select_feat(self, x: Tuple[Tensor]) -> Tensor:
        if not isinstance(x, (list, tuple)):
            return x
        if isinstance(self.in_index, (list, tuple)) and len(self.in_index) > 0:
            idx = int(self.in_index[-1])
        elif isinstance(self.in_index, int):
            idx = int(self.in_index)
        else:
            idx = -1
        return x[idx]

    def _fallback_forward_logits(self,
                                 x: Tuple[Tensor],
                                 target_size: Optional[Tuple[int, int]] = None
                                 ) -> Tensor:
        feat = self._select_feat(x)
        seg_logits = self.fallback_seg(self.fallback_act(self.fallback_proj(feat)))
        if target_size is not None:
            seg_logits = F.interpolate(
                seg_logits,
                size=tuple(int(v) for v in target_size),
                mode='bilinear',
                align_corners=self.align_corners)
        return seg_logits

    def _fallback_target_size_from_metas(self, batch_img_metas):
        if not isinstance(batch_img_metas, list) or len(batch_img_metas) == 0:
            return None
        meta0 = batch_img_metas[0]
        for key in ('batch_input_shape', 'img_shape', 'pad_shape', 'ori_shape'):
            if key in meta0 and meta0[key] is not None:
                return meta0[key]
        return None

    def forward(self,
                x: Tuple[Tensor],
                batch_data_samples: Optional[SampleList] = None):
        if not self._mmdet_available:
            target_size = None
            if isinstance(batch_data_samples, list) and len(batch_data_samples) > 0:
                meta0 = batch_data_samples[0].metainfo
                target_size = self._fallback_target_size_from_metas([meta0])
            return self._fallback_forward_logits(x, target_size)
        return super().forward(x, batch_data_samples)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        if not self._mmdet_available:
            del train_cfg
            gt_sem_seg = torch.stack(
                [sample.gt_sem_seg.data for sample in batch_data_samples], dim=0)
            gt_sem_seg = gt_sem_seg.squeeze(1).long()
            seg_logits = self._fallback_forward_logits(
                x, target_size=gt_sem_seg.shape[-2:])
            loss_ce = F.cross_entropy(
                seg_logits, gt_sem_seg, ignore_index=self.ignore_index)
            return {'loss_ce': loss_ce}

        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        del test_cfg
        if not self._mmdet_available:
            target_size = self._fallback_target_size_from_metas(batch_img_metas)
            return self._fallback_forward_logits(x, target_size)

        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits
