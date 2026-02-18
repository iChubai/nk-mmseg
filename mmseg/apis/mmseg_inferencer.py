# Copyright (c) OpenMMLab. All rights reserved.
"""Pure-Jittor MMSegInferencer implementation."""

from __future__ import annotations

import os
import os.path as osp
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.registry import init_default_scope

from .inference import inference_model, init_model, show_result_pyplot

InputType = Union[str, np.ndarray, dict]
InputsType = Union[InputType, Sequence[InputType]]


class MMSegInferencer(BaseInferencer):
    """Semantic segmentation inferencer for pure Jittor runtime."""

    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode', 'out_dir'}
    visualize_kwargs: set = {
        'show', 'wait_time', 'img_out_dir', 'opacity', 'return_vis',
        'with_labels'
    }
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 classes: Optional[Union[str, List]] = None,
                 palette: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 device: Optional[str] = 'cuda:0',
                 scope: Optional[str] = 'mmseg') -> None:
        del classes, palette, dataset_name
        init_default_scope(scope if scope else 'mmseg')

        if isinstance(model, str):
            self.model = init_model(model, checkpoint=weights, device=device)
        else:
            self.model = model
        self.num_visualized_imgs = 0
        self.num_pred_imgs = 0

    def _inputs_to_list(self, inputs: InputsType) -> List[InputType]:
        if isinstance(inputs, (list, tuple)):
            return list(inputs)
        return [inputs]

    def forward(self, inputs: InputsType, **kwargs):
        del kwargs
        input_list = self._inputs_to_list(inputs)
        outputs = [inference_model(self.model, item) for item in input_list]
        return outputs

    def visualize(self,
                  inputs: InputsType,
                  preds,
                  show: bool = False,
                  wait_time: float = 0,
                  img_out_dir: Optional[str] = None,
                  opacity: float = 0.5,
                  return_vis: bool = False,
                  with_labels: bool = True):
        vis_results = []
        input_list = self._inputs_to_list(inputs)

        if img_out_dir is not None:
            os.makedirs(img_out_dir, exist_ok=True)

        for i, (inp, pred) in enumerate(zip(input_list, preds)):
            if isinstance(inp, dict):
                inp = inp.get('img', inp.get('rgb'))
            if isinstance(inp, str):
                img = cv2.cvtColor(cv2.imread(inp), cv2.COLOR_BGR2RGB)
                basename = osp.splitext(osp.basename(inp))[0]
            else:
                img = inp
                basename = f'{self.num_visualized_imgs + i:06d}'

            out_file = None
            if img_out_dir is not None:
                out_file = osp.join(img_out_dir, f'{basename}.png')

            vis = show_result_pyplot(
                self.model,
                img,
                pred,
                opacity=opacity,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                with_labels=with_labels,
            )
            if return_vis:
                vis_results.append(vis)

        self.num_visualized_imgs += len(input_list)
        return vis_results

    def postprocess(self,
                    preds,
                    pred_out_dir: Optional[str] = None,
                    return_datasample: bool = False,
                    return_vis: bool = False,
                    vis_results=None):
        if pred_out_dir is not None:
            os.makedirs(pred_out_dir, exist_ok=True)
            for i, pred in enumerate(preds):
                seg = pred.pred_sem_seg.data
                if hasattr(seg, 'numpy'):
                    seg = seg.numpy()
                if seg.ndim == 3:
                    seg = seg[0]
                out_path = osp.join(pred_out_dir,
                                    f'{self.num_pred_imgs + i:06d}.png')
                cv2.imwrite(out_path, seg.astype(np.uint8))
            self.num_pred_imgs += len(preds)

        results = {}
        if return_datasample:
            results['predictions'] = preds
        else:
            masks = []
            for pred in preds:
                seg = pred.pred_sem_seg.data
                if hasattr(seg, 'numpy'):
                    seg = seg.numpy()
                if seg.ndim == 3:
                    seg = seg[0]
                masks.append(seg.astype(np.uint8))
            results['predictions'] = masks

        if return_vis and vis_results is not None:
            results['visualization'] = vis_results

        return results

    def __call__(self,
                 inputs: InputsType,
                 return_datasample: bool = False,
                 pred_out_dir: Optional[str] = None,
                 show: bool = False,
                 wait_time: float = 0,
                 img_out_dir: Optional[str] = None,
                 opacity: float = 0.5,
                 return_vis: bool = False,
                 with_labels: bool = True,
                 **kwargs):
        preds = self.forward(inputs, **kwargs)
        vis_results = None
        if show or img_out_dir is not None or return_vis:
            vis_results = self.visualize(
                inputs,
                preds,
                show=show,
                wait_time=wait_time,
                img_out_dir=img_out_dir,
                opacity=opacity,
                return_vis=return_vis,
                with_labels=with_labels,
            )
        return self.postprocess(
            preds,
            pred_out_dir=pred_out_dir,
            return_datasample=return_datasample,
            return_vis=return_vis,
            vis_results=vis_results,
        )
