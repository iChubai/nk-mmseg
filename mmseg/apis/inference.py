import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import jittor as jt
import numpy as np

from models import build_model
from mmseg.structures import SegDataSample
from mmseg.utils import (SampleList, dataset_aliases, get_classes,
                         get_palette)
from mmseg.visualization import SegLocalVisualizer
from utils.jt_utils import load_model
from .utils import ImageType, _prepare_data, load_local_config


def init_model(config: Union[str, Path, object],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a Jittor segmentor from local config."""
    cfg = load_local_config(config)
    if cfg_options:
        for k, v in cfg_options.items():
            setattr(cfg, k, v)

    if device.startswith('cuda') and jt.has_cuda:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0

    model = build_model(cfg)
    if checkpoint is not None:
        model = load_model(model, checkpoint)

    dataset_name = getattr(cfg, 'dataset_name', None)
    if dataset_name is not None:
        model.dataset_meta = {
            'classes': list(getattr(cfg, 'class_names', get_classes(dataset_name))),
            'palette': get_palette(dataset_name),
        }
    else:
        warnings.warn('dataset_name is not set in config, using fallback metadata')
        model.dataset_meta = {
            'classes': ['class_0'],
            'palette': [[0, 0, 0]],
        }
    model.cfg = cfg
    model.eval()
    return model


def _to_data_sample(pred: jt.Var) -> SegDataSample:
    sample = SegDataSample()
    if pred.ndim == 4:
        pred = pred.argmax(dim=1)[0]
    elif pred.ndim == 3:
        pred = pred.argmax(dim=0)
    sample.set_pred_sem_seg(pred.int32())
    return sample


def inference_model(model, img: ImageType) -> Union[SegDataSample, SampleList]:
    """Inference image(s) with Jittor segmentor."""
    data, is_batch = _prepare_data(img, model.cfg)
    results = []
    with jt.no_grad():
        for rgb, modal_x in zip(data['inputs'], data['modal_x']):
            pred, _ = model(rgb, modal_x, None)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            results.append(_to_data_sample(pred))
    return results if is_batch else results[0]


def show_result_pyplot(model,
                       img: Union[str, np.ndarray],
                       result: SegDataSample,
                       opacity: float = 0.5,
                       title: str = '',
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       wait_time: float = 0,
                       show: bool = True,
                       with_labels: Optional[bool] = True,
                       save_dir=None,
                       out_file=None):
    """Visualize segmentation prediction."""
    if isinstance(img, str):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    else:
        image = img

    visualizer = SegLocalVisualizer(save_dir=save_dir, alpha=opacity)
    visualizer.dataset_meta = dict(
        classes=model.dataset_meta.get('classes', None),
        palette=model.dataset_meta.get('palette', None),
    )
    visualizer.add_datasample(
        name=title if title else 'result',
        image=image,
        data_sample=result,
        draw_gt=draw_gt,
        draw_pred=draw_pred,
        wait_time=wait_time,
        out_file=out_file,
        show=show,
        with_labels=with_labels,
    )
    return visualizer.get_image()

