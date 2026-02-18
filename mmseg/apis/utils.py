import importlib
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

import cv2
import jittor as jt
import numpy as np


ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def load_local_config(config: Union[str, Path, object]):
    if not isinstance(config, (str, Path)):
        return config

    config = str(config)
    if config.endswith('.py') and os.path.isfile(config):
        cfg_path = Path(config).resolve()
        repo_root = Path(__file__).resolve().parents[2]

        # Prefer package import so relative imports in config files work.
        mod = None
        try:
            rel = cfg_path.relative_to(repo_root)
            if rel.suffix == '.py':
                module_name = '.'.join(rel.with_suffix('').parts)
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                mod = importlib.import_module(module_name)
        except Exception:
            mod = None

        if mod is None:
            spec = importlib.util.spec_from_file_location('cfg_mod', str(cfg_path))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
        if hasattr(mod, 'C'):
            return getattr(mod, 'C')
        return mod

    mod = importlib.import_module(config)
    if hasattr(mod, 'C'):
        return getattr(mod, 'C')
    return mod


def _to_tensor(image: np.ndarray, mean, std):
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean, dtype=np.float32)) / np.array(
        std, dtype=np.float32)
    image = image.transpose(2, 0, 1)[None, ...]
    return jt.array(image)


def _prepare_data(imgs: ImageType, cfg):
    is_batch = isinstance(imgs, (list, tuple))
    if not is_batch:
        imgs = [imgs]

    data = defaultdict(list)
    mean = getattr(cfg, 'norm_mean', [0.485, 0.456, 0.406])
    std = getattr(cfg, 'norm_std', [0.229, 0.224, 0.225])

    for item in imgs:
        modal_x = None
        if isinstance(item, dict):
            rgb = item.get('img', item.get('rgb'))
            modal = item.get('modal_x', None)
            if isinstance(rgb, str):
                rgb = cv2.cvtColor(cv2.imread(rgb), cv2.COLOR_BGR2RGB)
            if isinstance(modal, str):
                modal = cv2.imread(modal, cv2.IMREAD_UNCHANGED)
            if modal is not None and modal.ndim == 2:
                modal = modal[..., None]
            if modal is not None:
                modal_x = _to_tensor(modal, mean=[0.0], std=[1.0])
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            rgb, modal = item
            if isinstance(rgb, str):
                rgb = cv2.cvtColor(cv2.imread(rgb), cv2.COLOR_BGR2RGB)
            if isinstance(modal, str):
                modal = cv2.imread(modal, cv2.IMREAD_UNCHANGED)
            if modal is not None and modal.ndim == 2:
                modal = modal[..., None]
            if modal is not None:
                modal_x = _to_tensor(modal, mean=[0.0], std=[1.0])
        elif isinstance(item, str):
            rgb = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
        else:
            rgb = item

        data['inputs'].append(_to_tensor(rgb, mean, std))
        data['modal_x'].append(modal_x)

    return data, is_batch
