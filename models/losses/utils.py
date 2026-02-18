"""
Utility functions for loss computation in DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import jittor as jt
from jittor import nn
import numpy as np


def _to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, jt.Var):
        return data.numpy()
    return np.array(data)


def get_class_weight(class_weight=None, num_classes=None):
    """Get class weights.

    Supported modes:
    - ``class_weight`` is a filepath/list/ndarray/Var of weights.
    - ``class_weight`` is label tensor/array and ``num_classes`` is provided:
      compute inverse-frequency weights from labels.
    """
    if num_classes is not None and class_weight is not None:
        labels = _to_numpy(class_weight).astype(np.int64)
        labels = labels.reshape(-1)
        labels = labels[(labels >= 0) & (labels < int(num_classes))]
        hist = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
        hist = np.maximum(hist, 1.0)
        inv = hist.sum() / (float(num_classes) * hist)
        return jt.array(inv).float32()

    if isinstance(class_weight, str):
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            with open(class_weight, 'r', encoding='utf-8') as f:
                class_weight = [float(line.strip()) for line in f]

    if class_weight is None:
        return None
    if isinstance(class_weight, jt.Var):
        return class_weight.float32()
    return jt.array(class_weight).float32()


def weight_reduce_loss(loss,
                      weight=None,
                      reduction='mean',
                      avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (jt.Var): Element-wise loss.
        weight (jt.Var): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        jt.Var: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss

    if reduction == 'sum':
        return loss.sum()

    if reduction != 'mean':
        raise ValueError(f'Unsupported reduction: {reduction}')

    if avg_factor is None:
        return loss.mean()

    eps = jt.finfo(jt.float32).eps
    if not isinstance(avg_factor, jt.Var):
        avg_factor = jt.array(float(avg_factor)).float32()
    return loss.sum() / (avg_factor + eps)


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (jt.Var): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        jt.Var: Reduced loss tensor.
    """
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f'Unsupported reduction: {reduction}')


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import jittor as jt
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = jt.Var([0, 2, 3], dtype=jt.float32)
    >>> target = jt.Var([1, 1, 1], dtype=jt.float32)
    >>> weight = jt.Var([1, 0, 1], dtype=jt.float32)

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (jt.Var): The prediction.
        target (jt.Var): The learning target of the prediction.

    Returns:
        jt.Var: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = jt.abs(pred - target)
    return loss


@weighted_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (jt.Var): The prediction.
        target (jt.Var): The learning target of the prediction.

    Returns:
        jt.Var: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = (pred - target) ** 2
    return loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (jt.Var): The prediction.
        target (jt.Var): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        jt.Var: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    assert beta > 0
    diff = jt.abs(pred - target)
    loss = jt.ternary(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


def convert_to_one_hot(targets, num_classes, ignore_index=255):
    """Convert targets to one-hot format.
    
    Args:
        targets (jt.Var): Target tensor with shape (N, H, W).
        num_classes (int): Number of classes.
        ignore_index (int): Index to ignore.
        
    Returns:
        jt.Var: One-hot tensor with shape (N, C, H, W).
    """
    assert targets.ndim == 3, f"Expected 3D tensor, got {targets.ndim}D"
    
    N, H, W = targets.shape
    one_hot = jt.zeros((N, num_classes, H, W), dtype=jt.float32)
    
    # Create mask for valid pixels
    valid_mask = (targets != ignore_index)
    valid_targets = targets[valid_mask]
    
    if valid_targets.numel() > 0:
        # Get indices for valid pixels
        batch_idx, h_idx, w_idx = jt.where(valid_mask)
        one_hot[batch_idx, valid_targets, h_idx, w_idx] = 1.0
    
    return one_hot


def focal_loss_with_logits(input, target, alpha=1, gamma=2, reduction='mean', ignore_index=255):
    """Focal loss with logits.
    
    Args:
        input (jt.Var): Input logits with shape (N, C, H, W).
        target (jt.Var): Target tensor with shape (N, H, W).
        alpha (float): Weighting factor for rare class.
        gamma (float): Focusing parameter.
        reduction (str): Reduction method.
        ignore_index (int): Index to ignore.
        
    Returns:
        jt.Var: Computed focal loss.
    """
    ce_loss = nn.cross_entropy_loss(input, target, ignore_index=ignore_index, reduction='none')
    pt = jt.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
