"""torch.nn.modules compatibility namespace."""

from .batchnorm import _BatchNorm
from .utils import _pair, _single, _triple

__all__ = ['_BatchNorm', '_single', '_pair', '_triple']
