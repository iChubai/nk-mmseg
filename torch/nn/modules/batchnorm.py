"""torch.nn.modules.batchnorm compatibility."""

import jittor.nn as nn

_BatchNorm = nn.BatchNorm

__all__ = ['_BatchNorm']
