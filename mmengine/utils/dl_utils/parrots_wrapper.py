"""Parrots wrapper compatibility for norm type checks."""

import jittor.nn as nn

_BatchNorm = (nn.BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_InstanceNorm = (nn.InstanceNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
                 nn.InstanceNorm3d)
