"""torch.nn.modules.utils compatibility helpers."""


def _ntuple(n):

    def parse(x):
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        return tuple([x] * n)

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

__all__ = ['_single', '_pair', '_triple']
