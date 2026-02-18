"""Device helpers."""

import jittor as jt


def get_device() -> str:
    return 'cuda' if jt.flags.use_cuda else 'cpu'
