"""Subset of torch._utils symbols required by Jittor save/load shims."""


def _rebuild_tensor_v2(*args, **kwargs):
    raise NotImplementedError("_rebuild_tensor_v2 is only used for serialization hooks")

