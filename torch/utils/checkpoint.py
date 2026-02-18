"""torch.utils.checkpoint compatibility shim."""


def checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)
