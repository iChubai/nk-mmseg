"""torch.distributed compatibility shim."""


class ReduceOp:
    SUM = 'sum'


def is_available():
    return False


def is_initialized():
    return False


def get_world_size():
    return 1


def get_rank():
    return 0


def all_reduce(tensor, op=None):
    del op
    return tensor


def barrier():
    return None


def init_process_group(*args, **kwargs):
    del args, kwargs
    return None


def destroy_process_group():
    return None
