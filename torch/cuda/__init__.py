import jittor as jt


def is_available():
    return bool(jt.has_cuda)


def device_count():
    return 1 if jt.has_cuda else 0


def current_device():
    return 0
