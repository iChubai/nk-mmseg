import contextlib


@contextlib.contextmanager
def autocast(enabled=True, **kwargs):
    del enabled, kwargs
    yield
