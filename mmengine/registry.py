"""Registry helpers."""

try:
    from mmseg.registry import MODELS
except Exception:
    from mmseg.engine.register import MODELS


def init_default_scope(scope_name: str | None = None):
    del scope_name
    return None


__all__ = ['MODELS', 'init_default_scope']
