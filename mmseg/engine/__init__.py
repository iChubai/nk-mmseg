"""Jittor-friendly engine package exports.

Keep package import side effects minimal. Downstream code can import optional
submodules directly (e.g. ``mmseg.engine.config``) when needed.
"""

# Always expose registries.
from .register import *  # noqa: F401, F403
from .register import __all__ as _register_all

__all__ = list(_register_all) + ['SegVisualizationHook']


def __getattr__(name):
    # Avoid importing hooks at module import time to prevent circular imports:
    # mmseg.registry -> mmseg.engine.register -> mmseg.engine -> hooks -> mmseg.registry
    if name == 'SegVisualizationHook':
        from .hooks import SegVisualizationHook  # local import by design
        return SegVisualizationHook
    raise AttributeError(name)
