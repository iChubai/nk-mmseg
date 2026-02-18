"""Jittor-native registries for mmseg-style modules.

This mirrors the key names used by MMSegmentation while using the local
Jittor registry implementation.
"""

from mmseg.engine.register import (DATASETS, EVALUATORS, HOOKS, LOOPS, MODELS,
                                   OPTIMIZERS, SCHEDULERS, TASK_UTILS,
                                   TRANSFORMS, Register)

RUNNERS = Register('runner')
RUNNER_CONSTRUCTORS = Register('runner constructor')
DATA_SAMPLERS = Register('data sampler')

MODEL_WRAPPERS = Register('model wrapper')
WEIGHT_INITIALIZERS = Register('weight initializer')

OPTIM_WRAPPERS = Register('optimizer wrapper')
OPTIM_WRAPPER_CONSTRUCTORS = Register('optimizer wrapper constructor')

PARAM_SCHEDULERS = SCHEDULERS

METRICS = Register('metric')
EVALUATOR = EVALUATORS

VISUALIZERS = Register('visualizer')
VISBACKENDS = Register('vis_backend')
LOG_PROCESSORS = Register('log_processor')
INFERENCERS = Register('inferencer')

# Trigger module registration for built-in segmentors.
try:
    import mmseg.models  # noqa: F401
except Exception:
    pass
