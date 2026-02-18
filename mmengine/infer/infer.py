"""Inferencer compatibility base class."""

from __future__ import annotations

from typing import Any, Union

ModelType = Union[str, dict, object]


class BaseInferencer:

    def __init__(self, model: Any = None, **kwargs):
        del kwargs
        self.model = model

    def __call__(self, inputs, **kwargs):
        return self.postprocess(self.forward(self.preprocess(inputs, **kwargs), **kwargs), **kwargs)

    def preprocess(self, inputs, **kwargs):
        del kwargs
        return inputs

    def forward(self, inputs, **kwargs):
        del kwargs
        return inputs

    def postprocess(self, preds, **kwargs):
        del kwargs
        return preds
