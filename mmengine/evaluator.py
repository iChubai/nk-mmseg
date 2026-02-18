"""Minimal BaseMetric implementation."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod


class BaseMetric(metaclass=ABCMeta):

    def __init__(self, collect_device: str = 'cpu', prefix: str | None = None):
        self.collect_device = collect_device
        self.prefix = prefix
        self.results = []
        self.dataset_meta = {}

    @abstractmethod
    def process(self, data_batch: dict, data_samples):
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, results: list):
        raise NotImplementedError

    def evaluate(self, size=None):
        del size
        metrics = self.compute_metrics(self.results)
        if self.prefix is not None and isinstance(metrics, dict):
            metrics = {f'{self.prefix}/{k}': v for k, v in metrics.items()}
        self.results = []
        return metrics
