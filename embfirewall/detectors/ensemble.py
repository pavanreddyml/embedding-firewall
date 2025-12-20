# file: embfirewall/detectors/ensemble.py
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .base import Detector


class EnsembleDetector(Detector):
    """
    Fit multiple detectors on the same data and aggregate their scores.

    Aggregations support "mean", "median", and "max". All base detectors receive the
    same training data and labels (if provided).
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        aggregation: str = "mean",
        name: str = "ensemble",
    ) -> None:
        if not detectors:
            raise ValueError("EnsembleDetector requires at least one base detector")
        super().__init__(name=name)
        self.detectors: List[Detector] = list(detectors)
        self.aggregation = aggregation.lower()
        if self.aggregation not in {"mean", "median", "max"}:
            raise ValueError(f"Unsupported aggregation '{aggregation}' for EnsembleDetector")

    def fit(self, X_train: np.ndarray, y_train: Iterable[int] | None = None) -> "EnsembleDetector":
        for det in self.detectors:
            det.fit(X_train, y_train)
        return self

    def _aggregate(self, scores: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(scores, axis=0)
        if self.aggregation == "mean":
            return np.mean(stacked, axis=0)
        if self.aggregation == "median":
            return np.median(stacked, axis=0)
        return np.max(stacked, axis=0)

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.detectors:
            raise RuntimeError("EnsembleDetector has no fitted base detectors")
        scores = [det.score(X) for det in self.detectors]
        return self._aggregate(scores)
