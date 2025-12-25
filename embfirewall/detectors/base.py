from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Detector(ABC):
    """Higher score => more anomalous / more likely malicious."""

    def __init__(self, name: str) -> None:
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "Detector":
        raise NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None
