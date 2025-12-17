# file: embfirewall/detectors/supervised.py
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Detector


class LogisticRegressionDetector(Detector):
    def __init__(self, C: float = 1.0, max_iter: int = 2000, name: str = "logreg") -> None:
        super().__init__(name=name)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.model_: Optional[LogisticRegression] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "LogisticRegressionDetector":
        if y_train is None:
            raise ValueError("LogReg requires y_train labels.")
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter, n_jobs=-1)
        self.model_.fit(X_train, y_train.astype(int))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fit().")
        return self.model_.predict_proba(X)[:, 1]
