# file: embfirewall/detectors/supervised.py
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .base import Detector


class LogisticRegressionDetector(Detector):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: Optional[Union[str, dict]] = None,
        name: str = "logreg",
    ) -> None:
        super().__init__(name=name)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.model_: Optional[LogisticRegression] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "LogisticRegressionDetector":
        if y_train is None:
            raise ValueError("LogReg requires y_train labels.")
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            n_jobs=-1,
            class_weight=self.class_weight,
        )
        self.model_.fit(X_train, y_train.astype(int))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fit().")
        return self.model_.predict_proba(X)[:, 1]


class LinearSVMDetector(Detector):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 5000,
        class_weight: Optional[Union[str, dict]] = "balanced",
        calibration_cv: int = 3,
        calibration_method: str = "sigmoid",
        name: str = "linsvm_calibrated",
    ) -> None:
        super().__init__(name=name)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.calibration_cv = int(calibration_cv)
        self.calibration_method = str(calibration_method)
        self.model_: Optional[CalibratedClassifierCV] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "LinearSVMDetector":
        if y_train is None:
            raise ValueError("LinearSVM requires y_train labels.")
        base = LinearSVC(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight)
        self.model_ = CalibratedClassifierCV(
            estimator=base,
            method=self.calibration_method,
            cv=self.calibration_cv,
            n_jobs=-1,
        )
        self.model_.fit(X_train, y_train.astype(int))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fit().")
        return self.model_.predict_proba(X)[:, 1]


class GradientBoostingDetector(Detector):
    def __init__(
        self,
        learning_rate: float = 0.08,
        max_depth: int = 8,
        max_iter: int = 300,
        l2_regularization: float = 0.0,
        name: str = "hgbt",
    ) -> None:
        super().__init__(name=name)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.max_iter = int(max_iter)
        self.l2_regularization = float(l2_regularization)
        self.model_: Optional[HistGradientBoostingClassifier] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "GradientBoostingDetector":
        if y_train is None:
            raise ValueError("GradientBoostingDetector requires y_train labels.")
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            l2_regularization=self.l2_regularization,
        )
        self.model_.fit(X_train, y_train.astype(int))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fit().")
        proba = self.model_.predict_proba(X)
        return proba[:, 1]
