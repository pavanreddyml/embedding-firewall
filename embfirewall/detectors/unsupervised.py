# file: embfirewall/detectors/unsupervised.py
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from .base import Detector


class CentroidDistance(Detector):
    def __init__(self, name: str = "centroid") -> None:
        super().__init__(name=name)
        self.centroid_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "CentroidDistance":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"CentroidDistance.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        self.centroid_ = np.mean(X_train, axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.centroid_ is None:
            raise RuntimeError("CentroidDistance not fitted")
        diff = X - self.centroid_[None, :]
        return np.linalg.norm(diff, axis=1)


class KNNDistance(Detector):
    def __init__(self, k: int = 10, name: str = "knn") -> None:
        super().__init__(name=name)
        self.k = int(k)
        self.nn_: Optional[NearestNeighbors] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "KNNDistance":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"KNNDistance.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        k = max(1, int(self.k))
        self.nn_ = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self.nn_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.nn_ is None:
            raise RuntimeError("KNNDistance not fitted")
        dists, _ = self.nn_.kneighbors(X, return_distance=True)
        return np.mean(dists, axis=1)


class OneClassSVMDetector(Detector):
    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        name: str = "ocsvm",
    ) -> None:
        super().__init__(name=name)
        self.nu = float(nu)
        self.kernel = str(kernel)
        self.gamma = gamma
        self.model_: Optional[OneClassSVM] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "OneClassSVMDetector":
        self.model_ = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("OneClassSVMDetector not fitted")
        # decision_function: + for inliers, - for outliers => negate
        return -self.model_.decision_function(X).reshape(-1)


class IsolationForestDetector(Detector):
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        random_state: Optional[int] = 0,
        n_jobs: int = -1,
        name: str = "iforest",
        **_ignored: object,
    ) -> None:
        # **_ignored keeps backward-compat if configs pass extra keys (won't crash)
        super().__init__(name=name)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.model_: Optional[IsolationForest] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("IsolationForestDetector not fitted")
        # score_samples: higher => more normal, so negate
        return -self.model_.score_samples(X).reshape(-1)


class MahalanobisDistance(Detector):
    def __init__(self, name: str = "mahalanobis") -> None:
        super().__init__(name=name)
        self.mean_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "MahalanobisDistance":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(
                f"MahalanobisDistance.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}"
            )
        lw = LedoitWolf().fit(X_train)
        self.mean_ = lw.location_
        self.precision_ = lw.precision_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisDistance not fitted")
        diff = X - self.mean_[None, :]
        # squared Mahalanobis distance
        m = np.einsum("ij,jk,ik->i", diff, self.precision_, diff)
        return m.reshape(-1)


class LocalOutlierFactorDetector(Detector):
    def __init__(
        self,
        n_neighbors: int = 35,
        leaf_size: int = 30,
        name: str = "lof",
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name=name)
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.n_jobs = int(n_jobs)
        self.model_: Optional[LocalOutlierFactor] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "LocalOutlierFactorDetector":
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            leaf_size=self.leaf_size,
            metric="euclidean",
            novelty=True,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("LocalOutlierFactorDetector not fitted")
        # decision_function: + inlier, - outlier => negate
        return -self.model_.decision_function(X).reshape(-1)


class PCAReconstructionError(Detector):
    def __init__(
        self,
        n_components: int = 64,
        whiten: bool = False,
        random_state: Optional[int] = 0,
        name: str = "pca",
    ) -> None:
        super().__init__(name=name)
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.random_state = random_state
        self.pca_: Optional[PCA] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "PCAReconstructionError":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"PCAReconstructionError.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        n = max(1, min(int(self.n_components), int(X_train.shape[1])))
        self.pca_ = PCA(n_components=n, whiten=self.whiten, random_state=self.random_state)
        self.pca_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCAReconstructionError not fitted")
        Z = self.pca_.transform(X)
        X_hat = self.pca_.inverse_transform(Z)
        err = X - X_hat
        return np.sum(err * err, axis=1).reshape(-1)
