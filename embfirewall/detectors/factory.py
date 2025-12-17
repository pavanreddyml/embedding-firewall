# file: embfirewall/detectors/factory.py
from __future__ import annotations

from typing import Any, Dict, Union

from .base import Detector
from .supervised import LogisticRegressionDetector
from .unsupervised import (
    CentroidDistance,
    IsolationForestDetector,
    KNNDistance,
    LocalOutlierFactorDetector,
    MahalanobisDistance,
    OneClassSVMDetector,
    PCAReconstructionError,
)

DetectorSpec = Union[str, Dict[str, Any]]


def build_detector(spec: DetectorSpec) -> Detector:
    """
    Build a detector from a short string or a config mapping.

    Examples:
      {"type":"knn", "k":10, "name":"knn10"}
      {"type":"ocsvm", "nu":0.1, "kernel":"rbf"}
      {"type":"iforest", "n_estimators":200}
      {"type":"mahalanobis"}
      {"type":"lof", "n_neighbors":35}
      {"type":"pca", "n_components":64}
      {"type":"logreg", "C":1.0}
    """
    if isinstance(spec, str):
        t = spec
        cfg: Dict[str, Any] = {"name": spec}
    elif isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError(f"Detector spec missing 'type': {spec}")
        t = str(spec["type"])
        cfg = dict(spec)
        cfg.pop("type", None)
    else:
        raise TypeError(f"Invalid detector spec: {spec!r}")

    t = str(t).lower().strip()

    if t in ("centroid", "mean", "l2"):
        return CentroidDistance(**cfg)

    if t == "knn":
        return KNNDistance(**cfg)

    if t in ("ocsvm", "oneclasssvm", "one_class_svm"):
        return OneClassSVMDetector(**cfg)

    if t in ("iforest", "isolationforest", "isolation_forest"):
        return IsolationForestDetector(**cfg)

    if t in ("mahal", "mahalanobis"):
        return MahalanobisDistance(**cfg)

    if t in ("lof", "localoutlierfactor", "local_outlier_factor"):
        return LocalOutlierFactorDetector(**cfg)

    if t in ("pca", "pca_recon", "pca_reconstruction"):
        return PCAReconstructionError(**cfg)

    if t in ("logreg", "logisticregression", "logistic_regression"):
        return LogisticRegressionDetector(**cfg)

    raise ValueError(f"Unknown detector type: {t}")
