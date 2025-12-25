from __future__ import annotations

from typing import Any, Dict, Union

from .base import Detector
from .ensemble import EnsembleDetector
from .supervised import GradientBoostingDetector, LinearSVMDetector, LogisticRegressionDetector
from .unsupervised import (
    AutoencoderDetector,
    CentroidDistance,
    GANDiscriminatorDetector,
    IsolationForestDetector,
    KNNDistance,
    LocalOutlierFactorDetector,
    MahalanobisDistance,
    OneClassSVMDetector,
    PCAReconstructionError,
    VariationalAutoencoderDetector,
    GaussianMixtureEnergy,
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

    if t in ("gmm", "gmm_energy", "gaussian_mixture"):
        return GaussianMixtureEnergy(**cfg)

    if t in ("ae", "autoencoder"):
        return AutoencoderDetector(**cfg)

    if t in ("vae", "variational_autoencoder"):
        return VariationalAutoencoderDetector(**cfg)

    if t in ("gan", "gan_detector", "gan_disc", "gan_discriminator"):
        return GANDiscriminatorDetector(**cfg)

    if t in ("logreg", "logisticregression", "logistic_regression"):
        return LogisticRegressionDetector(**cfg)

    if t in ("linsvm", "linear_svm", "linsvm_calibrated"):
        return LinearSVMDetector(**cfg)

    if t in ("hgbt", "histgradientboosting", "hist_gbt", "gradient_boosting"):
        return GradientBoostingDetector(**cfg)

    if t in ("ensemble", "blend", "aggregate"):
        members = cfg.pop("members", None)
        if not members:
            raise ValueError("Ensemble detector requires a non-empty 'members' list")
        aggregation = cfg.pop("aggregation", "mean")
        return EnsembleDetector(detectors=[build_detector(s) for s in members], aggregation=aggregation, **cfg)

    raise ValueError(f"Unknown detector type: {t}")
