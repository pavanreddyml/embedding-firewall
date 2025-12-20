# file: embfirewall/detectors/__init__.py
from .base import Detector
from .ensemble import EnsembleDetector
from .factory import build_detector
from .keywords import DEFAULT_PATTERNS, KeywordBaseline
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

__all__ = [
    "Detector",
    "EnsembleDetector",
    "build_detector",
    "KeywordBaseline",
    "DEFAULT_PATTERNS",
    "LogisticRegressionDetector",
    "LinearSVMDetector",
    "GradientBoostingDetector",
    "CentroidDistance",
    "KNNDistance",
    "OneClassSVMDetector",
    "IsolationForestDetector",
    "MahalanobisDistance",
    "LocalOutlierFactorDetector",
    "PCAReconstructionError",
    "GaussianMixtureEnergy",
    "AutoencoderDetector",
    "VariationalAutoencoderDetector",
    "GANDiscriminatorDetector",
]
