# file: embfirewall/detectors/__init__.py
from .base import Detector
from .factory import build_detector
from .keywords import DEFAULT_PATTERNS, KeywordBaseline
from .supervised import GradientBoostingDetector, LinearSVMDetector, LogisticRegressionDetector
from .unsupervised import (
    CentroidDistance,
    IsolationForestDetector,
    KNNDistance,
    LocalOutlierFactorDetector,
    MahalanobisDistance,
    OneClassSVMDetector,
    PCAReconstructionError,
)

__all__ = [
    "Detector",
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
]
