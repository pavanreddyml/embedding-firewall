# file: embfirewall/runner.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .detectors import DEFAULT_PATTERNS, KeywordBaseline
from .detectors.factory import build_detector
from .embeddings import EmbeddingSpec, build_embedder
from .embeddings.cache import EmbeddingCache
from .viz import write_all_figures


def _y_binary(label: str, malicious_label: str) -> int:
    # We treat only the configured malicious label as positive
    return 1 if label == malicious_label else 0


def _y_multi(label: str, positive_labels: Tuple[str, ...]) -> int:
    return 1 if label in positive_labels else 0


@dataclass
class DatasetSlices:
    train_texts: List[str]
    val_texts: List[str]
    val_labels: List[str]
    test_texts: List[str]
    test_labels: List[str]

    def __post_init__(self) -> None:
        if len(self.val_texts) != len(self.val_labels):
            raise ValueError("val_texts and val_labels must be same length")
        if len(self.test_texts) != len(self.test_labels):
            raise ValueError("test_texts and test_labels must be same length")


@dataclass
class RunConfig:
    run_dir: str

    embedding_cache_dir: Optional[str] = None

    normal_label: str = "normal"
    borderline_label: str = "borderline"
    malicious_label: str = "malicious"

    # Operating points (target FPR) are calibrated on NORMAL validation rows by default
    fpr_points: Tuple[float, float] = (0.05, 0.10)
    threshold_calibration_set: str = "val"  # "val" (recommended) or "test" (not recommended; leakage)

    # Supervised detectors are trained on a portion of val, and thresholds are calibrated on the rest
    supervised_val_fit_frac: float = 0.5

    embedding_models: List[EmbeddingSpec] = None  # type: ignore

    enable_unsupervised: bool = True
    enable_supervised: bool = True
    enable_keyword: bool = True

    unsupervised_detectors: Optional[List[Dict[str, Any]]] = None
    supervised_detectors: Optional[List[Dict[str, Any]]] = None
    keyword_patterns: Optional[List[str]] = None

    # Metrics for unsupervised/keyword runs can optionally widen the positive set beyond malicious.
    # Defaults to (malicious_label,) to preserve prior behavior.
    unsupervised_positive_labels: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.embedding_models is None:
            self.embedding_models = [
                EmbeddingSpec(
                    kind="st",
                    name="E1_minilm",
                    model_id="sentence-transformers/all-MiniLM-L6-v2",
                    device="cuda",
                    batch_size=128,
                    normalize=True,
                ),
                EmbeddingSpec(
                    kind="st",
                    name="E2_bge_base",
                    model_id="BAAI/bge-base-en-v1.5",
                    device="cuda",
                    batch_size=64,
                    normalize=True,
                ),
            ]

        if self.unsupervised_detectors is None:
            self.unsupervised_detectors = [
                {"type": "centroid", "name": "centroid"},
                {"type": "knn", "k": 10, "name": "knn10"},
                {"type": "ocsvm", "nu": 0.1, "kernel": "rbf", "gamma": "scale", "name": "ocsvm"},
                {"type": "iforest", "n_estimators": 200, "name": "iforest"},
                # extras (strong baselines)
                {"type": "mahalanobis", "name": "mahal"},
                {"type": "lof", "n_neighbors": 35, "name": "lof35"},
                {"type": "pca", "n_components": 64, "name": "pca64"},
            ]

        if self.supervised_detectors is None:
            self.supervised_detectors = [
                {"type": "logreg", "C": 1.0, "name": "logreg"},
            ]

        if self.unsupervised_positive_labels is None:
            self.unsupervised_positive_labels = (self.malicious_label,)


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, float(center - half))
    hi = min(1.0, float(center + half))
    return lo, hi


class ExperimentRunner:
    def __init__(self, cfg: RunConfig, data: DatasetSlices):
        self.cfg = cfg
        self.data = data

        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if cfg.embedding_cache_dir is not None:
            self.embedding_cache_dir = Path(cfg.embedding_cache_dir)
        else:
            self.embedding_cache_dir = self.run_dir.parent / "embedding_cache"
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        self.val_y = np.array([_y_binary(x, cfg.malicious_label) for x in data.val_labels], dtype=np.int32)
        self.test_y = np.array([_y_binary(x, cfg.malicious_label) for x in data.test_labels], dtype=np.int32)
        self.unsup_test_y = np.array(
            [_y_multi(x, cfg.unsupervised_positive_labels) for x in data.test_labels], dtype=np.int32
        )

        self.val_is_normal = np.array([x == cfg.normal_label for x in data.val_labels], dtype=bool)
        self.test_is_normal = np.array([x == cfg.normal_label for x in data.test_labels], dtype=bool)

        self.test_is_malicious = np.array([x == cfg.malicious_label for x in data.test_labels], dtype=bool)
        self.test_is_borderline = np.array([x == cfg.borderline_label for x in data.test_labels], dtype=bool)

        # Supervised: split val into fit/calibration (stratified by val_y)
        self.sup_fit_idx, self.sup_cal_idx = self._stratified_split_indices(
            self.val_y, frac=float(cfg.supervised_val_fit_frac), seed=1337
        )
        self.val_fit_y = self.val_y[self.sup_fit_idx]
        self.val_cal_y = self.val_y[self.sup_cal_idx]
        self.val_fit_is_normal = self.val_is_normal[self.sup_fit_idx]
        self.val_cal_is_normal = self.val_is_normal[self.sup_cal_idx]

    @staticmethod
    def _stratified_split_indices(y: np.ndarray, *, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        frac = float(frac)
        frac = max(0.0, min(1.0, frac))
        rng = np.random.RandomState(int(seed))

        fit: List[int] = []
        cal: List[int] = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            cut = int(np.floor(len(idx) * frac))
            fit.extend(idx[:cut].tolist())
            cal.extend(idx[cut:].tolist())

        fit_idx = np.array(sorted(fit), dtype=np.int64)
        cal_idx = np.array(sorted(cal), dtype=np.int64)
        return fit_idx, cal_idx

    @staticmethod
    def _metrics_summary(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        # y: {0,1}; scores: higher => more likely positive
        out: Dict[str, Any] = {}
        try:
            out["auroc"] = float(roc_auc_score(y, scores))
        except Exception:
            out["auroc"] = None
        try:
            out["auprc"] = float(average_precision_score(y, scores))
        except Exception:
            out["auprc"] = None
        return out

    @staticmethod
    def _operating_threshold_from_normals(normal_scores: np.ndarray, target_fpr: float) -> float:
        """
        Choose a threshold so that FPR ~= target_fpr under rule: predict_positive = (score >= thr).

        Uses a discrete-safe selection: pick the smallest threshold with FPR <= target_fpr.
        """
        if normal_scores.size == 0:
            return float("inf")
        target_fpr = float(target_fpr)
        target_fpr = max(0.0, min(1.0, target_fpr))

        vals = np.unique(normal_scores)
        vals.sort()

        # Try increasing thresholds until we meet the FPR budget.
        best_thr = float(vals[-1])
        found = False
        for thr in vals:
            fpr = float(np.mean(normal_scores >= thr))
            if fpr <= target_fpr + 1e-12:
                best_thr = float(thr)
                found = True
                break

        # If nothing meets budget (can happen if all scores identical and target_fpr < 1),
        # force thr above max to yield FPR=0.
        if not found:
            best_thr = float(vals[-1]) + 1e-12

        return best_thr

    def _operating_point(
        self,
        *,
        fpr_target: float,
        thr: float,
        scores_test: np.ndarray,
    ) -> Dict[str, Any]:
        pred = scores_test >= thr

        # test FPR on normal
        n_norm = int(np.sum(self.test_is_normal))
        fp = int(np.sum(pred & self.test_is_normal))
        test_fpr = float(fp / n_norm) if n_norm > 0 else 0.0
        fpr_ci = _wilson_ci(fp, n_norm)

        # TPR on malicious
        n_mal = int(np.sum(self.test_is_malicious))
        tp = int(np.sum(pred & self.test_is_malicious))
        tpr = float(tp / n_mal) if n_mal > 0 else 0.0
        tpr_ci = _wilson_ci(tp, n_mal)

        # borderline block rate
        n_bl = int(np.sum(self.test_is_borderline))
        bl_block = int(np.sum(pred & self.test_is_borderline))
        bl_rate = float(bl_block / n_bl) if n_bl > 0 else 0.0
        bl_ci = _wilson_ci(bl_block, n_bl)

        return {
            "fpr_target": float(fpr_target),
            "thr": float(thr),
            "test_fpr": test_fpr,
            "test_fpr_ci95": [float(fpr_ci[0]), float(fpr_ci[1])],
            "tpr_malicious": tpr,
            "tpr_malicious_ci95": [float(tpr_ci[0]), float(tpr_ci[1])],
            "borderline_block_rate": bl_rate,
            "borderline_block_ci95": [float(bl_ci[0]), float(bl_ci[1])],
        }

    @staticmethod
    def _materialize_embeddings(vecs: List[Optional[np.ndarray]]) -> np.ndarray:
        if not vecs:
            return np.empty((0, 0), dtype=np.float32)

        dim: Optional[int] = None
        for v in vecs:
            if v is not None:
                dim = int(v.size)
                break

        if dim is None:
            return np.empty((len(vecs), 0), dtype=np.float32)

        zero = np.zeros((dim,), dtype=np.float32)
        out: List[np.ndarray] = []
        for v in vecs:
            if v is None:
                out.append(zero.copy())
            else:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                out.append(arr)

        return np.vstack([a.reshape(1, -1) for a in out])

    def _embed(self, emb_spec: EmbeddingSpec, texts: List[str]) -> Tuple[np.ndarray, float]:
        t0_total = time.time()

        with EmbeddingCache(self.embedding_cache_dir, emb_spec.model_id) as cache:
            cached_vecs, missing_idx, hashes = cache.lookup(texts)

            dt_embed = 0.0
            if missing_idx:
                embedder = build_embedder(emb_spec)
                missing_texts = [texts[i] for i in missing_idx]
                missing_vecs, dt_embed = embedder.embed(missing_texts, desc=f"embed[{emb_spec.name}]")
                cache.store(hashes=hashes, embeddings=missing_vecs, indices=missing_idx)

                for arr_idx, text_idx in enumerate(missing_idx):
                    cached_vecs[text_idx] = missing_vecs[arr_idx].reshape(-1)

            X = self._materialize_embeddings(cached_vecs)

        dt_total = time.time() - t0_total
        # Report the wall-clock time spent retrieving/embedding for transparency.
        return X, float(dt_total)

    def run(self) -> str:
        results: Dict[str, Any] = {
            "meta": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "threshold_calibration_set": self.cfg.threshold_calibration_set,
                "supervised_val_fit_frac": float(self.cfg.supervised_val_fit_frac),
                "unsupervised_positive_labels": list(self.cfg.unsupervised_positive_labels),
            },
            "dataset": {
                "train_n": len(self.data.train_texts),
                "val_n": len(self.data.val_texts),
                "test_n": len(self.data.test_texts),
                "val_counts": {
                    "normal": int(np.sum(self.val_is_normal)),
                    "positive": int(np.sum(self.val_y)),
                },
                "test_counts": {
                    "normal": int(np.sum(self.test_is_normal)),
                    "borderline": int(np.sum(self.test_is_borderline)),
                    "malicious": int(np.sum(self.test_is_malicious)),
                },
            },
            "embeddings": {},
            "runs": [],
        }

        print(
            f"[runner] Unsupervised metrics positive labels: {self.cfg.unsupervised_positive_labels}"
        )

        # Keyword baseline
        if self.cfg.enable_keyword:
            patterns = self.cfg.keyword_patterns or DEFAULT_PATTERNS
            kw = KeywordBaseline(patterns)
            print(f"[runner] Keyword baseline patterns={len(patterns)} (calibrate on {self.cfg.threshold_calibration_set})")

            scores_val = kw.score_texts(self.data.val_texts)
            scores_test = kw.score_texts(self.data.test_texts)

            # calibration normals
            if self.cfg.threshold_calibration_set == "test":
                cal_normals = scores_test[self.test_is_normal]
            else:
                cal_normals = scores_val[self.val_is_normal]

            m = self._metrics_summary(self.test_y, scores_test)
            m_unsup = self._metrics_summary(self.unsup_test_y, scores_test)

            op: Dict[str, Any] = {}
            for fpr_target in self.cfg.fpr_points:
                thr = self._operating_threshold_from_normals(cal_normals, float(fpr_target))
                op[str(fpr_target)] = self._operating_point(fpr_target=float(fpr_target), thr=thr, scores_test=scores_test)

            results["runs"].append(
                {
                    "type": "keyword",
                    "embedding": None,
                    "detector": "keyword",
                    "detector_spec": {"patterns": patterns},
                    "metrics": m,
                    "metrics_unsup_labels": m_unsup,
                    "operating_points": op,
                    "latency_s": {"detector": None, "total": None},
                }
            )

        # Embeddings
        emb_bar = tqdm(self.cfg.embedding_models, desc="[runner] embeddings", unit="emb", leave=True)
        for emb_spec in emb_bar:
            emb_bar.set_postfix_str(emb_spec.name)
            print(f"\n[runner] Embedding: {emb_spec.name} ({emb_spec.kind})")

            X_train, dt_train = self._embed(emb_spec, self.data.train_texts)
            X_val, dt_val = self._embed(emb_spec, self.data.val_texts)
            X_test, dt_test = self._embed(emb_spec, self.data.test_texts)

            dt_total_emb = float(dt_train + dt_val + dt_test)
            results["embeddings"][emb_spec.name] = {
                "spec": emb_spec.to_dict(),
                "latency_s": {
                    "train": float(dt_train),
                    "val": float(dt_val),
                    "test": float(dt_test),
                    "total": float(dt_total_emb),
                },
                "shapes": {
                    "train": list(X_train.shape),
                    "val": list(X_val.shape),
                    "test": list(X_test.shape),
                },
            }

            # Unsupervised detectors: fit(train_normal_only) -> score(val/test)
            if self.cfg.enable_unsupervised and self.cfg.unsupervised_detectors:
                det_specs = self.cfg.unsupervised_detectors
                det_bar = tqdm(det_specs, desc=f"[runner] detectors (unsup) [{emb_spec.name}]", unit="det", leave=False)
                for spec in det_bar:
                    det = build_detector(spec)
                    det_bar.set_postfix_str(det.name)
                    print(f"[runner] Unsupervised detector={det.name}: fit(train) + score(val/test)")

                    t0 = time.time()
                    det.fit(X_train)
                    scores_val = det.score(X_val)
                    scores_test = det.score(X_test)
                    dt_det = time.time() - t0

                    # calibration normals
                    if self.cfg.threshold_calibration_set == "test":
                        cal_normals = scores_test[self.test_is_normal]
                    else:
                        cal_normals = scores_val[self.val_is_normal]

                    m = self._metrics_summary(self.test_y, scores_test)
                    m_unsup = self._metrics_summary(self.unsup_test_y, scores_test)

                    op: Dict[str, Any] = {}
                    for fpr_target in self.cfg.fpr_points:
                        thr = self._operating_threshold_from_normals(cal_normals, float(fpr_target))
                        op[str(fpr_target)] = self._operating_point(
                            fpr_target=float(fpr_target),
                            thr=thr,
                            scores_test=scores_test,
                        )

                    results["runs"].append(
                        {
                            "type": "embedding_unsupervised",
                            "embedding": emb_spec.name,
                            "detector": det.name,
                            "detector_spec": spec,
                            "metrics": m,
                            "metrics_unsup_labels": m_unsup,
                            "operating_points": op,
                            "latency_s": {
                                "detector": float(dt_det),
                                "total": float(dt_total_emb + dt_det),
                            },
                            "calibration": {
                                "set": self.cfg.threshold_calibration_set,
                                "n_normal": int(cal_normals.shape[0]),
                            },
                        }
                    )

            # Supervised detectors: fit(val_fit) -> score(val_cal/test); thresholds calibrated on val_cal normals
            if self.cfg.enable_supervised and self.cfg.supervised_detectors:
                det_specs = self.cfg.supervised_detectors
                det_bar = tqdm(det_specs, desc=f"[runner] detectors (sup) [{emb_spec.name}]", unit="det", leave=False)
                for spec in det_bar:
                    det = build_detector(spec)
                    det_bar.set_postfix_str(det.name)
                    print(f"[runner] Supervised detector={det.name}: fit(val_fit) + score(val_cal/test)")

                    t0 = time.time()
                    det.fit(X_val[self.sup_fit_idx], self.val_y[self.sup_fit_idx])
                    scores_val_cal = det.score(X_val[self.sup_cal_idx])
                    scores_test = det.score(X_test)
                    dt_det = time.time() - t0

                    # calibration normals from val_cal unless user forces test
                    if self.cfg.threshold_calibration_set == "test":
                        cal_normals = scores_test[self.test_is_normal]
                    else:
                        cal_normals = scores_val_cal[self.val_cal_is_normal]

                    m = self._metrics_summary(self.test_y, scores_test)

                    op: Dict[str, Any] = {}
                    for fpr_target in self.cfg.fpr_points:
                        thr = self._operating_threshold_from_normals(cal_normals, float(fpr_target))
                        op[str(fpr_target)] = self._operating_point(
                            fpr_target=float(fpr_target),
                            thr=thr,
                            scores_test=scores_test,
                        )

                    results["runs"].append(
                        {
                            "type": "embedding_supervised",
                            "embedding": emb_spec.name,
                            "detector": det.name,
                            "detector_spec": spec,
                            "metrics": m,
                            "operating_points": op,
                            "latency_s": {
                                "detector": float(dt_det),
                                "total": float(dt_total_emb + dt_det),
                            },
                            "calibration": {
                                "set": self.cfg.threshold_calibration_set,
                                "val_fit_n": int(self.sup_fit_idx.shape[0]),
                                "val_cal_n": int(self.sup_cal_idx.shape[0]),
                                "val_cal_normals": int(np.sum(self.val_cal_is_normal)),
                            },
                        }
                    )

        out_path = ensure_parent_dir(self.run_dir / "results.json")
        print(f"\n[runner] Writing results -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        figures_dir = self.run_dir / "figures"
        print(f"[runner] Writing figures -> {figures_dir}")
        write_all_figures(str(out_path), str(figures_dir))

        print("[runner] Done")
        return str(self.run_dir)
