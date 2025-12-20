# file: hypothesis.py
"""
Diagnostic script to probe why unsupervised AUROC can look bad when supervised models do well.

It reuses the same data loading, embedding, and detector construction code as `run_eval.py`,
but reports AUROC under two label definitions:

1) Positive = malicious only (the default in the runner).
2) Positive = malicious OR borderline (treat borderline as "attack-like"), which tests the
   hypothesis that unsupervised detectors are penalized because they flag borderline rows.

Usage (defaults match run_eval.py paths):

    python hypothesis.py

You can override the paths with environment variables:

- HYPOTHESIS_EVAL_CONFIG (default: configs/eval_config.yaml)
- HYPOTHESIS_DATA_DIR (default: ./data)
- HYPOTHESIS_RUN_DIR (default: ./runs/hypothesis)

The script prints per-detector AUROC for both label definitions and basic score statistics
for each label group. It does not write figures or JSON results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from embfirewall.detectors.factory import build_detector
from embfirewall.runner import DatasetSlices, ExperimentRunner, RunConfig
from run_eval import (
    DATA_DIR,
    EVAL_CONFIG_PATH,
    _list_dataset_dirs,
    _load_eval_config,
    _load_test,
    _load_train_normal,
    _load_val,
    _parse_embeddings,
)


def _metrics_pair(
    *,
    scores: np.ndarray,
    y_malicious_only: np.ndarray,
    y_mal_or_borderline: np.ndarray,
) -> Tuple[Dict[str, float | None], Dict[str, float | None]]:
    """Compute AUROC/AUPRC for two label definitions using the runner's metric helper."""

    m1 = ExperimentRunner._metrics_summary(y_malicious_only, scores)
    m2 = ExperimentRunner._metrics_summary(y_mal_or_borderline, scores)
    return m1, m2


def _group_stats(scores: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "p99": float("nan")}
    sel = scores[mask]
    if sel.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "mean": float(np.mean(sel)),
        "p95": float(np.percentile(sel, 95)),
        "p99": float(np.percentile(sel, 99)),
    }


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _load_dataset(eval_cfg: dict, data_dir: str, dataset_name: str) -> Tuple[DatasetSlices, Dict[str, str | int]]:
    ds_cfg = eval_cfg.get("dataset") or {}
    seed = int(ds_cfg.get("seed", 7))
    max_train_normal = ds_cfg.get("max_train_normal", 20000)
    max_val_total = ds_cfg.get("max_val_total", 20000)
    max_test_total = ds_cfg.get("max_test_total", 60000)
    max_chars = ds_cfg.get("max_chars", 10000)

    max_train_normal_i = int(max_train_normal) if max_train_normal is not None else None
    max_val_total_i = int(max_val_total) if max_val_total is not None else None
    max_test_total_i = int(max_test_total) if max_test_total is not None else None
    max_chars_i = int(max_chars) if max_chars is not None else None

    labels_cfg = eval_cfg.get("labels") or {}
    normal_label = str(labels_cfg.get("normal_label", "normal"))
    borderline_label = str(labels_cfg.get("borderline_label", "borderline"))
    malicious_label = str(labels_cfg.get("malicious_label", "malicious"))

    train_texts = _load_train_normal(
        data_dir,
        seed=seed,
        cap=max_train_normal_i,
        max_chars=max_chars_i,
        normal_label=normal_label,
    )

    val_texts, val_labels = _load_val(
        data_dir,
        seed=seed + 1,
        cap=max_val_total_i,
        max_chars=max_chars_i,
        normal_label=normal_label,
        malicious_label=malicious_label,
    )

    test_texts, test_labels = _load_test(
        data_dir,
        seed=seed + 2,
        cap=max_test_total_i,
        max_chars=max_chars_i,
        normal_label=normal_label,
        borderline_label=borderline_label,
        malicious_label=malicious_label,
    )

    data = DatasetSlices(
        train_texts=train_texts,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )

    meta = {
        "normal_label": normal_label,
        "borderline_label": borderline_label,
        "malicious_label": malicious_label,
        "seed": seed,
        "max_train_normal": max_train_normal,
        "max_val_total": max_val_total,
        "max_test_total": max_test_total,
        "max_chars": max_chars,
        "dataset_name": dataset_name,
    }

    return data, meta


def _build_runner(eval_cfg: dict, data_dir: str, run_dir: str, dataset_name: str) -> Tuple[ExperimentRunner, Dict]:
    data, meta = _load_dataset(eval_cfg, data_dir, dataset_name)

    fpr_points = eval_cfg.get("fpr_points", [0.05, 0.10])
    fpr_points_t = tuple(float(x) for x in fpr_points)

    kw_cfg = eval_cfg.get("keyword_baseline") or {}
    enable_keyword = bool(kw_cfg.get("enabled", True))
    keyword_patterns = kw_cfg.get("patterns")

    det_cfg = eval_cfg.get("detectors") or {}
    enable_unsup = bool(det_cfg.get("enable_unsupervised", True))
    enable_sup = bool(det_cfg.get("enable_supervised", True))
    unsup_list = det_cfg.get("unsupervised")
    sup_list = det_cfg.get("supervised")

    embeddings = _parse_embeddings(eval_cfg)

    cfg = RunConfig(
        run_dir=run_dir,
        normal_label=meta["normal_label"],
        borderline_label=meta["borderline_label"],
        malicious_label=meta["malicious_label"],
        fpr_points=fpr_points_t,  # type: ignore[arg-type]
        embedding_models=embeddings,
        enable_keyword=enable_keyword,
        enable_unsupervised=enable_unsup,
        enable_supervised=enable_sup,
        unsupervised_detectors=(list(unsup_list) if isinstance(unsup_list, list) else None),
        supervised_detectors=(list(sup_list) if isinstance(sup_list, list) else None),
        keyword_patterns=(list(keyword_patterns) if isinstance(keyword_patterns, list) else None),
        dataset_name=dataset_name,
        unsupervised_positive_labels=(meta["malicious_label"], meta["borderline_label"]),
    )

    runner = ExperimentRunner(cfg, data)
    return runner, eval_cfg


def _summarize_detector(
    *,
    det_name: str,
    scores_test: np.ndarray,
    runner: ExperimentRunner,
) -> None:
    y_mal_only = runner.test_y
    y_mal_or_bl = np.where(runner.test_is_malicious | runner.test_is_borderline, 1, 0)
    y_normal = np.where(runner.test_is_normal, 1, 0)
    scores_normal = -scores_test

    m1, m2 = _metrics_pair(
        scores=scores_test,
        y_malicious_only=y_mal_only,
        y_mal_or_borderline=y_mal_or_bl,
    )
    m_norm = ExperimentRunner._metrics_summary(y_normal, scores_normal)

    stats_norm = _group_stats(scores_test, runner.test_is_normal)
    stats_bl = _group_stats(scores_test, runner.test_is_borderline)
    stats_mal = _group_stats(scores_test, runner.test_is_malicious)

    print(f"\n{det_name}")
    print("  AUROC (malicious only +):   ", m1.get("auroc"))
    print("  AUROC (malicious + borderline +):", m2.get("auroc"))
    print("  AUPRC (malicious only +):   ", m1.get("auprc"))
    print("  AUPRC (malicious + borderline +):", m2.get("auprc"))
    print(
        "  AUROC (normal-only +, inverted scores):",
        m_norm.get("auroc"),
        f"[normals={int(np.sum(y_normal))}]",
    )
    print("  Score means/p95/p99 by label:")
    print(f"    normal     mean={stats_norm['mean']:.4f} p95={stats_norm['p95']:.4f} p99={stats_norm['p99']:.4f}")
    print(f"    borderline mean={stats_bl['mean']:.4f} p95={stats_bl['p95']:.4f} p99={stats_bl['p99']:.4f}")
    print(f"    malicious  mean={stats_mal['mean']:.4f} p95={stats_mal['p95']:.4f} p99={stats_mal['p99']:.4f}")


def run_diagnostic(eval_config: str, data_dir: str, run_dir: str) -> None:
    eval_cfg = _load_eval_config(eval_config)

    labels_cfg = eval_cfg.get("labels") or {}
    normal_label = str(labels_cfg.get("normal_label", "normal"))
    borderline_label = str(labels_cfg.get("borderline_label", "borderline"))
    malicious_label = str(labels_cfg.get("malicious_label", "malicious"))
    labels_tuple = (normal_label, borderline_label, malicious_label)

    dataset_dirs = _list_dataset_dirs(Path(data_dir), labels_tuple)
    if not dataset_dirs:
        raise SystemExit(f"[hypothesis] No dataset folders found under {data_dir}")

    for dataset_name, dataset_dir in dataset_dirs:
        runner, _ = _build_runner(eval_cfg, str(dataset_dir), str(Path(run_dir) / dataset_name), dataset_name)

        _print_header(f"Dataset: {dataset_name}")
        print(
            f"train={len(runner.data.train_texts)} val={len(runner.data.val_texts)} test={len(runner.data.test_texts)}"
        )
        print(
            "val counts:",
            {
                "normal": int(np.sum(runner.val_is_normal)),
                "malicious": int(np.sum(runner.val_y)),
            },
        )
        print(
            "test counts:",
            {
                "normal": int(np.sum(runner.test_is_normal)),
                "borderline": int(np.sum(runner.test_is_borderline)),
                "malicious": int(np.sum(runner.test_is_malicious)),
            },
        )

        for emb_spec in runner.cfg.embedding_models:
            _print_header(f"Embedding: {emb_spec.name} ({emb_spec.model_id}) [{dataset_name}]")
            X_train, _ = runner._embed(emb_spec, runner.data.train_texts)
            X_val, _ = runner._embed(emb_spec, runner.data.val_texts)
            X_test, _ = runner._embed(emb_spec, runner.data.test_texts)

            if runner.cfg.enable_unsupervised and runner.cfg.unsupervised_detectors:
                print("\n[unsupervised]")
                for spec in runner.cfg.unsupervised_detectors:
                    det = build_detector(spec)
                    det.fit(X_train)
                    scores_test = det.score(X_test)
                    _summarize_detector(det_name=det.name, scores_test=scores_test, runner=runner)

            if runner.cfg.enable_supervised and runner.cfg.supervised_detectors:
                print("\n[supervised]")
                for spec in runner.cfg.supervised_detectors:
                    det = build_detector(spec)
                    det.fit(X_val[runner.sup_fit_idx], runner.val_y[runner.sup_fit_idx])
                    scores_test = det.score(X_test)
                    _summarize_detector(det_name=det.name, scores_test=scores_test, runner=runner)


def _env_or_default(env_key: str, default: str) -> str:
    val = os.environ.get(env_key)
    return val if val else default


if __name__ == "__main__":
    eval_config = _env_or_default("HYPOTHESIS_EVAL_CONFIG", EVAL_CONFIG_PATH)
    data_dir = _env_or_default("HYPOTHESIS_DATA_DIR", DATA_DIR)
    run_dir = _env_or_default("HYPOTHESIS_RUN_DIR", str(Path("runs") / "hypothesis"))
    run_diagnostic(eval_config=str(eval_config), data_dir=str(data_dir), run_dir=str(run_dir))
