# file: embfirewall/viz.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _run_label(r: Dict[str, Any]) -> str:
    det = r.get("detector") or r.get("name") or "unknown"
    emb = r.get("embedding")
    if isinstance(emb, str) and emb:
        return f"{det}@{emb}"
    return str(det)


def _finite(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except Exception:
        return None


def _load_results(results_path: str) -> Dict[str, Any]:
    with open(Path(results_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in res.get("runs", []) or []:
        metrics = r.get("metrics") or {}
        op = r.get("operating_points") or {}

        row: Dict[str, Any] = {
            "label": _run_label(r),
            "type": r.get("type"),
            "embedding": r.get("embedding") if isinstance(r.get("embedding"), str) else None,
            "detector": r.get("detector") or r.get("name"),
            "auroc": _finite(metrics.get("auroc")),
            "auprc": _finite(metrics.get("auprc")),
            "latency_total_s": _finite((r.get("latency_s") or {}).get("total")),
            "latency_detector_s": _finite((r.get("latency_s") or {}).get("detector")),
        }

        # operating points (flatten)
        if isinstance(op, dict):
            for k, v in op.items():
                if not isinstance(v, dict):
                    continue
                row[f"thr@{k}"] = _finite(v.get("thr"))
                row[f"tpr_malicious@{k}"] = _finite(v.get("tpr_malicious"))
                row[f"border_block_rate@{k}"] = _finite(v.get("border_block_rate"))

        rows.append(row)
    return rows


def write_summary_csv(results_path: str, out_csv: str) -> None:
    res = _load_results(results_path)
    rows = _collect_rows(res)
    if not rows:
        return

    # stable column order
    cols = ["label", "type", "embedding", "detector", "auroc", "auprc", "latency_total_s", "latency_detector_s"]
    extra_cols = sorted({k for row in rows for k in row.keys() if k not in cols})
    cols = cols + extra_cols

    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _topk(rows: List[Dict[str, Any]], key: str, k: int) -> List[Dict[str, Any]]:
    xs = [r for r in rows if r.get(key) is not None]
    xs.sort(key=lambda r: float(r[key]), reverse=True)
    return xs[:k]


def plot_metric_bar(results_path: str, out_png: str, metric: str, *, top_k: int = 25) -> None:
    res = _load_results(results_path)
    rows = _collect_rows(res)
    rows = _topk(rows, metric, top_k)

    if not rows:
        return

    labels = [r["label"] for r in rows]
    vals = [float(r[metric]) for r in rows]

    x = list(range(len(labels)))
    plt.figure(figsize=(max(7, len(labels) * 0.35), 4))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel(metric.upper())
    plt.title(f"Top-{len(labels)} by {metric.upper()}")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_operating_points(results_path: str, out_dir: str, fpr_key: str) -> None:
    res = _load_results(results_path)
    rows = _collect_rows(res)

    tpr_key = f"tpr_malicious@{fpr_key}"
    bdr_key = f"border_block_rate@{fpr_key}"

    tpr_rows = [r for r in rows if r.get(tpr_key) is not None]
    if not tpr_rows:
        return
    tpr_rows.sort(key=lambda r: float(r[tpr_key]), reverse=True)

    labels = [r["label"] for r in tpr_rows]
    tprs = [float(r[tpr_key]) for r in tpr_rows]
    bdrs = [float(r[bdr_key]) if r.get(bdr_key) is not None else 0.0 for r in tpr_rows]

    x = list(range(len(labels)))
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # TPR bar
    plt.figure(figsize=(max(7, len(labels) * 0.35), 4))
    plt.bar(x, tprs)
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("TPR on malicious")
    plt.title(f"TPR at {float(fpr_key)*100:.0f}% FPR (benign)")
    plt.savefig(str(out_dir_p / f"tpr_at_fpr_{fpr_key}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Borderline bar
    plt.figure(figsize=(max(7, len(labels) * 0.35), 4))
    plt.bar(x, bdrs)
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Block rate on borderline")
    plt.title(f"Borderline block rate at {float(fpr_key)*100:.0f}% FPR (benign)")
    plt.savefig(str(out_dir_p / f"border_at_fpr_{fpr_key}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_tpr_vs_border_scatter(results_path: str, out_png: str, fpr_key: str) -> None:
    res = _load_results(results_path)
    rows = _collect_rows(res)

    tpr_key = f"tpr_malicious@{fpr_key}"
    bdr_key = f"border_block_rate@{fpr_key}"

    pts: List[Tuple[float, float, str]] = []
    for r in rows:
        tpr = r.get(tpr_key)
        bdr = r.get(bdr_key)
        if tpr is None or bdr is None:
            continue
        pts.append((float(bdr), float(tpr), r["label"]))

    if not pts:
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Borderline block rate")
    plt.ylabel("TPR on malicious")
    plt.title(f"TPR vs Borderline block @ {float(fpr_key)*100:.0f}% FPR (benign)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_latency_scatter(results_path: str, out_png: str, metric: str = "auroc") -> None:
    res = _load_results(results_path)
    rows = _collect_rows(res)

    pts: List[Tuple[float, float, str]] = []
    for r in rows:
        lat = r.get("latency_total_s") or r.get("latency_detector_s")
        m = r.get(metric)
        if lat is None or m is None:
            continue
        pts.append((float(lat), float(m), r["label"]))

    if not pts:
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Latency (s)")
    plt.ylabel(metric.upper())
    plt.title(f"Latency vs {metric.upper()}")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_box_by_embedding(results_path: str, out_png: str, metric: str = "auroc") -> None:
    res = _load_results(results_path)
    rows = _collect_rows(res)

    by_emb: Dict[str, List[float]] = {}
    for r in rows:
        emb = r.get("embedding") or "keyword"
        v = r.get(metric)
        if v is None:
            continue
        by_emb.setdefault(str(emb), []).append(float(v))

    items = [(k, v) for k, v in by_emb.items() if v]
    if not items:
        return
    items.sort(key=lambda kv: sum(kv[1]) / max(1, len(kv[1])), reverse=True)

    labels = [k for k, _ in items]
    data = [v for _, v in items]

    plt.figure(figsize=(max(7, len(labels) * 0.5), 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} by embedding (boxplot)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def write_all_figures(results_path: str, figures_dir: str) -> None:
    figdir = Path(figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    # Paper-ish essentials
    plot_metric_bar(results_path, str(figdir / "auroc_top.png"), "auroc", top_k=25)
    plot_metric_bar(results_path, str(figdir / "auprc_top.png"), "auprc", top_k=25)

    for k in ["0.05", "0.10"]:
        plot_operating_points(results_path, str(figdir), fpr_key=k)
        plot_tpr_vs_border_scatter(results_path, str(figdir / f"tpr_vs_border_fpr_{k}.png"), fpr_key=k)

    plot_latency_scatter(results_path, str(figdir / "latency_vs_auroc.png"), metric="auroc")
    plot_latency_scatter(results_path, str(figdir / "latency_vs_auprc.png"), metric="auprc")

    plot_metric_box_by_embedding(results_path, str(figdir / "auroc_by_embedding_box.png"), metric="auroc")
    plot_metric_box_by_embedding(results_path, str(figdir / "auprc_by_embedding_box.png"), metric="auprc")

    write_summary_csv(results_path, str(figdir / "summary.csv"))
