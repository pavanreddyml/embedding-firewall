from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def plot_rocs(results_path: str, out_png: str, *, max_curves: int = 12) -> None:
    p = Path(results_path)
    with open(p, "r", encoding="utf-8") as f:
        res = json.load(f)

    runs = res.get("runs", [])
    curves = []
    for r in runs:
        m = r.get("metrics", {})
        roc = m.get("roc_curve")
        if not roc:
            continue
        label = r.get("name") or r.get("detector")
        emb = r.get("embedding", {}).get("name")
        if emb:
            label = f"{label}@{emb}"
        curves.append((label, roc["fpr"], roc["tpr"]))
    curves = curves[:max_curves]

    plt.figure()
    for label, fpr, tpr in curves:
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=7)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_operating_points(results_path: str, out_png: str, fpr_key: str = "0.05") -> None:
    p = Path(results_path)
    with open(p, "r", encoding="utf-8") as f:
        res = json.load(f)

    names = []
    tprs = []
    border = []

    for r in res.get("runs", []):
        op = r.get("operating_points", {}).get(fpr_key)
        if not op:
            continue
        label = r.get("name") or r.get("detector")
        emb = r.get("embedding", {}).get("name")
        if emb:
            label = f"{label}@{emb}"
        names.append(label)
        tprs.append(op.get("tpr_malicious", 0.0) or 0.0)
        border.append(op.get("border_block_rate", 0.0) or 0.0)

    if not names:
        return

    x = list(range(len(names)))
    plt.figure(figsize=(max(7, len(names) * 0.35), 4))
    plt.bar(x, tprs)
    plt.xticks(x, names, rotation=75, ha="right", fontsize=7)
    plt.ylabel("TPR on malicious")
    plt.title(f"TPR at {float(fpr_key)*100:.0f}% FPR (benign)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Borderline plot
    plt.figure(figsize=(max(7, len(names) * 0.35), 4))
    plt.bar(x, border)
    plt.xticks(x, names, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Block rate on borderline")
    plt.title(f"Borderline block rate at {float(fpr_key)*100:.0f}% FPR (benign)")
    out2 = str(Path(out_png).with_name(Path(out_png).stem + "_borderline.png"))
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
