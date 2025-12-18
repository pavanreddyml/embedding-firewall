# file: run_consolidate.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from embfirewall.viz import write_all_figures


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


# -----------------------------
# GLOBAL PATHS (edit this file)
# -----------------------------
IN_COLAB = _in_colab()

# Run identifier to consolidate. Should match the RUN_ID used in run_eval.py
# across parallel notebooks.
RUN_ID = "demo_run"

LOCAL_BASE_DIR = "."
COLAB_BASE_DIR = "/content/drive/MyDrive/research/embfirewall"  # <-- change to your folder on Drive

WORKING_DIR = LOCAL_BASE_DIR
STORAGE_DIR = COLAB_BASE_DIR if IN_COLAB else LOCAL_BASE_DIR

RUNS_DIR = str(Path(STORAGE_DIR) / "runs")
RUN_DIR = Path(RUNS_DIR) / RUN_ID
CONSOLIDATE_DIR = RUN_DIR / "consolidate"
CONSOLIDATED_RESULTS_PATH = CONSOLIDATE_DIR / "results.json"
FIGURES_DIR = CONSOLIDATE_DIR / "figures"
# -----------------------------


def _load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _result_files(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []

    out: List[Path] = []
    for p in sorted(run_dir.iterdir()):
        if p.name == "consolidate":
            continue
        if not p.is_dir():
            continue
        res = p / "results.json"
        if res.exists():
            out.append(res)
    return out


def consolidate_runs(run_dir: Path = RUN_DIR) -> None:
    print(f"[consolidate] IN_COLAB={IN_COLAB}")
    print(f"[consolidate] WORKING_DIR={Path(WORKING_DIR).resolve()}")
    print(f"[consolidate] RUN_DIR={run_dir}")
    print(f"[consolidate] OUTPUT={CONSOLIDATED_RESULTS_PATH}")

    res_files = _result_files(run_dir)
    if not res_files:
        raise SystemExit(f"[consolidate] No results.json files found under {run_dir}")

    combined: Dict[str, Any] | None = None
    source_dirs: List[str] = []

    for res_path in res_files:
        data = _load_results(res_path)
        source_dirs.append(str(res_path.parent))

        if combined is None:
            combined = {
                "meta": dict(data.get("meta") or {}),
                "dataset": data.get("dataset"),
                "embeddings": dict(data.get("embeddings") or {}),
                "runs": list(data.get("runs") or []),
            }
        else:
            combined.setdefault("embeddings", {}).update(data.get("embeddings") or {})
            combined.setdefault("runs", []).extend(data.get("runs") or [])

    assert combined is not None
    meta = combined.setdefault("meta", {})
    meta["consolidated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["source_run_dirs"] = source_dirs

    CONSOLIDATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONSOLIDATED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"[consolidate] Wrote combined results -> {CONSOLIDATED_RESULTS_PATH}")

    write_all_figures(str(CONSOLIDATED_RESULTS_PATH), str(FIGURES_DIR))
    print(f"[consolidate] Wrote figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    consolidate_runs()
