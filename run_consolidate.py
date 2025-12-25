from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from embfirewall.viz import write_all_figures


def _in_colab() -> bool:
    try:
        import importlib

        return importlib.util.find_spec("google.colab") is not None
    except Exception:
        return False


IN_COLAB = _in_colab()
RUN_ID = "demo_run"

LOCAL_BASE_DIR = "."
COLAB_BASE_DIR = "/content/drive/MyDrive/research/embfirewall"

WORKING_DIR = LOCAL_BASE_DIR
STORAGE_DIR = COLAB_BASE_DIR if IN_COLAB else LOCAL_BASE_DIR

RUNS_DIR = str(Path(STORAGE_DIR) / "runs")


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


def _dataset_dirs(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    return [p for p in sorted(run_dir.iterdir()) if p.is_dir() and p.name != "consolidate"]


def _path_config(run_id: str, storage_dir: str) -> dict[str, Path]:
    runs_dir = Path(storage_dir) / "runs"
    run_dir = runs_dir / run_id
    consolidate_dir = run_dir / "consolidate_all"
    consolidated_results_path = consolidate_dir / "results.json"
    figures_dir = consolidate_dir / "figures"

    return {
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "consolidate_dir": consolidate_dir,
        "consolidated_results_path": consolidated_results_path,
        "figures_dir": figures_dir,
    }


def consolidate_runs(run_id: str = RUN_ID, *, storage_dir: str = STORAGE_DIR) -> None:
    paths = _path_config(run_id, storage_dir)
    run_dir = paths["run_dir"]
    consolidate_dir = paths["consolidate_dir"]
    consolidated_results_path = paths["consolidated_results_path"]
    figures_dir = paths["figures_dir"]

    print(f"[consolidate] IN_COLAB={IN_COLAB}")
    print(f"[consolidate] WORKING_DIR={Path(WORKING_DIR).resolve()}")
    print(f"[consolidate] RUN_DIR={run_dir}")
    print(f"[consolidate] OUTPUT={consolidated_results_path}")

    dataset_dirs = _dataset_dirs(run_dir)
    if not dataset_dirs:
        raise SystemExit(f"[consolidate] No dataset directories found under {run_dir}")

    consolidate_dir.mkdir(parents=True, exist_ok=True)

    all_runs: List[Dict[str, Any]] = []
    all_embeddings: Dict[str, Any] = {}
    dataset_results: Dict[str, Any] = {}

    for ds_dir in dataset_dirs:
        dataset_name = ds_dir.name
        res_files = _result_files(ds_dir)
        if not res_files:
            print(f"[consolidate][warn] No results.json files found under {ds_dir}, skipping")
            continue

        combined: Dict[str, Any] | None = None
        datasets_meta: List[Any] = []
        source_dirs: List[str] = []

        for res_path in res_files:
            data = _load_results(res_path)
            source_dirs.append(str(res_path.parent))
            datasets_meta.append(data.get("dataset"))

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

        if combined is None:
            continue

        meta = combined.setdefault("meta", {})
        meta["run_id"] = run_id
        meta["dataset_name"] = dataset_name
        meta["consolidated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["source_run_dirs"] = source_dirs

        dataset_results[dataset_name] = combined
        all_runs.extend([{**r, "dataset_name": dataset_name} for r in combined.get("runs") or []])
        all_embeddings.update(combined.get("embeddings") or {})

        ds_consolidate_dir = ds_dir / "consolidate"
        ds_consolidate_dir.mkdir(parents=True, exist_ok=True)
        ds_results_path = ds_consolidate_dir / "results.json"
        ds_figures_dir = ds_consolidate_dir / "figures"

        with open(ds_results_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"[consolidate] Wrote {dataset_name} results -> {ds_results_path}")

        write_all_figures(str(ds_results_path), str(ds_figures_dir))
        print(f"[consolidate] Wrote {dataset_name} figures -> {ds_figures_dir}")

    if not dataset_results:
        raise SystemExit(f"[consolidate] No datasets consolidated under {run_dir}")

    all_meta = {
        "run_id": run_id,
        "consolidated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": sorted(dataset_results.keys()),
    }

    combined_all = {
        "meta": all_meta,
        "datasets": dataset_results,
        "runs": all_runs,
        "embeddings": all_embeddings,
    }

    with open(consolidated_results_path, "w", encoding="utf-8") as f:
        json.dump(combined_all, f, ensure_ascii=False, indent=2)
    print(f"[consolidate] Wrote cross-dataset summary -> {consolidated_results_path}")

    write_all_figures(str(consolidated_results_path), str(figures_dir))
    print(f"[consolidate] Wrote cross-dataset figures -> {figures_dir}")


if __name__ == "__main__":
    consolidate_runs()
