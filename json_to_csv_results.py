from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON object must be a mapping")
    return data


def _flatten_operating_points(operating_points: Dict[str, Any] | None) -> Dict[str, Any]:
    if not operating_points:
        return {}
    flat: Dict[str, Any] = {}
    for key, vals in operating_points.items():
        prefix = f"op_{key}_"
        flat[prefix + "thr"] = vals.get("thr")
        flat[prefix + "test_fpr"] = vals.get("test_fpr")
        flat[prefix + "tpr_malicious"] = vals.get("tpr_malicious")
        flat[prefix + "borderline_block_rate"] = vals.get("borderline_block_rate")
    return flat


def _flatten_run(dataset_name: str | None, run: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "run_type": run.get("type"),
        "embedding": run.get("embedding"),
        "detector": run.get("detector"),
    }
    metrics = run.get("metrics") or {}
    record.update({f"metric_{k}": v for k, v in metrics.items()})
    metrics_unsup = run.get("metrics_unsup_labels") or {}
    record.update({f"metric_unsup_{k}": v for k, v in metrics_unsup.items()})
    op = _flatten_operating_points(run.get("operating_points"))
    record.update(op)
    latency = run.get("latency_s") or {}
    record.update({f"latency_{k}": v for k, v in latency.items()})
    tuning = run.get("tuning") or {}
    record.update({f"tuning_{k}": v for k, v in tuning.items()})
    calibration = run.get("calibration") or {}
    record.update({f"calibration_{k}": v for k, v in calibration.items()})
    return record


def _flatten_embedding(dataset_name: str | None, name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    rec = {"dataset": dataset_name, "embedding": name}
    spec = cfg.get("spec") or {}
    rec.update({f"spec_{k}": v for k, v in spec.items()})
    latency = cfg.get("latency_s") or {}
    rec.update({f"latency_{k}": v for k, v in latency.items()})
    shapes = cfg.get("shapes") or {}
    rec.update({f"shape_{k}": v for k, v in shapes.items()})
    return rec


def _flatten_dataset_meta(dataset_name: str | None, meta: Dict[str, Any] | None) -> Dict[str, Any]:
    meta = meta or {}
    rec = {"dataset": dataset_name}
    rec.update(meta)
    return rec


def _collect_records(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    runs: List[Dict[str, Any]] = []
    embeddings: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []

    if "datasets" in data:
        datasets_block = data.get("datasets") or {}
        for ds_name, ds_data in datasets_block.items():
            runs.extend(_flatten_run(ds_name, r) for r in ds_data.get("runs") or [])
            embeddings.extend(
                _flatten_embedding(ds_name, name, cfg) for name, cfg in (ds_data.get("embeddings") or {}).items()
            )
            datasets.append(_flatten_dataset_meta(ds_name, ds_data.get("dataset")))
    else:
        dataset_name = None
        dataset_meta = data.get("dataset")
        if isinstance(dataset_meta, dict):
            dataset_name = dataset_meta.get("name")
        elif isinstance(data.get("meta"), dict):
            dataset_name = data["meta"].get("dataset_name")
        runs.extend(_flatten_run(dataset_name, r) for r in data.get("runs") or [])
        embeddings.extend(
            _flatten_embedding(dataset_name, name, cfg) for name, cfg in (data.get("embeddings") or {}).items()
        )
        datasets.append(_flatten_dataset_meta(dataset_name, dataset_meta))

    return runs, embeddings, datasets


def _write_excel(out_path: Path, runs: List[Dict[str, Any]], embeddings: List[Dict[str, Any]], datasets: List[Dict[str, Any]]) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if runs:
            runs_df = pd.DataFrame(runs).sort_values(by=["dataset", "embedding", "detector", "run_type"], na_position="last")
            runs_df.to_excel(writer, sheet_name="runs", index=False)
        if embeddings:
            emb_df = pd.DataFrame(embeddings).sort_values(by=["dataset", "embedding"], na_position="last")
            emb_df.to_excel(writer, sheet_name="embeddings", index=False)
        if datasets:
            ds_df = pd.DataFrame(datasets).sort_values(by=["dataset"], na_position="last")
            ds_df.to_excel(writer, sheet_name="datasets", index=False)


def _write_csvs(csv_dir: Path, runs: List[Dict[str, Any]], embeddings: List[Dict[str, Any]], datasets: List[Dict[str, Any]]) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    if runs:
        pd.DataFrame(runs).sort_values(by=["dataset", "embedding", "detector", "run_type"], na_position="last").to_csv(
            csv_dir / "runs.csv", index=False
        )
    if embeddings:
        pd.DataFrame(embeddings).sort_values(by=["dataset", "embedding"], na_position="last").to_csv(
            csv_dir / "embeddings.csv", index=False
        )
    if datasets:
        pd.DataFrame(datasets).sort_values(by=["dataset"], na_position="last").to_csv(
            csv_dir / "datasets.csv", index=False
        )


def convert_results(json_path: str, *, excel_path: str | None = None, csv_dir: str | None = None) -> Path:
    src = Path(json_path)
    if not src.exists():
        raise FileNotFoundError(json_path)
    data = _load_json(src)

    runs, embeddings, datasets = _collect_records(data)

    out_excel = Path(excel_path) if excel_path else src.with_suffix(".xlsx")
    _write_excel(out_excel, runs, embeddings, datasets)

    if csv_dir:
        _write_csvs(Path(csv_dir), runs, embeddings, datasets)

    return out_excel


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert results JSON to Excel/CSV summaries")
    parser.add_argument("json_path", help="Path to results.json")
    parser.add_argument("--excel", dest="excel_path", default=None, help="Optional output Excel file path")
    parser.add_argument("--csv-dir", dest="csv_dir", default=None, help="Optional directory for CSV outputs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = convert_results(args.json_path, excel_path=args.excel_path, csv_dir=args.csv_dir)
    print(f"Wrote summaries to {out_path}")
    if args.csv_dir:
        print(f"CSV files saved under {args.csv_dir}")


if __name__ == "__main__":
    main()
