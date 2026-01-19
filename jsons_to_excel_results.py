from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from json_to_csv_results import _collect_records, _load_json, _write_excel


def _expand_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        candidate = Path(raw)
        if any(char in raw for char in ["*", "?", "["]):
            paths.extend(Path().glob(raw))
        elif candidate.is_dir():
            paths.extend(candidate.rglob("*.json"))
        else:
            paths.append(candidate)
    return sorted({path.resolve() for path in paths})


def _collect_with_source(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    data = _load_json(path)
    runs, embeddings, datasets = _collect_records(data)
    source = path.as_posix()
    for record in runs:
        record["source_file"] = source
    for record in embeddings:
        record["source_file"] = source
    for record in datasets:
        record["source_file"] = source
    return runs, embeddings, datasets


def convert_many(json_inputs: Sequence[str], output_excel: str) -> Path:
    paths = _expand_inputs(json_inputs)
    if not paths:
        raise FileNotFoundError("No JSON files found from provided inputs")

    all_runs: List[dict] = []
    all_embeddings: List[dict] = []
    all_datasets: List[dict] = []

    for path in paths:
        runs, embeddings, datasets = _collect_with_source(path)
        all_runs.extend(runs)
        all_embeddings.extend(embeddings)
        all_datasets.extend(datasets)

    out_path = Path(output_excel)
    _write_excel(out_path, all_runs, all_embeddings, all_datasets)
    return out_path


def _parse_args(argv: Iterable[str]) -> Tuple[List[str], str]:
    args = list(argv)
    if not args:
        raise SystemExit(
            "Usage: python jsons_to_excel_results.py <json inputs...> [--output <path>]"
        )

    output = "results_summary.xlsx"
    if "--output" in args:
        output_index = args.index("--output")
        if output_index == len(args) - 1:
            raise SystemExit("Missing value after --output")
        output = args[output_index + 1]
        del args[output_index : output_index + 2]

    if not args:
        raise SystemExit("Provide at least one JSON input path, directory, or glob")
    return args, output


def main(argv: Iterable[str] | None = None) -> None:
    json_inputs, output = _parse_args(argv if argv is not None else sys.argv[1:])
    out_path = convert_many(json_inputs, output)
    print(f"Wrote merged summaries to {out_path}")


if __name__ == "__main__":
    main()
