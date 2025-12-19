# file: run_eval.py
from __future__ import annotations

import time
import shutil
from pathlib import Path

import yaml

from embfirewall.data.loaders import _list_label_files, interleave_label_files
from embfirewall.embeddings import EmbeddingSpec
from embfirewall.runner import DatasetSlices, ExperimentRunner, RunConfig


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

# Unique identifier for this experiment run. Set the same RUN_ID across
# multiple notebooks to merge results later.
RUN_ID = "demo_run"

# Optional: limit which embedding model_ids to run. Leave empty to run all
# embeddings from configs/eval_config.yaml.
# Available model_ids are printed at runtime.
RUN_MODEL_IDS: list[str] = []

LOCAL_BASE_DIR = "."
COLAB_BASE_DIR = "/content/drive/MyDrive/research/embfirewall"  # <-- change to your folder on Drive

WORKING_DIR = LOCAL_BASE_DIR
STORAGE_DIR = COLAB_BASE_DIR if IN_COLAB else LOCAL_BASE_DIR

DATA_DIR = str(Path(STORAGE_DIR) / "data")
RUNS_DIR = str(Path(STORAGE_DIR) / "runs")
EVAL_CONFIG_PATH = str(Path(WORKING_DIR) / "configs" / "eval_config.yaml")
# -----------------------------


def _counts(ls: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in ls:
        d[x] = d.get(x, 0) + 1
    return d


def _load_eval_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[run] Missing eval config: {path}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"[run] Invalid eval config (not a mapping): {path}")
    return cfg


def _parse_embeddings(cfg: dict) -> list[EmbeddingSpec]:
    embs = cfg.get("embeddings") or []
    if not isinstance(embs, list) or not embs:
        raise SystemExit("[run] eval_config.yaml: embeddings must be a non-empty list")

    out: list[EmbeddingSpec] = []
    for e in embs:
        if not isinstance(e, dict):
            raise SystemExit(f"[run] embeddings entry must be mapping, got: {e}")

        out.append(
            EmbeddingSpec(
                kind=str(e["kind"]),
                name=str(e["name"]),
                model_id=str(e["model_id"]),
                batch_size=int(e.get("batch_size", 64)),
                normalize=bool(e.get("normalize", True)),
                device=str(e.get("device", "cpu")),
                trust_remote_code=bool(e.get("trust_remote_code", True)),
                dimensions=(int(e["dimensions"]) if "dimensions" in e and e["dimensions"] is not None else None),
                openai_api_key_env=str(e.get("openai_api_key_env", "OPENAI_API_KEY")),
                openai_base_url=e.get("openai_base_url"),
                openai_organization=e.get("openai_organization"),
                openai_project=e.get("openai_project"),
                ollama_base_url=e.get("ollama_base_url", "http://localhost:11434"),
                ollama_request_timeout=float(e.get("ollama_request_timeout", 120.0)),
            )
        )
    return out


def _list_dataset_dirs(data_root: Path, labels: tuple[str, str, str]) -> list[tuple[str, Path]]:
    if not data_root.exists():
        return []

    out: list[tuple[str, Path]] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        # Accept either flat label shards (<label>-00000.json) or per-label subfolders
        # (<label>/<label>-00000.json) as produced by run_download_data.py.
        if all(_list_label_files(p, lab) for lab in labels):
            out.append((p.name, p))
    return out


def _load_train_normal(
    data_dir: str,
    seed: int,
    cap: int | None,
    *,
    max_chars: int | None,
    normal_label: str,
) -> list[str]:
    texts, _labels = interleave_label_files(
        data_dir,
        labels=(normal_label,),
        per_label_cap=cap,
        total_cap=cap,
        seed=seed,
        max_chars=max_chars,
        show_progress=True,
        desc="train_load[normal]",
    )
    return texts


def _load_val(
    data_dir: str,
    seed: int,
    cap: int | None,
    *,
    max_chars: int | None,
    normal_label: str,
    malicious_label: str,
) -> tuple[list[str], list[str]]:
    return interleave_label_files(
        data_dir,
        labels=(normal_label, malicious_label),
        per_label_cap=None,
        total_cap=cap,
        seed=seed,
        max_chars=max_chars,
        show_progress=True,
        desc="val_mix[normal+malicious]",
    )


def _load_test(
    data_dir: str,
    seed: int,
    cap: int | None,
    *,
    max_chars: int | None,
    normal_label: str,
    borderline_label: str,
    malicious_label: str,
) -> tuple[list[str], list[str]]:
    return interleave_label_files(
        data_dir,
        labels=(normal_label, borderline_label, malicious_label),
        per_label_cap=None,
        total_cap=cap,
        seed=seed,
        max_chars=max_chars,
        show_progress=True,
        desc="test_mix[normal+borderline+malicious]",
    )


def run_eval(
    eval_config: str = EVAL_CONFIG_PATH,
    runs_dir: str = RUNS_DIR,
    data_dir: str = DATA_DIR,
) -> None:
    """Run an embedding firewall experiment.

    Parameters mirror the previous command-line options so callers can invoke
    this module programmatically.
    """

    runs_dir_path = Path(runs_dir)
    data_dir_path = Path(data_dir)
    eval_config_path = Path(eval_config)

    print(f"[run] IN_COLAB={IN_COLAB}")
    print(f"[run] WORKING_DIR={Path(WORKING_DIR).resolve()}")
    print(f"[run] DATA_DIR={data_dir_path}")
    print(f"[run] RUNS_DIR={runs_dir_path}")
    print(f"[run] EVAL_CONFIG_PATH={eval_config_path}")

    runs_dir_path.mkdir(parents=True, exist_ok=True)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    eval_cfg = _load_eval_config(str(eval_config_path))

    if not RUN_ID:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    else:
        run_id = RUN_ID

    # dataset settings from eval_config.yaml
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

    print(
        f"[run] dataset: seed={seed} "
        f"max_train_normal={max_train_normal_i} max_val_total={max_val_total_i} max_test_total={max_test_total_i} max_chars={max_chars_i}"
    )

    labels_cfg = eval_cfg.get("labels") or {}
    normal_label = str(labels_cfg.get("normal_label", "normal"))
    borderline_label = str(labels_cfg.get("borderline_label", "borderline"))
    malicious_label = str(labels_cfg.get("malicious_label", "malicious"))

    labels_tuple = (normal_label, borderline_label, malicious_label)

    dataset_dirs = _list_dataset_dirs(data_dir_path, labels_tuple)
    if not dataset_dirs:
        raise SystemExit(f"[run] No dataset folders found under {data_dir_path}; expected subdirs with label shards")

    run_root = Path(runs_dir_path) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] run_root={run_root}")

    embeddings_all = _parse_embeddings(eval_cfg)
    available_model_ids = [e.model_id for e in embeddings_all]
    print(f"[run] available model_ids: {available_model_ids}")

    if RUN_MODEL_IDS:
        targets = set(RUN_MODEL_IDS)
        embeddings = [e for e in embeddings_all if e.model_id in targets]
        missing = sorted(targets - {e.model_id for e in embeddings})
        if missing:
            raise SystemExit(f"[run] Unknown model_ids requested via RUN_MODEL_IDS: {missing}")
        print(f"[run] filtered embeddings via RUN_MODEL_IDS -> {len(embeddings)} models")
    else:
        embeddings = embeddings_all

    if not embeddings:
        raise SystemExit("[run] No embeddings selected to run")

    fpr_points = eval_cfg.get("fpr_points", [0.05, 0.10])
    if not isinstance(fpr_points, list) or not fpr_points:
        raise SystemExit("[run] eval_config.yaml: fpr_points must be a non-empty list")
    fpr_points_t = tuple(float(x) for x in fpr_points)

    kw_cfg = eval_cfg.get("keyword_baseline") or {}
    enable_keyword = bool(kw_cfg.get("enabled", True))
    keyword_patterns = kw_cfg.get("patterns")

    det_cfg = eval_cfg.get("detectors") or {}
    enable_unsup = bool(det_cfg.get("enable_unsupervised", True))
    enable_sup = bool(det_cfg.get("enable_supervised", True))
    unsup_list = det_cfg.get("unsupervised")
    sup_list = det_cfg.get("supervised")
    unsup_pos_labels_t = (malicious_label, borderline_label)

    for dataset_name, dataset_dir in dataset_dirs:
        print(f"\n[run] DATASET={dataset_name} at {dataset_dir}")

        print("[run] Loading train_texts (normal only)")
        train_texts = _load_train_normal(
            str(dataset_dir),
            seed=seed,
            cap=max_train_normal_i,
            max_chars=max_chars_i,
            normal_label=normal_label,
        )
        print(f"[run] train_texts loaded: {len(train_texts)}")

        print("[run] Loading val (normal + malicious)")
        val_texts, val_labels = _load_val(
            str(dataset_dir),
            seed=seed + 1,
            cap=max_val_total_i,
            max_chars=max_chars_i,
            normal_label=normal_label,
            malicious_label=malicious_label,
        )
        print(f"[run] val loaded: n={len(val_texts)} counts={_counts(val_labels)}")

        print("[run] Loading test (normal + borderline + malicious)")
        test_texts, test_labels = _load_test(
            str(dataset_dir),
            seed=seed + 2,
            cap=max_test_total_i,
            max_chars=max_chars_i,
            normal_label=normal_label,
            borderline_label=borderline_label,
            malicious_label=malicious_label,
        )
        print(f"[run] test loaded: n={len(test_texts)} counts={_counts(test_labels)}")

        data = DatasetSlices(
            train_texts=train_texts,
            val_texts=val_texts,
            val_labels=val_labels,
            test_texts=test_texts,
            test_labels=test_labels,
        )

        dataset_run_dir = run_root / dataset_name
        dataset_run_dir.mkdir(parents=True, exist_ok=True)

        for emb_spec in embeddings:
            model_dir_name = emb_spec.model_id.replace("/", "_")
            model_run_dir = dataset_run_dir / model_dir_name

            if model_run_dir.exists():
                print(
                    f"[run] Existing results for model_id={emb_spec.model_id} at {model_run_dir}, deleting for rerun"
                )
                shutil.rmtree(model_run_dir)

            cfg = RunConfig(
                run_dir=str(model_run_dir),
                normal_label=normal_label,
                borderline_label=borderline_label,
                malicious_label=malicious_label,
                fpr_points=fpr_points_t,  # type: ignore
                embedding_models=[emb_spec],
                enable_keyword=enable_keyword,
                enable_unsupervised=enable_unsup,
                enable_supervised=enable_sup,
                unsupervised_detectors=(list(unsup_list) if isinstance(unsup_list, list) else None),
                supervised_detectors=(list(sup_list) if isinstance(sup_list, list) else None),
                unsupervised_positive_labels=unsup_pos_labels_t,
                keyword_patterns=(list(keyword_patterns) if isinstance(keyword_patterns, list) else None),
                dataset_name=dataset_name,
            )

            runner = ExperimentRunner(cfg, data)
            runner.run()

            print(f"[run] Saved results for {emb_spec.model_id}: {model_run_dir / 'results.json'}")


if __name__ == "__main__":
    run_eval()
