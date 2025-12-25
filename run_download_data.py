from __future__ import annotations

from pathlib import Path

import yaml

from embfirewall.data.downloader import DatasetDownloader
from embfirewall.data.types import TextFilters


def _in_colab() -> bool:
    try:
        import importlib

        return importlib.util.find_spec("google.colab") is not None
    except Exception:
        return False


IN_COLAB = _in_colab()
LOCAL_BASE_DIR = "."
COLAB_BASE_DIR = "/content/drive/MyDrive/research/embfirewall"

WORKING_DIR = LOCAL_BASE_DIR
STORAGE_DIR = COLAB_BASE_DIR if IN_COLAB else LOCAL_BASE_DIR

DATASET_CONFIG_DIR = Path(WORKING_DIR) / "configs"
DATASET_CONFIG_PATTERN = "dataset_data_*.yaml"
OUT_DIR = str(Path(STORAGE_DIR) / "data")

FLUSH_EVERY = 1000
SHARD_SIZE = 1000
SEED = 7
OVERWRITE = True


def _dataset_configs() -> list[Path]:
    paths = sorted(DATASET_CONFIG_DIR.glob(DATASET_CONFIG_PATTERN))
    if not paths:
        raise SystemExit(f"[download_data] No dataset configs matching {DATASET_CONFIG_PATTERN} under {DATASET_CONFIG_DIR}")
    return paths


def _dataset_name_from_config(path: Path) -> str:
    stem = path.stem
    prefix = "dataset_data_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def main() -> None:
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}

    print(
        f"[download_data] IN_COLAB={IN_COLAB} WORKING_DIR={WORKING_DIR}\n"
        f"[download_data] config_dir={DATASET_CONFIG_DIR}\n"
        f"[download_data] out_dir={OUT_DIR}\n"
        f"[download_data] flush_every={FLUSH_EVERY} seed={SEED} overwrite={OVERWRITE}"
    )

    for cfg_path in _dataset_configs():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        tf_cfg = cfg.get("text_filters") or {}
        tf = TextFilters(
            min_chars=int(tf_cfg.get("min_chars", 10)),
            max_chars=int(tf_cfg.get("max_chars", 20000)),
            normalize_ws=bool(tf_cfg.get("normalize_ws", True)),
        )

        seed = int(cfg.get("seed", SEED))
        dl = DatasetDownloader(
            seed=seed,
            text_filters=tf,
            flush_every=FLUSH_EVERY,
            overwrite=OVERWRITE,
            shard_size=SHARD_SIZE,
        )

        labels = cfg.get("labels") or {}
        if not labels:
            raise SystemExit(f"{cfg_path}: missing labels")

        for label in ["normal", "borderline", "malicious"]:
            if label not in labels:
                raise SystemExit(f"{cfg_path}: missing labels.{label}")

        dataset_name = _dataset_name_from_config(cfg_path)
        dataset_out_dir = out_root / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[download_data] DATASET={dataset_name} from {cfg_path}")

        stats: dict[str, dict] = {}
        for label, lc in labels.items():
            out_path = str(dataset_out_dir / label)
            print(f"[download_data] START label={label} -> {out_path}")
            st = dl.download_label(label=label, label_cfg=lc, out_path=out_path)
            stats[label] = st
            print(f"[download_data] DONE label={label}: {st}")

        stats_path = dataset_out_dir / "download_stats.yaml"
        with open(stats_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(stats, f, sort_keys=False, allow_unicode=True)
        all_stats[dataset_name] = stats
        print(f"[download_data] Wrote stats: {stats_path}")

    summary_path = out_root / "download_stats.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(all_stats, f, sort_keys=False, allow_unicode=True)
    print(f"\n[download_data] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
