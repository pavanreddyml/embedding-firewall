# file: scripts/download_data.py
from __future__ import annotations

from pathlib import Path
import yaml

from embfirewall.data.downloader import DatasetDownloader
from embfirewall.data.types import TextFilters


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


# -------- GLOBALS (edit these) --------
IN_COLAB = _in_colab()

# Assume Drive already mounted in Colab.
LOCAL_BASE_DIR = "."
COLAB_BASE_DIR = "/content/drive/MyDrive/research/embfirewall"  # <-- change to your folder

WORKING_DIR = LOCAL_BASE_DIR
STORAGE_DIR = COLAB_BASE_DIR if IN_COLAB else LOCAL_BASE_DIR

DATASET_CONFIG_PATH = str(Path(WORKING_DIR) / "configs" / "dataset_data.yaml")
OUT_DIR = str(Path(STORAGE_DIR) / "data")

FLUSH_EVERY = 1000
SHARD_SIZE = 1000  # writes one JSON array file per 1000 rows (crash-safe, no JSONL)
SEED = 7
OVERWRITE = True  # downloads ALL rows; overwrite output each run to avoid duplication
# --------------------------------------


def main() -> None:
    cfg = yaml.safe_load(Path(DATASET_CONFIG_PATH).read_text(encoding="utf-8"))

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    tf_cfg = cfg.get("text_filters") or {}
    tf = TextFilters(
        min_chars=int(tf_cfg.get("min_chars", 10)),
        max_chars=int(tf_cfg.get("max_chars", 20000)),
        normalize_ws=bool(tf_cfg.get("normalize_ws", True)),
    )

    dl = DatasetDownloader(seed=SEED, text_filters=tf, flush_every=FLUSH_EVERY, overwrite=OVERWRITE)
        shard_size=SHARD_SIZE,

    labels = cfg.get("labels") or {}
    if not labels:
        raise SystemExit(f"{DATASET_CONFIG_PATH}: missing labels")

    for label in ["normal", "borderline", "malicious"]:
        if label not in labels:
            raise SystemExit(f"{DATASET_CONFIG_PATH}: missing labels.{label}")

    print(
        f"[download_data] IN_COLAB={IN_COLAB} WORKING_DIR={WORKING_DIR}\n"
        f"[download_data] config={DATASET_CONFIG_PATH}\n"
        f"[download_data] out_dir={OUT_DIR}\n"
        f"[download_data] flush_every={FLUSH_EVERY} seed={SEED} overwrite={OVERWRITE}"
    )

    stats = {}
    for label, lc in labels.items():
        out_path = str(Path(OUT_DIR) / f"{label}.json")
        print(f"\n[download_data] START label={label} -> {out_path}")
        st = dl.download_label(label=label, label_cfg=lc, out_path=out_path)
        stats[label] = st
        print(f"[download_data] DONE label={label}: {st}")

    stats_path = Path(OUT_DIR) / "download_stats.yaml"
    with open(stats_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False, allow_unicode=True)
    print(f"\n[download_data] Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
