# Embedding Firewalls (Large Dataset + Experiments)

This repo builds large prompt datasets (hundreds of thousands to millions of prompts) and runs embedding-based anomaly detectors + baselines.

Key changes vs earlier versions:
- Dataset storage is **plain JSON** (a JSON array). No JSONL.
- Dataset builder supports **many sources per label**, with per-source weights/percentages, streaming download, and deterministic splits by ratio.
- Splits accept either ratios (e.g. 0.4/0.3/0.3) or exact counts.

## Quickstart

1) Install deps:
- `pip install -r requirements.txt`

2) Edit a dataset config (examples in `configs/`), then build:
- `python run.py`
  - This will build a dataset JSON (if missing) and then run experiments (optional toggles inside `run.py`).

## Dataset JSON format

`data/prompts_*.json` is a single JSON array of rows:

```json
[
  {"id":"...","text":"...","label":"benign|malicious|borderline","split":"train|val|test","source":"..."},
  ...
]
```

For very large datasets, writing is streamed to avoid holding everything in memory.

## Configs

See `configs/dataset_500k.yaml` and `configs/dataset_3x1m.yaml`.

- `labels.<name>.size`: int or float.
  - If int: exact count for that label.
  - If float: treated as a ratio of `global_total` (if provided).
- `labels.<name>.sources`: list of sources with `weight` and Hugging Face dataset details.
- `split`: ratios or counts. If ratios, split assignment uses stable hashing of text (deterministic).

The builder will save a resolved copy (with computed quotas) at the path you set in:
`output.save_resolved_config_to`.

## Notes (Windows)

If you have TensorFlow + Keras 3 installed, Transformers may try to import TF/Keras and error.
This repo sets `TRANSFORMERS_NO_TF=1` before importing SentenceTransformers to avoid TF imports.

