# Embedding Firewalls (Large Dataset + Experiments)

This repo builds large prompt datasets (hundreds of thousands to millions of prompts) and runs embedding-based anomaly detectors + baselines.

Key changes vs earlier versions:
- Dataset storage is **plain JSON** (a JSON array). No JSONL.
- Dataset builder supports **many sources per label**, with per-source weights/percentages, streaming download, and deterministic splits by ratio.
- Splits accept either ratios (e.g. 0.4/0.3/0.3) or exact counts.

## Quickstart

1) Install deps:
- `pip install -r requirements.txt`

2) Make sure an Ollama server is running (default port `11434`). On Colab, start it in a background cell and pull the
   embedding models once to avoid cold-start latency:
   - `ollama pull nomic-embed-text`
   - `ollama pull mxbai-embed-large`

3) Edit a dataset config (examples in `configs/`), then build:
- `python run.py`
  - This will build a dataset JSON (if missing) and then run experiments (optional toggles inside `run.py`).

## Stronger paper-ready baselines (Ollama-first)

The default `configs/eval_config.yaml` now targets **Ollama** embeddings so the full evaluation grid can run on Colab or
any CPU box without pulling large Hugging Face checkpoints:

- **Embeddings**: `nomic-embed-text` (fast/general) and `mxbai-embed-large` (higher-capacity). Both are served through
  Ollama; the first request will trigger a download if the model is missing.
- **GPU first**: the default `ollama_options.num_gpu=1` in `configs/eval_config.yaml` pushes embedding to GPU on Colab (or
  any host with a CUDA-capable card). Increase `num_gpu` if you have multiple GPUs; drop the field if you want CPU-only.
- **Supervised detectors**: beside the standard logistic regression, try the class-balanced logistic variant,
  a calibrated linear SVM (`linsvm_calibrated`), and a HistGradientBoostingClassifier (`hgbt`). These models are
  better at carving out borderline prompts without letting false positives spike.

Run `python run_eval.py --eval-config configs/eval_config.yaml --run-dir runs/ollama_baseline` once Ollama is up to
capture plots/metrics for the paper. You can also override the dataset path or embedding cache location if you keep
data on Google Drive:

- The default eval config caps each split at **3000** rows while you acquire/download the full datasets; set the caps to
  `null` to remove limits once everything is available.

- `--data-dir /content/drive/MyDrive/research/embfirewall/data`
- `--embed-db-path /content/drive/MyDrive/research/embfirewall/embeddings_cache.sqlite3`

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

