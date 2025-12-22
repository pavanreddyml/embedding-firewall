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

### Selecting specific datasets during evaluation

Both `run_eval.py` and `hypothesis.py` accept a comma-separated environment variable that filters which downloaded datasets to load. Provide the dataset **folder names** under `./data` (the stem of the corresponding `configs/dataset_data_<name>.yaml` file):

- `RUN_DATASETS` for `run_eval.py`
- `HYPOTHESIS_DATASETS` for `hypothesis.py`

Example: if you downloaded from `configs/dataset_data_wiki_tweets.yaml` and `configs/dataset_data_code_jigsaw.yaml`, the shards live in `./data/wiki_tweets` and `./data/code_jigsaw`. To evaluate only those two:

```bash
export RUN_DATASETS=wiki_tweets,code_jigsaw
python run_eval.py
```

Leave the variable unset, empty, or set to `None`/`none` to process every dataset folder under `./data` that contains `normal`, `borderline`, and `malicious` shard files.

To run four datasets at once (for example: wiki_tweets, alpaca_beaver, code_jigsaw, metamath_jigsaw):

```bash
export RUN_DATASETS=wiki_tweets,alpaca_beaver,code_jigsaw,metamath_jigsaw
python run_eval.py
```

## Stronger paper-ready baselines

The default `configs/eval_config.yaml` now includes higher-capacity options that noticeably improve separation between
benign, borderline, and malicious prompts:

- **Embeddings**: add `BAAI/bge-large-en-v1.5` and `intfloat/e5-large-v2` (both normalized). They are slower on CPU but
  deliver stronger semantic recall; if you have a GPU, set `device: cuda` to keep latency manageable.
- **Supervised detectors**: beside the standard logistic regression, try the class-balanced logistic variant,
  a calibrated linear SVM (`linsvm_calibrated`), and a HistGradientBoostingClassifier (`hgbt`). These models are
  better at carving out borderline prompts without letting false positives spike.

### Advanced unsupervised detectors

You can now try deeper unsupervised models implemented in PyTorch (no TensorFlow needed):

- `autoencoder` / `ae`: MLP autoencoder scored via reconstruction error. Tunables: `hidden_dims`, `latent_dim`,
  `epochs`, `dropout`, and `device`.
- `vae`: variational autoencoder with configurable `beta` weight on the KL term for sharper separation of hard cases.
- `gan` / `gan_discriminator`: lightweight GAN that treats `(1 - D(x))` as the anomaly score and supports feature
  matching (`feature_match_weight`) to stabilize training.

Config snippet:

```yaml
detectors:
  - {type: autoencoder, hidden_dims: [512, 256], latent_dim: 128, epochs: 50, batch_size: 512, device: cuda}
  - {type: vae, beta: 1.5, hidden_dims: [512, 256], latent_dim: 96, epochs: 60, device: cuda}
  - {type: gan, hidden_dims: [512, 256], noise_dim: 96, feature_match_weight: 5.0, epochs: 60, device: cuda}
```

They plug into `run_eval.py` the same way as the classical baselines.

Run `python run_eval.py --eval-config configs/eval_config.yaml --run-dir runs/baseline_upgrade` to execute the enlarged
grid and capture plots/metrics for the paper.

## Dataset JSON format

`data/prompts_*.json` is a single JSON array of rows:

```json
[
  {"id":"...","text":"...","label":"benign|malicious|borderline","split":"train|val|test","source":"..."},
  ...
]
```

For very large datasets, writing is streamed to avoid holding everything in memory.

```
# Experiment: Unsupervised-friendly dataset (Recipe – Injection Dominant)
# Key properties:
# - Normal: benign prompts (no attack-like patterns)
# - Borderline: adversarial benign (attack-like syntax but safe)
# - Malicious: adversarial harmful and prompt injections
# Goal: Increase Euclidean / cosine separation between normal and malicious embeddings.
```

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

