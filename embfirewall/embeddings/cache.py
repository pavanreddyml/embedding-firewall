# file: embfirewall/embeddings/cache.py
from __future__ import annotations

import pickle
import time
from collections import defaultdict
import os
from pathlib import Path
from typing import DefaultDict, Dict, Mapping, Optional

import numpy as np


class EmbeddingCache:
    """Simple filesystem cache for embeddings.

    Embeddings are grouped by model identifier and stored as pickle shards with the
    structure ``{"model": {"text": [embedding list]}}``. Each shard only contains
    new items accumulated since the previous flush, keeping write sizes around the
    configured ``shard_size`` (default 5,000). Shards are written as soon as the
    pending queue for a given model crosses that threshold, so persistence happens
    incrementally during long batches rather than only once everything finishes.
    """

    def __init__(self, cache_dir: Path, *, shard_size: int = 5000) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self._shard_counter = 0

        self._store: DefaultDict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self._pending: DefaultDict[str, Dict[str, np.ndarray]] = defaultdict(dict)

        self._load_existing()

    def _load_existing(self) -> None:
        for path in sorted(self.cache_dir.glob("*.pkl")):
            try:
                with path.open("rb") as f:
                    payload = pickle.load(f)
            except Exception:
                continue

            if not isinstance(payload, Mapping):
                continue

            for model_id, text_map in payload.items():
                if not isinstance(text_map, Mapping):
                    continue
                for text, emb_list in text_map.items():
                    try:
                        arr = np.asarray(emb_list, dtype=np.float32).reshape(-1)
                    except Exception:
                        continue
                    self._store[str(model_id)][str(text)] = arr

    def get(self, model_id: str, text: str) -> Optional[np.ndarray]:
        return self._store.get(model_id, {}).get(text)

    def add(self, model_id: str, text: str, embedding: np.ndarray) -> None:
        if text in self._store.get(model_id, {}):
            return

        emb_vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._store[model_id][text] = emb_vec
        self._pending[model_id][text] = emb_vec

        if len(self._pending[model_id]) >= self.shard_size:
            self._flush_model(model_id)

    def _flush_model(self, model_id: str) -> None:
        pending = self._pending.get(model_id)
        if not pending:
            return

        # model identifiers sometimes include path separators (e.g. HF repos
        # like "BAAI/bge-m3"), which would otherwise create unintended
        # directories. Replace them with a safe token so shards stay under the
        # configured cache directory.
        safe_model_id = model_id.replace(os.sep, "__")

        timestamp = int(time.time() * 1000)
        shard_idx = 0

        while pending:
            shard_items = dict(list(pending.items())[: self.shard_size])
            shard_name = f"{safe_model_id}_shard_{timestamp}_{shard_idx}.pkl"
            shard_path = self.cache_dir / shard_name
            serializable = {model_id: {t: vec.tolist() for t, vec in shard_items.items()}}
            with shard_path.open("wb") as f:
                pickle.dump(serializable, f)

            for key in shard_items.keys():
                pending.pop(key, None)

            shard_idx += 1

        self._pending[model_id] = pending

    def flush_all(self) -> None:
        for model_id in list(self._pending.keys()):
            self._flush_model(model_id)

