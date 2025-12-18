# file: embfirewall/embeddings/cache.py
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _safe_model_id(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in model_id)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """
    Lightweight SQLite-backed embedding cache keyed by (model_id, text_hash).

    Stores one embedding vector per text, allowing multiple notebooks to reuse
    previously-computed vectors for the same model without recomputing.
    """

    def __init__(self, base_dir: str | Path, model_id: str) -> None:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        safe_model = _safe_model_id(model_id)
        self.path = base_path / f"{safe_model}.sqlite3"
        self.model_id = model_id

        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (model_id, text_hash)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def lookup(
        self, texts: Sequence[str]
    ) -> Tuple[List[Optional[np.ndarray]], List[int], List[str]]:
        """
        Return cached embeddings (or None) for each text in order.

        Also returns the indices that were missing, plus the text hashes to
        avoid re-hashing on store().
        """

        hashes = [_hash_text(t) for t in texts]
        unique = sorted(set(hashes))

        mapping: dict[str, np.ndarray] = {}
        if unique:
            placeholders = ",".join(["?"] * len(unique))
            args = [self.model_id, *unique]
            cur = self.conn.execute(
                f"SELECT text_hash, embedding, dim FROM embeddings WHERE model_id=? AND text_hash IN ({placeholders})",
                args,
            )
            for text_hash, blob, dim in cur.fetchall():
                arr = np.frombuffer(blob, dtype=np.float32).copy()
                arr = arr.reshape(-1)
                if arr.size != int(dim):
                    arr = arr[: int(dim)]
                mapping[str(text_hash)] = arr

        results: List[Optional[np.ndarray]] = []
        missing_idx: List[int] = []
        for idx, h in enumerate(hashes):
            val = mapping.get(h)
            results.append(val)
            if val is None:
                missing_idx.append(idx)

        return results, missing_idx, hashes

    def store(self, *, hashes: Sequence[str], embeddings: np.ndarray, indices: Sequence[int]) -> None:
        if len(indices) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        rows = []

        for arr_idx, text_idx in enumerate(indices):
            vec = embeddings[arr_idx].reshape(-1).astype(np.float32)
            rows.append((self.model_id, hashes[text_idx], int(vec.size), memoryview(vec.tobytes())))

        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings(model_id, text_hash, dim, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )
