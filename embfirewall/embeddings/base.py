# file: embfirewall/embeddings/base.py
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _json_dumps_sorted(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


class CachedEmbedder(ABC):
    """
    Base class for embedders with SQLite caching.
    Cache key = (model_key, text_hash).
    Stores: text, config, embedding (float32 blob), dim, timestamps.
    """

    def __init__(
        self,
        *,
        sqlite_path: str,
        name: str,
        model_id: str,
        batch_size: int = 64,
        normalize: bool = True,
        extra_config: Optional[Dict] = None,
    ) -> None:
        self.name = str(name)
        self.model_id = str(model_id)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)

        cfg: Dict = {
            "type": self.type_name(),
            "name": self.name,
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }
        if extra_config:
            cfg.update(extra_config)

        self.config: Dict = cfg
        self.model_key: str = _sha256_hex(_json_dumps_sorted(self.config))

        self.sqlite_path = sqlite_path
        os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(sqlite_path, timeout=60.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_db()
        self._upsert_model_row()

    @classmethod
    @abstractmethod
    def type_name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Must return float32 array of shape (n, d).
        Should NOT apply normalization; base class handles normalize=True.
        """
        raise NotImplementedError

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                model_key   TEXT PRIMARY KEY,
                name        TEXT,
                config_json TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model_key   TEXT NOT NULL,
                text_hash   TEXT NOT NULL,
                text        TEXT NOT NULL,
                dim         INTEGER NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  REAL NOT NULL,
                PRIMARY KEY (model_key, text_hash),
                FOREIGN KEY (model_key) REFERENCES models(model_key) ON DELETE CASCADE
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_key);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(text_hash);")
        self._conn.commit()

    def _upsert_model_row(self) -> None:
        cfg_json = _json_dumps_sorted(self.config)
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO models(model_key, name, config_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_key) DO UPDATE SET
              name=excluded.name,
              config_json=excluded.config_json;
            """,
            (self.model_key, self.name, cfg_json, now),
        )
        self._conn.commit()

    def _select_cached(self, text_hashes: List[str]) -> Dict[str, np.ndarray]:
        if not text_hashes:
            return {}

        out: Dict[str, np.ndarray] = {}
        # SQLite has a variable limit; stay conservative.
        CHUNK = 900
        cur = self._conn.cursor()
        for i in range(0, len(text_hashes), CHUNK):
            chunk = text_hashes[i : i + CHUNK]
            qmarks = ",".join(["?"] * len(chunk))
            rows = cur.execute(
                f"""
                SELECT text_hash, dim, embedding
                FROM embeddings
                WHERE model_key = ?
                  AND text_hash IN ({qmarks});
                """,
                (self.model_key, *chunk),
            ).fetchall()

            for h, dim, blob in rows:
                try:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.size != int(dim):
                        continue
                    out[str(h)] = vec.copy()
                except Exception:
                    continue
        return out

    def _insert_cached(self, items: List[Tuple[str, str, np.ndarray]]) -> None:
        # items: (text_hash, text, vec)
        if not items:
            return
        now = time.time()
        rows = []
        for h, text, vec in items:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            rows.append((self.model_key, h, text, int(v.size), sqlite3.Binary(v.tobytes()), now))

        self._conn.executemany(
            """
            INSERT INTO embeddings(model_key, text_hash, text, dim, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_key, text_hash) DO UPDATE SET
              text=excluded.text,
              dim=excluded.dim,
              embedding=excluded.embedding,
              created_at=excluded.created_at;
            """,
            rows,
        )
        self._conn.commit()

    def _infer_fallback_dim(self) -> Optional[int]:
        try:
            probe = np.asarray(self._encode_batch([""]))
            if probe.ndim == 2 and probe.shape[0] == 1:
                vec = probe[0].astype(np.float32)
                if self.normalize:
                    vec = _l2_normalize(vec.reshape(1, -1))[0]
                return int(vec.size)
        except Exception:
            return None
        return None

    def embed(self, texts: List[str], *, desc: str = "") -> Tuple[np.ndarray, float]:
        t0 = time.time()
        n = len(texts)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32), 0.0

        valid_pairs = [(idx, t) for idx, t in enumerate(texts) if isinstance(t, str)]
        invalid_indices = [idx for idx, t in enumerate(texts) if not isinstance(t, str)]

        hashes = [_sha256_hex(t) for _, t in valid_pairs]
        uniq_hashes = list(dict.fromkeys(hashes).keys())

        cached = self._select_cached(uniq_hashes)
        for vec in cached.values():
            if vec.size > 0:
                fallback_dim = int(vec.size)
                break
        else:
            fallback_dim = None

        to_insert: List[Tuple[str, str, np.ndarray]] = []
        results: List[Optional[np.ndarray]] = [None] * n

        for (idx, text), h in zip(valid_pairs, hashes):
            vec = cached.get(h)
            if vec is not None:
                results[idx] = vec.reshape(-1).copy()
                fallback_dim = fallback_dim or int(vec.size)
                continue

            try:
                batch_arr = np.asarray(self._encode_batch([text]), dtype=np.float32)
                if batch_arr.ndim != 2 or batch_arr.shape[0] != 1:
                    raise ValueError(
                        f"Invalid embedding shape from {self.type_name()}: expected (1, d), got {tuple(batch_arr.shape)}"
                    )
                vec = batch_arr[0]
                if self.normalize:
                    vec = _l2_normalize(vec.reshape(1, -1))[0]
                vec = vec.reshape(-1).copy()
                fallback_dim = fallback_dim or int(vec.size)
                cached[h] = vec
                results[idx] = vec.copy()
                to_insert.append((h, text, vec))
            except Exception as exc:  # pragma: no cover - network/remote failures
                tqdm.write(
                    f"[warn] Failed to embed text at index {idx} via {self.type_name()}: {exc}"
                )
                results[idx] = None

        if to_insert:
            self._insert_cached(to_insert)

        if fallback_dim is None:
            fallback_dim = self._infer_fallback_dim()
        if fallback_dim is None:
            raise RuntimeError(
                "Unable to determine embedding dimension for zero-vector fallbacks"
            )

        zero_vec = np.zeros((fallback_dim,), dtype=np.float32)
        for idx in invalid_indices:
            results[idx] = zero_vec.copy()

        for i, val in enumerate(results):
            if val is None:
                results[i] = zero_vec.copy()

        out = np.vstack([vec.reshape(1, -1) for vec in results])

        dt = time.time() - t0
        return out, dt
