# file: embfirewall/embeddings/ollama_embedder.py
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import requests

from .base import CachedEmbedder


class OllamaCachedEmbedder(CachedEmbedder):
    """Calls a running Ollama instance for embeddings (with SQLite caching).

    Expects the Ollama HTTP API to be reachable (default: http://localhost:11434).
    """

    def __init__(
        self,
        *,
        sqlite_path: str,
        name: str,
        model_id: str,
        batch_size: int = 8,
        normalize: bool = True,
        base_url: str = "http://localhost:11434",
        request_timeout: float = 120.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/") or "http://localhost:11434"
        self.request_timeout = float(request_timeout)
        super().__init__(
            sqlite_path=sqlite_path,
            name=name,
            model_id=model_id,
            batch_size=batch_size,
            normalize=normalize,
            extra_config={
                "base_url": self.base_url,
                "request_timeout": self.request_timeout,
            },
        )

    @classmethod
    def type_name(cls) -> str:
        return "ollama"

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        url = f"{self.base_url}/api/embeddings"
        vecs = []
        for t in texts:
            payload: Dict[str, str] = {"model": self.model_id, "prompt": t}
            try:
                resp = requests.post(url, json=payload, timeout=self.request_timeout)
            except Exception as exc:  # pragma: no cover - network/connection errors
                raise RuntimeError(
                    f"Failed to reach Ollama at {url} (ensure ollama serve is running): {exc}"
                ) from exc

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama embeddings error (status {resp.status_code}) for model={self.model_id}: {resp.text}"
                )

            data = resp.json()
            if "embedding" not in data:
                raise RuntimeError(
                    f"Unexpected Ollama response; missing 'embedding' field: keys={list(data.keys())}"
                )

            vec = np.asarray(data["embedding"], dtype=np.float32)
            if vec.ndim != 1:
                raise RuntimeError(
                    f"Ollama returned embedding with invalid shape {vec.shape} for model={self.model_id}"
                )
            vecs.append(vec)

        # Convert to (n, d)
        first_dim = vecs[0].shape[0]
        arr = np.zeros((len(vecs), first_dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            if v.shape[0] != first_dim:
                raise RuntimeError(
                    f"Ollama embeddings dimension mismatch: expected {first_dim}, got {v.shape[0]} for model={self.model_id}"
                )
            arr[i] = v

        return arr
