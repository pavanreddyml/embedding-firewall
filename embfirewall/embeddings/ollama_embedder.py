from __future__ import annotations

import json
from typing import Dict, Sequence

import numpy as np
import requests

from .base import Embedder


class OllamaEmbedder(Embedder):
    """Calls a running Ollama instance for embeddings.

    Expects the Ollama HTTP API to be reachable (default: http://localhost:11434).
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        batch_size: int = 8,
        normalize: bool = True,
        base_url: str = "http://localhost:11434",
        request_timeout: float = 120.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/") or "http://localhost:11434"
        self.request_timeout = float(request_timeout)
        self._model_checked = False
        super().__init__(
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

        self._ensure_model_available()

        url = f"{self.base_url}/api/embeddings"
        vecs = []
        for t in texts:
            payload: Dict[str, str] = {"model": self.model_id, "prompt": t}
            try:
                resp = requests.post(url, json=payload, timeout=self.request_timeout)
            except Exception as exc:
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

        first_dim = vecs[0].shape[0]
        arr = np.zeros((len(vecs), first_dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            if v.shape[0] != first_dim:
                raise RuntimeError(
                    f"Ollama embeddings dimension mismatch: expected {first_dim}, got {v.shape[0]} for model={self.model_id}"
                )
            arr[i] = v

        return arr

    def _ensure_model_available(self) -> None:
        if self._model_checked:
            return

        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=self.request_timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reach Ollama at {tags_url} (ensure ollama serve is running): {exc}"
            ) from exc

        if resp.status_code == 200:
            try:
                models = resp.json().get("models", [])
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid response from Ollama tags endpoint for model={self.model_id}: {resp.text}"
                ) from exc
            if any(m.get("name") == self.model_id for m in models):
                self._model_checked = True
                return
        elif resp.status_code != 404:
            raise RuntimeError(
                f"Ollama tags error (status {resp.status_code}) while checking model={self.model_id}: {resp.text}"
            )

        pull_url = f"{self.base_url}/api/pull"
        try:
            resp = requests.post(
                pull_url,
                json={"name": self.model_id},
                timeout=self.request_timeout,
                stream=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reach Ollama at {pull_url} (ensure ollama serve is running): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama pull error (status {resp.status_code}) for model={self.model_id}: {resp.text}"
            )

        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                if payload.get("error"):
                    raise RuntimeError(
                        f"Ollama pull error for model={self.model_id}: {payload['error']}"
                    )
                if payload.get("status") == "success":
                    self._model_checked = True
                    return
        finally:
            resp.close()

        raise RuntimeError(
            f"Ollama pull did not complete successfully for model={self.model_id}"
        )
