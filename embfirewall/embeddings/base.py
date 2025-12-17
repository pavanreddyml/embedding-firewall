# file: embfirewall/embeddings/base.py
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


class CachedEmbedder(ABC):
    """
    Base class for embedders without on-disk caching.
    Handles batching, normalization, and fallback logic.
    """

    def __init__(
        self,
        *,
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
        return None

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

        fallback_dim: Optional[int] = None
        results: List[Optional[np.ndarray]] = [None] * n

        progress = tqdm(
            total=len(valid_pairs),
            desc=desc or "embed",
            unit="text",
            leave=False,
        )

        try:
            for start in range(0, len(valid_pairs), self.batch_size):
                batch_pairs = valid_pairs[start : start + self.batch_size]
                batch_texts = [t for _, t in batch_pairs]

                try:
                    batch_arr = np.asarray(self._encode_batch(batch_texts), dtype=np.float32)
                    if batch_arr.ndim != 2 or batch_arr.shape[0] != len(batch_texts):
                        raise ValueError(
                            f"Invalid embedding shape from {self.type_name()}: expected ({len(batch_texts)}, d), got {tuple(batch_arr.shape)}"
                        )
                    if self.normalize:
                        batch_arr = _l2_normalize(batch_arr)

                    fallback_dim = fallback_dim or int(batch_arr.shape[1])
                    for (idx, _), vec in zip(batch_pairs, batch_arr):
                        results[idx] = vec.reshape(-1).copy()
                except Exception:
                    for idx, text in batch_pairs:
                        try:
                            single_arr = np.asarray(self._encode_batch([text]), dtype=np.float32)
                            if single_arr.ndim != 2 or single_arr.shape[0] != 1:
                                raise ValueError(
                                    f"Invalid embedding shape from {self.type_name()}: expected (1, d), got {tuple(single_arr.shape)}"
                                )
                            vec = single_arr[0]
                            if self.normalize:
                                vec = _l2_normalize(vec.reshape(1, -1))[0]
                            vec = vec.reshape(-1).copy()
                            fallback_dim = fallback_dim or int(vec.size)
                            results[idx] = vec.copy()
                        except Exception as exc:  # pragma: no cover - network/remote failures
                            tqdm.write(
                                f"[warn] Failed to embed text at index {idx} via {self.type_name()}: {exc}"
                            )
                            results[idx] = None

                progress.update(len(batch_pairs))
        finally:
            progress.close()

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
