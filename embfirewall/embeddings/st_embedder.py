# file: embfirewall/embeddings/st_embedder.py
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

from .base import CachedEmbedder


class SentenceTransformerCachedEmbedder(CachedEmbedder):
    @classmethod
    def type_name(cls) -> str:
        return "st"

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        os.environ.setdefault("USE_TF", "0")  # avoid TensorFlow on environments like Colab
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            from transformers.cache_utils import DynamicCache  # type: ignore

            if not hasattr(DynamicCache, "get_usable_length"):
                # Some transformer versions used by sentence-transformers do not ship
                # the `get_usable_length` API required by recent model checkpoints.
                # Reintroduce the method to preserve compatibility with those builds.
                def _get_usable_length(
                    self: "DynamicCache", new_seq_length: int, layer_idx: Optional[int] = 0
                ) -> int:
                    max_length = self.get_max_length()
                    previous_seq_length = self.get_seq_length(layer_idx)
                    if max_length is not None and previous_seq_length + new_seq_length > max_length:
                        return max_length - new_seq_length
                    return previous_seq_length

                DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]
        except Exception:
            # transformers is an optional dependency for the sentence-transformers
            # backend; if it's unavailable we defer to the library to raise a
            # clearer error during model construction.
            pass

        self.device = str(device)
        self.trust_remote_code = bool(trust_remote_code)
        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )

        extra: Dict = {"device": self.device, "trust_remote_code": self.trust_remote_code}
        super().__init__(
            name=name,
            model_id=model_id,
            batch_size=batch_size,
            normalize=normalize,
            extra_config=extra,
        )

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=max(1, int(self.batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # base class handles normalization
        )
        return np.asarray(emb, dtype=np.float32)
