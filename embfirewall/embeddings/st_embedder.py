# file: embfirewall/embeddings/st_embedder.py
from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
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
        self.device = str(device)
        self.trust_remote_code = bool(trust_remote_code)
        model_kwargs = {"use_cache": False}
        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=model_kwargs,
        )

        try:
            transformer = self.model._first_module()
            if hasattr(transformer, "auto_model") and hasattr(
                transformer.auto_model, "config"
            ):
                transformer.auto_model.config.use_cache = False
            elif hasattr(transformer, "model") and hasattr(transformer.model, "config"):
                transformer.model.config.use_cache = False
        except Exception:
            # If we cannot toggle caching explicitly, proceed without failing init.
            pass

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
