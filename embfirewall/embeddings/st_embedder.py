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
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

        from sentence_transformers import SentenceTransformer
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
            normalize_embeddings=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def close(self) -> None:
        try:
            import torch
        except Exception:
            torch = None

        model = getattr(self, "model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        self.model = None

        if torch is not None:
            try:
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
