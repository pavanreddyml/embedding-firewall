# file: embfirewall/embeddings/__init__.py
from .factory import build_embedder
from .spec import EmbeddingSpec

__all__ = ["EmbeddingSpec", "build_embedder"]
