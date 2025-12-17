# file: embfirewall/embeddings/factory.py
from __future__ import annotations

from .openai_embedder import OpenAICachedEmbedder
from .spec import EmbeddingSpec
from .st_embedder import SentenceTransformerCachedEmbedder
from .azure_openai_embedder import AzureOpenAICachedEmbedder
from .ollama_embedder import OllamaCachedEmbedder


def build_embedder(spec: EmbeddingSpec, *, sqlite_path: str):
    if spec.kind == "st":
        return SentenceTransformerCachedEmbedder(
            sqlite_path=sqlite_path,
            name=spec.name,
            model_id=spec.model_id,
            device=spec.device,
            batch_size=spec.batch_size,
            normalize=spec.normalize,
        )

    if spec.kind == "openai":
        return OpenAICachedEmbedder(
            sqlite_path=sqlite_path,
            name=spec.name,
            model_id=spec.model_id,
            batch_size=spec.batch_size,
            normalize=spec.normalize,
            api_key_env=spec.openai_api_key_env,
            base_url=spec.openai_base_url,
            organization=spec.openai_organization,
            project=spec.openai_project,
            dimensions=spec.dimensions,
        )
    
    if spec.kind == "azure_openai":
        return AzureOpenAICachedEmbedder(
            sqlite_path=sqlite_path,
            name=spec.name,
            model_id=spec.model_id,
            batch_size=spec.batch_size,
            normalize=spec.normalize,
            api_key_env=spec.azure_openai_api_key_env,
            endpoint_env=spec.azure_openai_endpoint_env,
            api_version_env=spec.azure_openai_api_version_env,
            endpoint=getattr(spec, "azure_openai_endpoint", None),
            api_version=getattr(spec, "azure_openai_api_version", None),
            dimensions=getattr(spec, "dimensions", None),
        )

    if spec.kind == "ollama":
        return OllamaCachedEmbedder(
            sqlite_path=sqlite_path,
            name=spec.name,
            model_id=spec.model_id,
            batch_size=spec.batch_size,
            normalize=spec.normalize,
            base_url=getattr(spec, "ollama_base_url", "http://localhost:11434"),
            request_timeout=getattr(spec, "ollama_request_timeout", 120.0),
        )

    raise ValueError(f"Unknown embedding kind: {spec.kind}")
