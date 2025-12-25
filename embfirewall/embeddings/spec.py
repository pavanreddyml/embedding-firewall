from __future__ import annotations

from typing import Any, Dict, Optional


class EmbeddingSpec:
    def __init__(
        self,
        *,
        kind: str,
        name: str,
        model_id: str,
        batch_size: int = 64,
        normalize: bool = True,
        device: str = "cpu",
        dimensions: Optional[int] = None,
        openai_api_key_env: str = "OPENAI_API_KEY",
        openai_base_url: Optional[str] = None,
        openai_organization: Optional[str] = None,
        openai_project: Optional[str] = None,
        ollama_base_url: Optional[str] = "http://localhost:11434",
        ollama_request_timeout: float = 120.0,
        trust_remote_code: bool = True,
    ) -> None:
        self.kind = str(kind)
        self.name = str(name)
        self.model_id = str(model_id)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)

        self.device = str(device)
        self.trust_remote_code = bool(trust_remote_code)

        self.dimensions = int(dimensions) if dimensions is not None else None
        self.openai_api_key_env = str(openai_api_key_env)
        self.openai_base_url = openai_base_url
        self.openai_organization = openai_organization
        self.openai_project = openai_project

        self.ollama_base_url = ollama_base_url
        self.ollama_request_timeout = float(ollama_request_timeout)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "dimensions": self.dimensions,
            "openai_api_key_env": self.openai_api_key_env,
            "openai_base_url": self.openai_base_url,
            "openai_organization": self.openai_organization,
            "openai_project": self.openai_project,
            "ollama_base_url": self.ollama_base_url,
            "ollama_request_timeout": self.ollama_request_timeout,
        }
