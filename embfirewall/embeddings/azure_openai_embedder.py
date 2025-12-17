# file: embfirewall/embeddings/azure_openai_embedder.py
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .base import CachedEmbedder


def _build_token_counter(_model_hint: str) -> Callable[[str], int]:
    # Azure "model" is often a deployment name; tiktoken may not recognize it.
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")

        def _count(s: str) -> int:
            return len(enc.encode(s))

        return _count
    except Exception:

        def _count(s: str) -> int:
            return max(1, (len(s) + 3) // 4)

        return _count


class _FixedWindowTokenLimiter:
    def __init__(self, tokens_per_window: int, window_s: float = 60.0) -> None:
        self.tokens_per_window = int(tokens_per_window)
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._window_start = 0.0
        self._used = 0

    def acquire(self, tokens: int) -> None:
        if self.tokens_per_window <= 0:
            return
        need = int(max(0, tokens))
        if need == 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                w = self.window_s
                cur_start = now - (now % w)

                if self._window_start != cur_start:
                    self._window_start = cur_start
                    self._used = 0

                if self._used + need <= self.tokens_per_window:
                    self._used += need
                    return

                sleep_s = (self._window_start + w) - now

            if sleep_s > 0:
                time.sleep(min(sleep_s, w))
            else:
                time.sleep(0)


class AzureOpenAICachedEmbedder(CachedEmbedder):
    @classmethod
    def type_name(cls) -> str:
        return "azure_openai"

    def __init__(
        self,
        *,
        sqlite_path: str,
        name: str,
        # For Azure embeddings, this is typically the *deployment name*.
        model_id: str,
        batch_size: int = 128,
        normalize: bool = True,
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        api_version_env: str = "AZURE_OPENAI_API_VERSION",
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_retries: int = 6,
        # Rate limit (TPM). Default: 100k tokens/minute.
        rate_limit_tpm: int = 100_000,
        rate_limit_window_s: float = 60.0,
    ) -> None:
        self.api_key_env = str(api_key_env)
        self.endpoint_env = str(endpoint_env)
        self.api_version_env = str(api_version_env)

        self.endpoint = endpoint or os.environ.get(self.endpoint_env)
        self.api_version = api_version or os.environ.get(self.api_version_env)
        self.dimensions = int(dimensions) if dimensions is not None else None
        self.max_retries = int(max_retries)

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing Azure API key env var: {self.api_key_env}")
        if not self.endpoint:
            raise RuntimeError(f"Missing Azure endpoint (param endpoint or env var: {self.endpoint_env})")
        if not self.api_version:
            raise RuntimeError(f"Missing Azure api_version (param api_version or env var: {self.api_version_env})")

        # openai>=1.x
        from openai import AzureOpenAI  # type: ignore

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

        self._count_tokens = _build_token_counter(str(model_id))
        self._limiter = _FixedWindowTokenLimiter(int(rate_limit_tpm), float(rate_limit_window_s))

        extra: Dict = {
            "api_key_env": self.api_key_env,
            "endpoint_env": self.endpoint_env,
            "api_version_env": self.api_version_env,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "dimensions": self.dimensions,
        }
        super().__init__(
            sqlite_path=sqlite_path,
            name=name,
            model_id=model_id,
            batch_size=batch_size,
            normalize=normalize,
            extra_config=extra,
        )

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        inp = list(texts)

        kwargs: Dict = {"model": self.model_id, "input": inp}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                est_tokens = sum(self._count_tokens(t) for t in inp)
                self._limiter.acquire(est_tokens)

                res = self.client.embeddings.create(**kwargs)
                vecs = [d.embedding for d in res.data]
                arr = np.asarray(vecs, dtype=np.float32)
                if arr.shape[0] != len(inp):
                    raise RuntimeError(f"Azure OpenAI returned n={arr.shape[0]} embeddings for input n={len(inp)}")
                return arr
            except Exception as e:
                last_err = e
                if "dimensions" in kwargs:
                    kwargs.pop("dimensions", None)
                sleep_s = min(8.0, 0.5 * (2**attempt))
                time.sleep(sleep_s)

        raise RuntimeError(f"Azure OpenAI embeddings failed after {self.max_retries} retries: {last_err}")
