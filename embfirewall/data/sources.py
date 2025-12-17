# file: embfirewall/data/sources.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .types import SourceExample

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _get(ex: Any, key: str) -> Any:
    if isinstance(ex, dict):
        return ex.get(key)
    try:
        return getattr(ex, key)
    except Exception:
        return None


def _passes_filters(ex: Any, filters: Dict[str, Any]) -> bool:
    if not filters:
        return True

    eq = filters.get("eq") or {}
    for k, v in eq.items():
        if _get(ex, k) != v:
            return False

    isin = filters.get("in") or {}
    for k, vs in isin.items():
        if _get(ex, k) not in set(_as_list(vs)):
            return False

    contains = filters.get("contains") or {}
    for k, sub in contains.items():
        val = _get(ex, k)
        if val is None:
            return False
        if str(sub).lower() not in str(val).lower():
            return False

    any_contains = filters.get("any_contains") or {}
    for k, subs in any_contains.items():
        val = _get(ex, k)
        items = _as_list(val)
        ok = False
        for it in items:
            s = str(it).lower()
            for sub in _as_list(subs):
                if str(sub).lower() in s:
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return False

    return True


class PromptSource:
    name: str

    def iter_examples(self) -> Iterator[SourceExample]:
        raise NotImplementedError


@dataclass
class HFDatasetsSource(PromptSource):
    name: str
    dataset: str
    split: str = "train"
    subset: Optional[str] = None
    streaming: bool = True
    text_field: Optional[str] = None
    text_fields: Optional[Sequence[str]] = None
    filters: Optional[Dict[str, Any]] = None

    def _extract_text(self, ex: Any) -> Optional[str]:
        # If text_fields provided, use first non-empty str
        if self.text_fields:
            for f in self.text_fields:
                v = _get(ex, f)
                if isinstance(v, str) and v.strip():
                    return v
        # If single text_field
        if self.text_field:
            v = _get(ex, self.text_field)
            if isinstance(v, str) and v.strip():
                return v
        # Common fallbacks
        for f in ("prompt", "question", "instruction", "query", "text", "input"):
            v = _get(ex, f)
            if isinstance(v, str) and v.strip():
                return v
        return None

    def iter_examples(self) -> Iterator[SourceExample]:
        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Run: pip install datasets")

        ds = load_dataset(
            self.dataset,
            self.subset,
            split=self.split,
            streaming=bool(self.streaming),
        )

        flt = self.filters or {}
        for ex in ds:
            if not _passes_filters(ex, flt):
                continue
            t = self._extract_text(ex)
            if t is None:
                continue
            yield SourceExample(
                text=t,
                meta={"dataset": self.dataset, "subset": self.subset, "split": self.split},
            )


def make_source(kind: str, **spec: Any) -> PromptSource:
    kind = str(kind).lower().strip()

    # ignore config-only keys
    spec.pop("weight", None)
    spec.pop("target", None)
    spec.pop("quota", None)
    spec.pop("max_rows", None)
    spec.pop("cap", None)

    if kind in ("hf", "hf_datasets", "datasets", "huggingface"):
        name = spec.pop("name", spec.get("dataset", "hf"))
        dataset = spec.pop("dataset", None)
        if not dataset:
            raise ValueError("Missing required key: dataset")
        return HFDatasetsSource(name=str(name), dataset=str(dataset), **spec)

    raise ValueError(f"Unknown source kind: {kind}")
