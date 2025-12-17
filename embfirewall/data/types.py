# file: embfirewall/data/types.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

_whitespace_re = re.compile(r"\s+")


def normalize_whitespace(s: str) -> str:
    return _whitespace_re.sub(" ", s).strip()


@dataclass
class TextFilters:
    min_chars: int = 10
    max_chars: int = 20000
    normalize_ws: bool = True

    def apply(self, s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        t = s.strip()
        if self.normalize_ws:
            t = normalize_whitespace(t)
        if len(t) < self.min_chars or len(t) > self.max_chars:
            return None
        return t


@dataclass
class SourceExample:
    text: str
    meta: Dict[str, Any]
