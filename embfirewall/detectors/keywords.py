from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np


class KeywordBaseline:
    """Very cheap baseline: flag if any keyword regex matches (case-insensitive)."""

    def __init__(self, patterns: Sequence[str]) -> None:
        self.patterns = list(patterns)
        self._re = re.compile("|".join(f"(?:{p})" for p in self.patterns), re.IGNORECASE)

    def score_texts(self, texts: List[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = 1.0 if self._re.search(t or "") else 0.0
        return out


DEFAULT_PATTERNS = [
    r"ignore\s+all\s+previous",
    r"forget\s+all\s+previous",
    r"system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\b",
    r"reveal\s+the\s+prompt",
    r"bypass",
    r"policy",
    r"rules",
    r"confidential",
]
