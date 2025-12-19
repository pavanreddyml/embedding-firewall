# file: embfirewall/data/loaders.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class DatasetRow:
    text: str
    label: str
    source: Optional[str] = None
    id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _read_json_array(path: Path, *, show_progress: bool, desc: str) -> List[Dict[str, Any]]:
    """
    Load a JSON *array* from disk (not JSONL/NDJSON).

    Uses a byte-progress bar while reading the file.
    """
    if not path.exists():
        return []

    total = path.stat().st_size
    data = bytearray()

    bar = tqdm(
        total=total if total > 0 else None,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True,
        disable=not show_progress,
    )

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                data.extend(chunk)
                bar.update(len(chunk))
    finally:
        bar.close()

    b = bytes(data).strip()
    if not b:
        # empty file (likely interrupted download/write) -> treat as no rows
        print(f"[loaders] WARNING: empty file: {path}")
        return []

    try:
        obj = json.loads(b.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"[loaders] WARNING: JSON decode failed ({path}): {e}. Skipping file.")
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    # common patterns: {"data":[...]}, {"train":[...]}
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [x for x in v if isinstance(x, dict)]

    return []


def _list_label_files(data_dir: Path, label: str) -> List[Path]:
    """
    Prefer shards: <label>-00000.json, <label>-00001.json, ...
    Else single: <label>.json
    """
    # First, look directly under the provided data directory.
    shards = sorted(p for p in data_dir.glob(f"{label}-*.json") if p.is_file())
    if shards:
        return shards

    single = data_dir / f"{label}.json"
    if single.exists():
        return [single]

    # Fallback: allow datasets where each label lives in its own subfolder
    # (e.g., <dataset>/<label>/<label>-00000.json> created by run_download_data.py).
    nested_dir = data_dir / label
    if nested_dir.is_dir():
        shards = sorted(p for p in nested_dir.glob(f"{label}-*.json") if p.is_file())
        if shards:
            return shards

        single = nested_dir / f"{label}.json"
        if single.exists():
            return [single]

    return []


def _row_text(obj: Dict[str, Any], *, max_chars: Optional[int] = None) -> Optional[str]:
    t = obj.get("text")
    if not isinstance(t, str):
        return None

    text = t.strip()
    if not text:
        return None

    lower = text.lower()
    if lower in {"nan", "none", "null"}:
        return None

    if max_chars is not None and len(text) > int(max_chars):
        return None

    return text


def load_label_texts(
    data_dir: str | Path,
    label: str,
    *,
    limit: Optional[int] = None,
    max_chars: Optional[int] = None,
    show_progress: bool = True,
) -> List[str]:
    """
    Read texts for one label from JSON-array file(s) in data_dir.
    """
    d = Path(data_dir)
    files = _list_label_files(d, label)
    if not files:
        raise FileNotFoundError(str(d / f"{label}.json"))

    out: List[str] = []
    want = None if limit is None else int(limit)

    for fp in files:
        if want is not None and len(out) >= want:
            break

        rows = _read_json_array(fp, show_progress=show_progress, desc=f"load[{label}:{fp.name}]")
        for r in rows:
            t = _row_text(r, max_chars=max_chars)
            if t is None:
                continue
            out.append(t)
            if want is not None and len(out) >= want:
                break

    return out


def interleave_labels(
    data_dir: str | Path,
    *,
    labels: Sequence[str],
    total_cap: Optional[int],
    seed: int,
    per_label_cap: Optional[int] = None,
    max_chars: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "mix",
) -> Tuple[List[str], List[str]]:
    """
    Load texts from each label and interleave (round-robin) after shuffling within label.

    This avoids ordering artifacts like "all normal then all malicious".

    Returns (texts, labels).
    """
    rng = random.Random(int(seed))

    buckets: Dict[str, List[str]] = {}
    for lab in labels:
        texts = load_label_texts(
            data_dir,
            lab,
            limit=per_label_cap,
            max_chars=max_chars,
            show_progress=show_progress,
        )
        rng.shuffle(texts)
        buckets[lab] = texts

    out_texts: List[str] = []
    out_labels: List[str] = []

    rr = list(labels)
    idx = {lab: 0 for lab in rr}

    bar = tqdm(
        total=total_cap,
        desc=desc,
        unit="rows",
        leave=True,
        disable=not show_progress,
    )
    try:
        while True:
            progressed = False
            for lab in rr:
                i = idx[lab]
                if i < len(buckets[lab]):
                    out_texts.append(buckets[lab][i])
                    out_labels.append(lab)
                    idx[lab] = i + 1
                    progressed = True
                    bar.update(1)
                    if total_cap is not None and len(out_texts) >= int(total_cap):
                        return out_texts, out_labels
            if not progressed:
                break
    finally:
        bar.close()

    return out_texts, out_labels


# Back-compat alias used by older run files
def interleave_label_files(
    data_dir: str | Path,
    *,
    labels: Sequence[str],
    per_label_cap: Optional[int],
    total_cap: Optional[int],
    seed: int,
    max_chars: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "mix",
) -> Tuple[List[str], List[str]]:
    return interleave_labels(
        data_dir,
        labels=labels,
        total_cap=total_cap,
        seed=seed,
        per_label_cap=per_label_cap,
        max_chars=max_chars,
        show_progress=show_progress,
        desc=desc,
    )
