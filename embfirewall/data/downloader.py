from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from .sources import make_source
from .types import TextFilters


def _enable_hf_hub_download_progress_bars() -> None:
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    try:
        from huggingface_hub.utils.logging import enable_progress_bars

        enable_progress_bars()
    except Exception:
        pass


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _safe_load_json_any(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        import orjson

        data = orjson.dumps(obj)
        tmp.write_bytes(data + b"\n")
    except Exception:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    tmp.replace(path)



def _count_rows_in_shard(path: Path) -> int:
    obj = _safe_load_json_any(path)
    return len(obj) if isinstance(obj, list) else 0


@dataclass
class _ShardState:
    out_dir: Path
    label: str
    shard_size: int
    shard_idx: int
    shard_rows: int
    shard_counts: Dict[int, int]
    shard_buf: List[Dict[str, Any]] = field(default_factory=list)

    def shard_path(self, idx: int) -> Path:
        return self.out_dir / f"{self.label}-{idx:05d}.json"

    def current_path(self) -> Path:
        return self.shard_path(self.shard_idx)


def _list_shards(out_dir: Path, label: str) -> List[Path]:
    return sorted(p for p in out_dir.glob(f"{label}-*.json") if p.is_file())


def _delete_shards(out_dir: Path, label: str) -> None:
    for p in _list_shards(out_dir, label):
        try:
            p.unlink()
        except Exception:
            pass


def _init_shard_state(*, out_dir: Path, label: str, shard_size: int, overwrite: bool) -> Tuple[_ShardState, int, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        _delete_shards(out_dir, label)
        return _ShardState(out_dir, label, int(shard_size), 0, 0, {}, []), 0, False

    shards = _list_shards(out_dir, label)
    if not shards:
        return _ShardState(out_dir, label, int(shard_size), 0, 0, {}, []), 0, False

    meta = _safe_load_json(out_dir / "metadata.json")
    meta_counts: Dict[int, int] = {}
    try:
        for ent in (meta.get("labels", {}).get(label, {}) or {}).get("shards", []) or []:
            if isinstance(ent, dict) and "file" in ent and "rows" in ent:
                idx = int(str(ent["file"]).split("-")[-1].split(".")[0])
                meta_counts[idx] = int(ent["rows"])
    except Exception:
        meta_counts = {}

    counts: Dict[int, int] = {}
    total = 0
    last_idx = -1
    last_rows = 0

    for p in shards:
        try:
            idx = int(p.stem.split("-")[-1])
        except Exception:
            continue
        n = meta_counts.get(idx, _count_rows_in_shard(p))
        counts[idx] = n
        total += n
        if idx > last_idx:
            last_idx = idx
            last_rows = n

    if last_idx < 0:
        return _ShardState(out_dir, label, int(shard_size), 0, 0, {}, []), 0, True

    shard_idx = last_idx
    shard_rows = last_rows
    shard_buf: List[Dict[str, Any]] = []

    if 0 < shard_rows < int(shard_size):
        obj = _safe_load_json_any(out_dir / f"{label}-{shard_idx:05d}.json")
        if isinstance(obj, list):
            shard_buf = obj
            shard_rows = len(shard_buf)

    if shard_rows >= int(shard_size):
        shard_idx += 1
        shard_rows = 0
        shard_buf = []

    state = _ShardState(
        out_dir=out_dir,
        label=label,
        shard_size=int(shard_size),
        shard_idx=shard_idx,
        shard_rows=shard_rows,
        shard_counts=counts,
        shard_buf=shard_buf,
    )
    return state, total, True


def _flush_json_sharded(
    state: _ShardState, rows: Sequence[Dict[str, Any]], *, final: bool = False
) -> Tuple[int, int, _ShardState]:
    """
    Shards are stored as JSON arrays. We keep the current shard in memory and
    write it once when it fills (and once more at the very end for the final partial shard).
    """
    bytes_written = 0
    i = 0
    n_rows = len(rows)

    while i < n_rows:
        if state.shard_rows >= state.shard_size:
            p = state.current_path()
            _write_json_file(p, state.shard_buf)
            try:
                bytes_written += p.stat().st_size
            except Exception:
                pass
            state.shard_counts[state.shard_idx] = state.shard_rows

            state.shard_idx += 1
            state.shard_rows = 0
            state.shard_buf = []

        cap = max(0, state.shard_size - state.shard_rows)
        if cap <= 0:
            continue

        chunk = rows[i : i + cap]
        state.shard_buf.extend(chunk)
        state.shard_rows += len(chunk)
        i += len(chunk)

    if final and state.shard_rows > 0:
        p = state.current_path()
        _write_json_file(p, state.shard_buf)
        try:
            bytes_written += p.stat().st_size
        except Exception:
            pass
        state.shard_counts[state.shard_idx] = state.shard_rows

    total_bytes = 0
    for sp in _list_shards(state.out_dir, state.label):
        try:
            total_bytes += sp.stat().st_size
        except Exception:
            pass

    return bytes_written, total_bytes, state


def _update_metadata(
    *,
    out_dir: Path,
    label: str,
    shard_size: int,
    seed: int,
    text_filters: Any,
    overwrite: bool,
    shard_counts: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    meta_path = out_dir / "metadata.json"
    meta = _safe_load_json(meta_path)

    labels = meta.get("labels")
    if not isinstance(labels, dict):
        labels = {}
        meta["labels"] = labels

    shards = _list_shards(out_dir, label)
    shard_entries: List[Dict[str, Any]] = []
    total_rows = 0
    total_bytes = 0

    for p in shards:
        try:
            idx = int(p.stem.split("-")[-1])
        except Exception:
            idx = -1

        n = 0
        if shard_counts is not None and idx >= 0:
            n = int(shard_counts.get(idx, 0) or 0)
        if n <= 0:
            n = _count_rows_in_shard(p)

        b = 0
        try:
            b = p.stat().st_size
        except Exception:
            pass

        shard_entries.append({"file": p.name, "rows": n, "bytes": b})
        total_rows += n
        total_bytes += b

    labels[label] = {
        "label": label,
        "format": "json",
        "container": "array",
        "sharded": True,
        "shard_size": int(shard_size),
        "total_rows": int(total_rows),
        "total_bytes": int(total_bytes),
        "shards": shard_entries,
        "updated_at": _utc_iso(),
    }

    meta["schema_version"] = 1
    meta["updated_at"] = _utc_iso()
    meta["out_dir"] = str(out_dir)
    meta["seed"] = int(seed)
    meta["overwrite"] = bool(overwrite)

    try:
        meta["text_filters"] = getattr(text_filters, "to_dict")()
    except Exception:
        try:
            meta["text_filters"] = dict(text_filters)
        except Exception:
            meta["text_filters"] = str(text_filters)

    agg_rows = 0
    agg_bytes = 0
    for v in labels.values():
        if isinstance(v, dict):
            agg_rows += int(v.get("total_rows", 0) or 0)
            agg_bytes += int(v.get("total_bytes", 0) or 0)
    meta["total_rows"] = int(agg_rows)
    meta["total_bytes"] = int(agg_bytes)

    _write_json(meta_path, meta)
    return meta


class DatasetDownloader:
    def __init__(
        self,
        seed: int,
        text_filters: TextFilters,
        flush_every: int = 1000,
        overwrite: bool = True,
        shard_size: int = 100_000,
    ):
        self.seed = int(seed)
        self.text_filters = text_filters
        self.flush_every = int(flush_every)
        self.overwrite = bool(overwrite)
        self.shard_size = int(shard_size)
        set_global_seed(self.seed)
        _enable_hf_hub_download_progress_bars()

    def download_label(self, *, label: str, label_cfg: Dict[str, Any], out_path: str) -> Dict[str, Any]:
        """
        Adds per-source caps:
          labels.<label>.sources[].max_rows (or cap) : int, max kept rows from that source.
        """
        target_raw = label_cfg.get("target", None)
        target: Optional[int] = None
        if target_raw is not None:
            target_raw = int(target_raw)
            if target_raw > 0:
                target = target_raw

        sources = label_cfg.get("sources") or []
        if not sources:
            raise ValueError(f"labels.{label}.sources missing/empty")

        op = Path(out_path)
        out_dir = op if (op.suffix == "" or op.is_dir()) else op.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        state, existing, resumed = _init_shard_state(
            out_dir=out_dir, label=label, shard_size=self.shard_size, overwrite=self.overwrite
        )

        print(
            f"[download_label] label={label} overwrite={self.overwrite} resumed={resumed} existing={existing} "
            f"target={'ALL' if target is None else target} shard_size={self.shard_size}"
        )

        if target is not None and existing >= target:
            _update_metadata(
                out_dir=out_dir,
                label=label,
                shard_size=self.shard_size,
                seed=self.seed,
                text_filters=self.text_filters,
                overwrite=self.overwrite,
                shard_counts=state.shard_counts,
            )
            return {
                "label": label,
                "target": target,
                "written": existing,
                "resumed": resumed,
                "out_dir": str(out_dir),
            }

        buf: List[Dict[str, Any]] = []
        wrote = existing

        processed_pbar = tqdm(desc=f"rows_processed[{label}]", unit="rows", leave=True)

        try:
            for src_spec in sources:
                if target is not None and wrote >= target:
                    break

                src_cap_raw = src_spec.get("max_rows", src_spec.get("cap", None))
                src_cap: Optional[int] = None
                if src_cap_raw is not None:
                    src_cap_raw = int(src_cap_raw)
                    if src_cap_raw > 0:
                        src_cap = src_cap_raw

                kind = src_spec.get("kind", "hf")
                spec = dict(src_spec)
                spec.pop("kind", None)
                spec.pop("max_rows", None)
                spec.pop("cap", None)

                name = spec.get("name") or spec.get("repo_id") or spec.get("dataset") or kind

                print(f"[download_label] start source={name} max_rows={'ALL' if src_cap is None else src_cap}")
                t0 = time.time()

                src = make_source(kind, **spec)

                kept = 0
                processed = 0

                for ex in src.iter_examples():
                    if target is not None and wrote >= target:
                        break
                    if src_cap is not None and kept >= src_cap:
                        break

                    processed += 1
                    processed_pbar.update(1)

                    raw = ex.text
                    t = self.text_filters.apply(raw)
                    if t is None:
                        continue

                    buf.append(
                        {
                            "id": uuid.uuid4().hex,
                            "text": t,
                            "source": getattr(src, "name", name),
                        }
                    )
                    kept += 1
                    wrote += 1

                    processed_pbar.set_postfix_str(
                        f"written={wrote}{'' if target is None else f'/{target}'} "
                        f"kept_src={kept}{'' if src_cap is None else f'/{src_cap}'} shard={state.shard_idx:05d}"
                    )

                    if len(buf) >= self.flush_every:
                        flushed_n = len(buf)
                        _bytes, _total_bytes, state = _flush_json_sharded(state, buf, final=False)
                        buf.clear()
                        print(
                            f"[download_label] label={label} flushed rows={flushed_n} wrote={wrote}"
                            f"{'' if target is None else f'/{target}'} shard={state.shard_idx:05d}"
                        )

                dt = time.time() - t0
                print(f"[download_label] done source={name} processed={processed} kept={kept} time_s={dt:.1f}")

            if buf:
                flushed_n = len(buf)
                _bytes, _total_bytes, state = _flush_json_sharded(state, buf, final=False)
                buf.clear()
                print(
                    f"[download_label] label={label} final pre-write flush rows={flushed_n} wrote={wrote}"
                    f"{'' if target is None else f'/{target}'} shard={state.shard_idx:05d}"
                )

            _bytes, _total_bytes, state = _flush_json_sharded(state, [], final=True)

        finally:
            processed_pbar.close()

        if target is not None and wrote != target:
            raise RuntimeError(f"label={label} wrote={wrote} expected={target}. Add sources or reduce target.")

        meta = _update_metadata(
            out_dir=out_dir,
            label=label,
            shard_size=self.shard_size,
            seed=self.seed,
            text_filters=self.text_filters,
            overwrite=self.overwrite,
            shard_counts=state.shard_counts,
        )

        return {
            "label": label,
            "target": ("ALL" if target is None else target),
            "written": wrote,
            "resumed": resumed,
            "out_dir": str(out_dir),
            "metadata_path": str(out_dir / "metadata.json"),
            "label_meta": meta.get("labels", {}).get(label, {}),
        }
