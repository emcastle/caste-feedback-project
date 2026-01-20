from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class JsonReadConfig:
    """
    Configuration for reading JSON/JSON5.
    """
    max_chars_per_record_text: int = 20000
    max_depth_for_flatten: int = 3          # flatten nested dicts up to this depth for record_text
    flatten_lists: bool = False             # if True, lists become indexed keys in flatten output
    make_record_text: bool = True


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _truncate(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + " ...[TRUNCATED]"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x).strip()
    except Exception:
        return ""


def _flatten_json(
    obj: Any,
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 3,
    flatten_lists: bool = False,
) -> Dict[str, str]:
    """
    Flatten nested dicts (and optionally lists) into a flat {key_path: value} map of strings.
    Intended for building record_text; not a perfect JSON flattener.
    """
    out: Dict[str, str] = {}

    if depth > max_depth:
        out[prefix.rstrip(".")] = _safe_str(obj)
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}."
            out.update(
                _flatten_json(
                    v,
                    prefix=key,
                    depth=depth + 1,
                    max_depth=max_depth,
                    flatten_lists=flatten_lists,
                )
            )
        return out

    if isinstance(obj, list):
        if not flatten_lists:
            out[prefix.rstrip(".")] = _safe_str(obj)
            return out
        for i, v in enumerate(obj):
            key = f"{prefix}{i}."
            out.update(
                _flatten_json(
                    v,
                    prefix=key,
                    depth=depth + 1,
                    max_depth=max_depth,
                    flatten_lists=flatten_lists,
                )
            )
        return out

    out[prefix.rstrip(".")] = _safe_str(obj)
    return out


def _record_text_from_obj(obj: Any, cfg: JsonReadConfig) -> str:
    flat = _flatten_json(
        obj,
        max_depth=cfg.max_depth_for_flatten,
        flatten_lists=cfg.flatten_lists,
    )

    parts: List[str] = []
    for k, v in flat.items():
        v = _safe_str(v)
        if not v:
            continue
        parts.append(f"{k}={v}")
    text = " | ".join(parts)
    return _truncate(text, cfg.max_chars_per_record_text)


def _load_json(path: Path) -> Any:
    """
    Loads JSON. If you later need JSON5 support, we can add it via json5 library.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_to_relational(
    json_path: Path,
    source_rel_path: str,
    cfg: JsonReadConfig = JsonReadConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract a JSON file into documents_df and records_df.

    records_df includes:
      - one row per top-level list element (if list)
      - otherwise one row for the top-level object (dict/scalar)
    """
    json_path = json_path.resolve()
    doc_id = sha256_file(json_path)

    source_file = json_path.name
    source_ext = json_path.suffix.lower() if json_path.suffix else ".json"
    source_type = "json"
    extractor_used = "json.load"

    try:
        obj = _load_json(json_path)

        records: List[Dict[str, Any]] = []

        if isinstance(obj, list):
            for i, item in enumerate(obj):
                record_json = json.dumps(item, ensure_ascii=False)
                record_text = _record_text_from_obj(item, cfg) if cfg.make_record_text else None
                records.append(
                    {
                        "doc_id": doc_id,
                        "record_num": i,
                        "record_type": type(item).__name__,
                        "record_json": _truncate(record_json, cfg.max_chars_per_record_text * 5),
                        "record_text": record_text,
                        "error": None,
                    }
                )
            num_records = len(obj)
            top_level_type = "list"

        else:
            record_json = json.dumps(obj, ensure_ascii=False)
            record_text = _record_text_from_obj(obj, cfg) if cfg.make_record_text else None
            records.append(
                {
                    "doc_id": doc_id,
                    "record_num": 0,
                    "record_type": type(obj).__name__,
                    "record_json": _truncate(record_json, cfg.max_chars_per_record_text * 5),
                    "record_text": record_text,
                    "error": None,
                }
            )
            num_records = 1
            top_level_type = type(obj).__name__

        records_df = pd.DataFrame(records)

        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "top_level_type": top_level_type,
                    "num_records": int(num_records),
                    "extractor_used": extractor_used,
                    "error": None,
                }
            ]
        )

        return documents_df, records_df

    except Exception as e:
        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "top_level_type": None,
                    "num_records": None,
                    "extractor_used": extractor_used,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        records_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "record_num": None,
                    "record_type": None,
                    "record_json": None,
                    "record_text": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        return documents_df, records_df
