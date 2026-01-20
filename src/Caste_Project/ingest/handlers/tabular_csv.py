"""
document_csv.py

Purpose
-------
CSV ingestion handler focused on extraction (not template parsing).

This module:
1) Reads a .csv into a pandas DataFrame (with robust encoding fallback)
2) Produces two relational outputs:
   - documents_df: 1 row per CSV (identity + metadata + doc-level errors)
   - rows_df:      row-level records with stable ordering and optional per-row text serialization

Key Design Decisions
--------------------
- Extraction-only: no semantic parsing, no schema inference beyond reading the table.
- Stable identity: doc_id is sha256 of CSV bytes so it persists across runs.
- Preserve ordering: row_num preserves original row order for later processing.
- Flexible output: you can keep raw columns AND also generate a `row_text` string
  for LLM/NLP pipelines (optional).

Outputs (schemas)
-----------------
documents_df columns:
- doc_id
- source_file
- source_rel_path
- source_ext
- source_type
- num_rows
- num_cols
- columns_json
- extractor_used
- error

rows_df columns:
- doc_id
- row_num
- row_text            (nullable; only if enabled)
- <original columns>  (as read by pandas)
- error
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class CsvReadConfig:
    """
    Configuration for reading CSVs.
    """
    encoding: Optional[str] = None               # if None, try fallbacks
    delimiter: Optional[str] = None              # if None, let pandas infer (python engine)
    na_values: Optional[List[str]] = None        # extra NA markers
    keep_default_na: bool = True
    low_memory: bool = False
    make_row_text: bool = True                  # build a "k=v" row_text column for NLP
    row_text_max_chars: int = 20000             # safety cap


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_to_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    return s.strip()


def _build_row_text(row: pd.Series, max_chars: int) -> str:
    parts = []
    for k, v in row.items():
        if k == "doc_id" or k == "row_num":
            continue
        val = _safe_to_str(v)
        if not val:
            continue
        parts.append(f"{k}={val}")
    text = " | ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + " ...[TRUNCATED]"
    return text


def _read_csv_with_fallbacks(path: Path, cfg: CsvReadConfig) -> pd.DataFrame:
    encodings_to_try = [cfg.encoding] if cfg.encoding else ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    # If delimiter not provided, try common separators
    seps_to_try = [cfg.delimiter] if cfg.delimiter is not None else [",", "\t", "|", ";"]

    last_err = None
    for enc in encodings_to_try:
        for sep in seps_to_try:
            # Try C engine first
            try:
                read_kwargs = dict(
                    sep=sep,
                    engine="c",
                    keep_default_na=cfg.keep_default_na,
                    na_values=cfg.na_values,
                    low_memory=cfg.low_memory,
                )
                return pd.read_csv(path, encoding=enc, **read_kwargs)
            except Exception as e:
                last_err = e

            # Fallback to python engine (no low_memory)
            try:
                read_kwargs = dict(
                    sep=sep,
                    engine="python",
                    keep_default_na=cfg.keep_default_na,
                    na_values=cfg.na_values,
                )
                return pd.read_csv(path, encoding=enc, **read_kwargs)
            except Exception as e:
                last_err = e

    raise RuntimeError(
        f"Failed to read CSV with encodings {encodings_to_try} and seps {seps_to_try}: {last_err}"
    )


"""
def _read_csv_with_fallbacks(path: Path, cfg: CsvReadConfig) -> pd.DataFrame:
    
    #Robust CSV reader:
    #- allows delimiter inference using python engine if delimiter is None
    #- tries encoding fallbacks if encoding is None
    
    engine = "python" if cfg.delimiter is None else "c"

    read_kwargs = dict(
        sep=cfg.delimiter if cfg.delimiter else None,
        engine=engine,
        keep_default_na=cfg.keep_default_na,
        na_values=cfg.na_values
    )

    if engine == "c":
        read_kwargs["low_memory"] = cfg.low_memory

    if cfg.encoding:
        return pd.read_csv(path, encoding=cfg.encoding, **read_kwargs)

    # Common fallbacks for Windows/enterprise CSVs
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, **read_kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to read CSV with encodings {encodings_to_try}: {last_err}")
"""

def extract_csv_to_relational(
    csv_path: Path,
    source_rel_path: str,
    cfg: CsvReadConfig = CsvReadConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract a CSV into documents_df and rows_df.
    """
    csv_path = csv_path.resolve()
    doc_id = sha256_file(csv_path)

    source_file = csv_path.name
    source_ext = ".csv"
    source_type = "csv_document"
    extractor_used = "pandas.read_csv"

    try:
        df = _read_csv_with_fallbacks(csv_path, cfg)

        # Preserve order and attach doc_id
        rows_df = df.copy()
        rows_df.insert(0, "row_num", range(len(rows_df)))
        rows_df.insert(0, "doc_id", doc_id)

        # Optional row_text for NLP/LLM
        if cfg.make_row_text:
            rows_df.insert(
                2,
                "row_text",
                rows_df.apply(lambda r: _build_row_text(r, cfg.row_text_max_chars), axis=1),
            )
        else:
            rows_df.insert(2, "row_text", None)

        rows_df["error"] = None

        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_rows": int(len(df)),
                    "num_cols": int(df.shape[1]),
                    "columns_json": json.dumps([str(c) for c in df.columns], ensure_ascii=False),
                    "extractor_used": extractor_used,
                    "error": None,
                }
            ]
        )

        return documents_df, rows_df

    except Exception as e:
        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_rows": None,
                    "num_cols": None,
                    "columns_json": None,
                    "extractor_used": extractor_used,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        rows_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "row_num": None,
                    "row_text": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        return documents_df, rows_df