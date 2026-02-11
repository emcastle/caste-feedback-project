from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ExcelReadConfig:
    """
    Configuration for reading Excel files.
    """
    sheet_names: Optional[List[str]] = None      # None = all sheets
    header: int | None = 0                       # 0 = first row is header, None = no header
    dtype_as_str: bool = True                    # True = read everything as string
    keep_default_na: bool = True
    na_values: Optional[List[str]] = None
    make_row_text: bool = True
    row_text_max_chars: int = 20000              # safety cap
    engine_xlsx: str = "openpyxl"                # approved in env


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
    parts: List[str] = []
    for k, v in row.items():
        if k in {"doc_id", "sheet_name", "row_num"}:
            continue
        val = _safe_to_str(v)
        if not val:
            continue
        parts.append(f"{k}={val}")
    text = " | ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + " ...[TRUNCATED]"
    return text


def _read_excel_all_sheets(path: Path, cfg: ExcelReadConfig) -> Dict[str, pd.DataFrame]:
    """
    Internal helper: reads an Excel workbook and returns a dict of sheet_name -> DataFrame.
    """
    suffix = path.suffix.lower()

    # Choose engine based on extension
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        engine = cfg.engine_xlsx
    elif suffix == ".xls":
        # .xls requires xlrd (not always available). We fail with a clear message.
        # If later add xlrd to environment.yml, can set engine="xlrd" here.
        raise RuntimeError(
            "Reading .xls requires the 'xlrd' engine which is not installed. "
            "Either convert to .xlsx or add xlrd to environment.yml (if available in Nexus)."
        )
    else:
        raise RuntimeError(f"Unsupported Excel extension: {suffix}")

    read_kwargs = dict(
        sheet_name=cfg.sheet_names if cfg.sheet_names else None,
        header=cfg.header,
        keep_default_na=cfg.keep_default_na,
        na_values=cfg.na_values,
        engine=engine,
    )

    if cfg.dtype_as_str:
        read_kwargs["dtype"] = str

    # Returns dict if sheet_name=None or list; returns DataFrame if sheet_name is a string
    result = pd.read_excel(path, **read_kwargs)

    if isinstance(result, pd.DataFrame):
        # sheet_names was a single name; normalize to dict
        only_name = cfg.sheet_names[0] if cfg.sheet_names else "Sheet1"
        return {only_name: result}

    return result


def extract_excel_to_relational(
    excel_path: Path,
    source_rel_path: str,
    cfg: ExcelReadConfig = ExcelReadConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract an Excel workbook into:
      - documents_df: 1 row per workbook
      - sheets_df:    1 row per sheet (shape + columns)
      - rows_df:      1 row per row, tagged with (doc_id, sheet_name, row_num), plus optional row_text
    """
    excel_path = excel_path.resolve()
    doc_id = sha256_file(excel_path)

    source_file = excel_path.name
    source_ext = excel_path.suffix.lower()
    source_type = "excel_workbook"
    extractor_used = "pandas.read_excel"

    try:
        sheets_map = _read_excel_all_sheets(excel_path, cfg)

        sheets_records = []
        rows_frames = []

        for sheet_name, df in sheets_map.items():
            # Normalize columns to strings (safe for JSON + row_text)
            df = df.copy()
            df.columns = [str(c) for c in df.columns]

            # Build sheet metadata
            num_rows = int(len(df))
            num_cols = int(df.shape[1])
            sheets_records.append(
                {
                    "doc_id": doc_id,
                    "sheet_name": sheet_name,
                    "sheet_index": len(sheets_records),
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "columns_json": json.dumps([str(c) for c in df.columns], ensure_ascii=False),
                    "error": None,
                }
            )

            # Build rows table for this sheet
            rows_df = df.copy()
            rows_df.insert(0, "row_num", range(len(rows_df)))
            rows_df.insert(0, "sheet_name", sheet_name)
            rows_df.insert(0, "doc_id", doc_id)

            if cfg.make_row_text:
                rows_df.insert(
                    3,
                    "row_text",
                    rows_df.apply(lambda r: _build_row_text(r, cfg.row_text_max_chars), axis=1),
                )
            else:
                rows_df.insert(3, "row_text", None)

            rows_df["error"] = None
            rows_frames.append(rows_df)

        sheets_df = pd.DataFrame(sheets_records)
        rows_df_all = pd.concat(rows_frames, ignore_index=True) if rows_frames else pd.DataFrame(
            columns=["doc_id", "sheet_name", "row_num", "row_text", "error"]
        )

        total_rows = int(sheets_df["num_rows"].sum()) if not sheets_df.empty else 0
        total_sheets = int(len(sheets_df))

        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_sheets": total_sheets,
                    "total_rows": total_rows,
                    "sheet_names_json": json.dumps(list(sheets_map.keys()), ensure_ascii=False),
                    "extractor_used": extractor_used,
                    "error": None,
                }
            ]
        )

        return documents_df, sheets_df, rows_df_all

    except Exception as e:
        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_sheets": None,
                    "total_rows": None,
                    "sheet_names_json": None,
                    "extractor_used": extractor_used,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        sheets_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "sheet_name": None,
                    "sheet_index": None,
                    "num_rows": None,
                    "num_cols": None,
                    "columns_json": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        rows_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "sheet_name": None,
                    "row_num": None,
                    "row_text": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        return documents_df, sheets_df, rows_df
