# src/Caste_Project/segment/segment_xlsx.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class XlsxSegmentConfig:
    """
    XLSX segmentation = rows_df -> entries_df

    Inputs (from ingest):
      - excel_documents.parquet
      - excel_rows.parquet
      (excel_sheets.parquet is not required here)

    Output:
      - excel_entries.parquet

    Note: Mirrors segment_csv.py shape:
      doc_id, entry_num, entry_text, row_num, seg_method, seg_confidence, flags_json, error
    """
    min_entry_chars: int = 5
    max_entry_chars_for_full_conf: int = 500
    dedupe: bool = True


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def segment_xlsx_rows_to_entries(
    documents_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    cfg: XlsxSegmentConfig = XlsxSegmentConfig(),
) -> pd.DataFrame:
    """
    Returns entries_df columns:
      doc_id, entry_num, entry_text, row_num, seg_method, seg_confidence, flags_json, error

    Notes:
      - Excel rows often include sheet context; if present we preserve it in flags_json,
        and (optionally) pass through a 'sheet_name' column for convenience.
      - entry_num is re-assigned to be contiguous per doc_id at the end (mirrors segment_csv).
    """
    if rows_df is None or rows_df.empty:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "entry_num",
                "entry_text",
                "row_num",
                "seg_method",
                "seg_confidence",
                "flags_json",
                "error",
            ]
        )

    df = rows_df.copy()

    if "doc_id" not in df.columns:
        raise KeyError("excel_rows.parquet must contain a 'doc_id' column.")

    # Common columns emitted by your excel ingest
    row_num_col = _pick_first_present(df, ["row_num", "record_num", "idx", "index"])
    row_text_col = _pick_first_present(df, ["row_text", "record_text", "text"])
    sheet_col = _pick_first_present(df, ["sheet_name", "sheet", "tab", "worksheet"])

    # Build entry_text
    if row_text_col is None:
        # fallback: derive entry_text from remaining columns (excluding obvious meta cols)
        meta_cols = {c for c in ["doc_id", row_num_col, sheet_col, "error"] if c}
        text_cols = [c for c in df.columns if c not in meta_cols]
        df["entry_text"] = df[text_cols].astype(str).agg(" | ".join, axis=1)
        has_row_text = False
    else:
        df["entry_text"] = df[row_text_col].fillna("").astype(str)
        has_row_text = True

    # Establish entry_num (prefer row_num if clean numeric, else cumcount)
    if row_num_col:
        df["entry_num"] = pd.to_numeric(df[row_num_col], errors="coerce")
        if df["entry_num"].isna().any():
            df = df.sort_values(["doc_id"]).reset_index(drop=True)
            df["entry_num"] = df.groupby("doc_id").cumcount()
        else:
            df["entry_num"] = df["entry_num"].astype(int)
    else:
        df = df.sort_values(["doc_id"]).reset_index(drop=True)
        df["entry_num"] = df.groupby("doc_id").cumcount()

    # Confidence heuristic (same idea as CSV)
    lens = df["entry_text"].fillna("").astype(str).str.len()
    df["seg_confidence"] = 0.4 + 0.55 * (
        (lens.clip(lower=cfg.min_entry_chars, upper=cfg.max_entry_chars_for_full_conf) - cfg.min_entry_chars)
        / max(1, (cfg.max_entry_chars_for_full_conf - cfg.min_entry_chars))
    )
    df["seg_confidence"] = df["seg_confidence"].clip(lower=0.4, upper=0.95)

    df["seg_method"] = "tier1_xlsx_rows"

    # Attach doc-level flags (light metadata if present)
    doc_meta_cols = [
        c
        for c in ["total_rows", "num_sheets", "source_rel_path", "source_file", "error"]
        if c in documents_df.columns
    ]
    if documents_df is None or documents_df.empty:
        docs_meta = pd.DataFrame({"doc_id": df["doc_id"].unique()})
    else:
        keep_cols = ["doc_id"] + [c for c in doc_meta_cols if c != "error"]
        docs_meta = documents_df[keep_cols].drop_duplicates("doc_id")

    df = df.merge(docs_meta, on="doc_id", how="left")

    def make_flags(row) -> str:
        flags: Dict[str, object] = {
            "doc_id": row.get("doc_id"),
            "has_row_text": has_row_text,
            "sheet_name_present": sheet_col is not None,
            "sheet_name": row.get(sheet_col) if sheet_col and sheet_col in df.columns else None,
            # document-level meta if present
            "ingest_total_rows": row.get("total_rows", None),
            "ingest_num_sheets": row.get("num_sheets", None),
        }
        return json.dumps(flags, ensure_ascii=False)

    df["flags_json"] = df.apply(make_flags, axis=1)

    # error passthrough
    if "error" not in df.columns:
        df["error"] = None

    # Optional dedupe (include sheet context if present)
    if cfg.dedupe:
        # Base key: doc_id + entry_text + entry_num
        base = (
            df["doc_id"].astype(str)
            + "|"
            + df["entry_text"].fillna("").astype(str).apply(_stable_hash)
            + "|"
            + df["entry_num"].astype(str)
        )

        # Include sheet + row_num if present to avoid collapsing legitimate duplicates
        if sheet_col and sheet_col in df.columns:
            base = base + "|" + df[sheet_col].fillna("").astype(str)
        if row_num_col and row_num_col in df.columns:
            base = base + "|" + df[row_num_col].fillna(-1).astype(str)

        df["_hash"] = base.apply(_stable_hash)

        dedupe_subset = ["doc_id", "_hash"]
        df = df.drop_duplicates(subset=dedupe_subset, keep="first").drop(columns=["_hash"])

    # Normalize output columns (keep required ones; optionally pass through sheet_name)
    out_cols = [
        "doc_id",
        "entry_num",
        "entry_text",
        "seg_method",
        "seg_confidence",
        "flags_json",
        "error",
    ]

    # Normalize row_num field in output
    if row_num_col and row_num_col in df.columns:
        df["row_num"] = df[row_num_col]
    else:
        df["row_num"] = None
    out_cols.insert(3, "row_num")

    # Optional: keep sheet_name as a convenience column (won't break downstream)
    if sheet_col and sheet_col in df.columns:
        df["sheet_name"] = df[sheet_col]
        out_cols.insert(4, "sheet_name")

    df = df[out_cols].sort_values(["doc_id", "entry_num"]).reset_index(drop=True)

    # Enforce contiguous numbering per doc (mirrors segment_csv)
    df["entry_num"] = df.groupby("doc_id").cumcount()

    return df


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        required=True,
        help="Folder containing excel_documents.parquet and excel_rows.parquet",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Where to write excel_entries.parquet",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents_path = in_dir / "excel_documents.parquet"
    rows_path = in_dir / "excel_rows.parquet"

    if not documents_path.exists():
        raise FileNotFoundError(f"Missing {documents_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing {rows_path}")

    documents_df = pd.read_parquet(documents_path)
    rows_df = pd.read_parquet(rows_path)

    cfg = XlsxSegmentConfig()
    entries_df = segment_xlsx_rows_to_entries(documents_df, rows_df, cfg)

    out_path = out_dir / "excel_entries.parquet"
    entries_df.to_parquet(out_path, index=False)

    # 1-based for printing only
    if not entries_df.empty:
        entries_df["entry_num_1based"] = entries_df["entry_num"] + 1

    print(f"Saved: {out_path}")
    if entries_df.empty:
        print("No XLSX entries produced (rows_df empty).")
    else:
        cols = ["doc_id", "entry_num_1based", "seg_method", "seg_confidence", "error"]
        if "sheet_name" in entries_df.columns:
            cols.insert(2, "sheet_name")
        print(entries_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
