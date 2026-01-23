from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class CsvSegmentConfig:
    """
    CSV segmentation = rows_df -> entries_df

    Inputs (from ingest):
      - csv_documents.parquet
      - csv_rows.parquet

    Output:
      - csv_entries.parquet

    Note: We keep things consistent with other segmenters:
      doc_id, entry_num, entry_text, seg_method, seg_confidence, flags_json, error
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


def segment_csv_rows_to_entries(
    documents_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    cfg: CsvSegmentConfig = CsvSegmentConfig(),
) -> pd.DataFrame:
    """
    Returns entries_df columns:
      doc_id, entry_num, entry_text, row_num, seg_method, seg_confidence, flags_json, error
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
        raise KeyError("csv_rows.parquet must contain a 'doc_id' column.")

    row_num_col = _pick_first_present(df, ["row_num", "record_num", "idx", "index"])
    row_text_col = _pick_first_present(df, ["row_text", "record_text", "text"])

    if row_text_col is None:
        # fallback: derive entry_text from remaining columns (excluding obvious meta cols)
        meta_cols = {c for c in ["doc_id", row_num_col, "error"] if c}
        text_cols = [c for c in df.columns if c not in meta_cols]
        df["entry_text"] = df[text_cols].astype(str).agg(" | ".join, axis=1)
    else:
        df["entry_text"] = df[row_text_col].fillna("").astype(str)

    # entry_num: prefer row_num if present, else regenerate
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

    # confidence heuristic
    lens = df["entry_text"].fillna("").astype(str).str.len()
    df["seg_confidence"] = 0.4 + 0.55 * (
        (lens.clip(lower=cfg.min_entry_chars, upper=cfg.max_entry_chars_for_full_conf) - cfg.min_entry_chars)
        / max(1, (cfg.max_entry_chars_for_full_conf - cfg.min_entry_chars))
    )
    df["seg_confidence"] = df["seg_confidence"].clip(lower=0.4, upper=0.95)

    df["seg_method"] = "tier1_csv_rows"

    # doc-level flags (attach light metadata if present)
    doc_meta_cols = [c for c in ["num_rows", "num_cols", "source_rel_path", "source_file", "columns_json"] if c in documents_df.columns]
    docs_meta = documents_df[["doc_id"] + doc_meta_cols].drop_duplicates("doc_id") if not documents_df.empty else pd.DataFrame({"doc_id": df["doc_id"].unique()})
    df = df.merge(docs_meta, on="doc_id", how="left")

    def make_flags(row) -> str:
        flags: Dict[str, object] = {
            "doc_id": row["doc_id"],
            "ingest_num_rows": row.get("num_rows", None),
            "ingest_num_cols": row.get("num_cols", None),
            "has_row_text": row_text_col is not None,
        }
        return json.dumps(flags, ensure_ascii=False)

    df["flags_json"] = df.apply(make_flags, axis=1)

    # error passthrough
    if "error" not in df.columns:
        df["error"] = None

    # optional dedupe
    if cfg.dedupe:
        df["_hash"] = (
            df["doc_id"].astype(str)
            + "|"
            + df["entry_text"].fillna("").astype(str).apply(_stable_hash)
            + "|"
            + df["entry_num"].astype(str)
        ).apply(_stable_hash)

        # If row_num exists, include it in dedupe key to avoid collapsing legitimate duplicates
        dedupe_subset = ["doc_id", "_hash"]
        if row_num_col and row_num_col in df.columns:
            dedupe_subset = ["doc_id", row_num_col, "_hash"]

        df = df.drop_duplicates(subset=dedupe_subset, keep="first").drop(columns=["_hash"])

    # keep normalized columns
    out_cols = [
        "doc_id",
        "entry_num",
        "entry_text",
        "seg_method",
        "seg_confidence",
        "flags_json",
        "error",
    ]
    if row_num_col and row_num_col in df.columns:
        df["row_num"] = df[row_num_col]
        out_cols.insert(3, "row_num")
    else:
        df["row_num"] = None
        out_cols.insert(3, "row_num")

    df = df[out_cols].sort_values(["doc_id", "entry_num"]).reset_index(drop=True)

    # enforce contiguous numbering per doc
    df["entry_num"] = df.groupby("doc_id").cumcount()

    return df


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing csv_documents.parquet and csv_rows.parquet")
    ap.add_argument("--out_dir", required=True, help="Where to write csv_entries.parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents_df = pd.read_parquet(in_dir / "csv_documents.parquet")
    rows_df = pd.read_parquet(in_dir / "csv_rows.parquet")

    cfg = CsvSegmentConfig()
    entries_df = segment_csv_rows_to_entries(documents_df, rows_df, cfg)

    out_path = out_dir / "csv_entries.parquet"
    entries_df.to_parquet(out_path, index=False)

    # 1-based for printing only
    if not entries_df.empty:
        entries_df["entry_num_1based"] = entries_df["entry_num"] + 1

    print(f"Saved: {out_path}")
    if entries_df.empty:
        print("No CSV entries produced (rows_df empty).")
    else:
        print(
            entries_df[
                ["doc_id", "entry_num_1based", "seg_method", "seg_confidence", "error"]
            ].head(25).to_string(index=False)
        )


if __name__ == "__main__":
    main_cli()
