from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class JsonSegmentConfig:
    """
    JSON segmentation is basically "records -> entries".

    Assumes ingest produced:
      - json_documents.parquet
      - json_records.parquet

    We normalize records into entries_df with consistent columns used by other segmenters.
    """
    min_entry_chars: int = 5          # too-short entries get lower confidence
    max_entry_chars_for_full_conf: int = 500
    dedupe: bool = True


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def segment_json_records_to_entries(
    documents_df: pd.DataFrame,
    records_df: pd.DataFrame,
    cfg: JsonSegmentConfig = JsonSegmentConfig(),
) -> pd.DataFrame:
    """
    Returns entries_df with consistent columns:
      doc_id, entry_num, entry_text, entry_json, json_path, record_type,
      seg_method, seg_confidence, flags_json, error
    """
    if records_df is None or records_df.empty:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "entry_num",
                "entry_text",
                "entry_json",
                "json_path",
                "record_type",
                "seg_method",
                "seg_confidence",
                "flags_json",
                "error",
            ]
        )

    rec = records_df.copy()

    # ---- identify key columns produced by your ingest ----
    # Your smoke prints suggested: record_num, record_type, record_text, error
    record_num_col = _pick_first_present(rec, ["record_num", "row_num", "idx", "index"])
    record_text_col = _pick_first_present(rec, ["record_text", "row_text", "text"])
    record_type_col = _pick_first_present(rec, ["record_type", "type"])
    record_json_col = _pick_first_present(rec, ["record_json", "json", "record", "raw_json"])
    record_path_col = _pick_first_present(rec, ["record_path", "json_path", "path"])

    if "doc_id" not in rec.columns:
        raise KeyError("json_records.parquet must contain a 'doc_id' column.")

    if record_text_col is None and record_json_col is None:
        raise KeyError(
            "json_records.parquet must contain at least one of: "
            "record_text/row_text OR record_json/raw_json."
        )

    # ---- create entry_text ----
    if record_text_col is None:
        # fallback: derive text from json string
        rec["entry_text"] = rec[record_json_col].astype(str)
    else:
        rec["entry_text"] = rec[record_text_col].fillna("").astype(str)

    # ---- create entry_json ----
    if record_json_col is None:
        rec["entry_json"] = None
    else:
        rec["entry_json"] = rec[record_json_col].astype(str)

    # ---- json_path + record_type ----
    rec["json_path"] = rec[record_path_col].astype(str) if record_path_col else None
    rec["record_type"] = rec[record_type_col].astype(str) if record_type_col else None

    # ---- entry_num ----
    # Prefer record_num if present; else regenerate deterministically per doc
    if record_num_col:
        # ensure numeric-ish, but don't die if messy
        rec["entry_num"] = pd.to_numeric(rec[record_num_col], errors="coerce")
        if rec["entry_num"].isna().any():
            # fill missing by stable ordering within doc
            rec = rec.sort_values(["doc_id"]).reset_index(drop=True)
            rec["entry_num"] = rec.groupby("doc_id").cumcount()
        else:
            rec["entry_num"] = rec["entry_num"].astype(int)
    else:
        rec = rec.sort_values(["doc_id"]).reset_index(drop=True)
        rec["entry_num"] = rec.groupby("doc_id").cumcount()

    # ---- confidence heuristic ----
    # JSON is "high confidence" when we have a real record with some text.
    # Short/empty records get lower confidence.
    lens = rec["entry_text"].fillna("").astype(str).str.len()
    # scale: <= min_entry_chars => 0.4, >= max_entry_chars_for_full_conf => 0.95
    rec["seg_confidence"] = 0.4 + 0.55 * (lens.clip(lower=cfg.min_entry_chars, upper=cfg.max_entry_chars_for_full_conf) - cfg.min_entry_chars) / max(
        1, (cfg.max_entry_chars_for_full_conf - cfg.min_entry_chars)
    )
    rec["seg_confidence"] = rec["seg_confidence"].clip(lower=0.4, upper=0.95)

    rec["seg_method"] = "tier1_json_records"

    # ---- flags_json per doc ----
    # Attach document metadata if available
    doc_meta_cols = [c for c in ["top_level_type", "num_records", "source_rel_path", "source_file"] if c in documents_df.columns]
    docs_meta = documents_df[["doc_id"] + doc_meta_cols].drop_duplicates("doc_id") if not documents_df.empty else pd.DataFrame({"doc_id": rec["doc_id"].unique()})

    rec = rec.merge(docs_meta, on="doc_id", how="left")

    def make_flags(row) -> str:
        flags: Dict[str, object] = {
            "doc_id": row["doc_id"],
            "top_level_type": row.get("top_level_type", None),
            "ingest_num_records": row.get("num_records", None),
            "has_record_text": record_text_col is not None,
            "has_record_json": record_json_col is not None,
        }
        return json.dumps(flags, ensure_ascii=False)

    rec["flags_json"] = rec.apply(make_flags, axis=1)

    # ---- error passthrough ----
    if "error" not in rec.columns:
        rec["error"] = None

    # ---- optional dedupe ----
    if cfg.dedupe:
        rec["_hash"] = (
            rec["doc_id"].astype(str)
            + "|"
            + rec["entry_num"].astype(str)
            + "|"
            + rec["entry_text"].fillna("").astype(str).apply(_stable_hash)
            + "|"
            + rec["entry_json"].fillna("").astype(str).apply(_stable_hash)
        ).apply(_stable_hash)

        rec = rec.drop_duplicates(subset=["doc_id", "json_path", "_hash"], keep="first").drop(columns=["_hash"])

    # ---- final normalized columns ----
    keep_cols = [
        "doc_id",
        "entry_num",
        "entry_text",
        "entry_json",
        "json_path",
        "record_type",
        "seg_method",
        "seg_confidence",
        "flags_json",
        "error",
    ]
    for c in keep_cols:
        if c not in rec.columns:
            rec[c] = None

    rec = rec[keep_cols].sort_values(["doc_id", "entry_num"]).reset_index(drop=True)

    # Ensure entry_num is contiguous 0..N-1 per doc (deterministic final numbering)
    rec["entry_num"] = rec.groupby("doc_id").cumcount()

    return rec


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing json_documents.parquet and json_records.parquet")
    ap.add_argument("--out_dir", required=True, help="Where to write json_entries.parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents_df = pd.read_parquet(in_dir / "json_documents.parquet")
    records_df = pd.read_parquet(in_dir / "json_records.parquet")

    cfg = JsonSegmentConfig()
    entries_df = segment_json_records_to_entries(documents_df, records_df, cfg)

    out_path = out_dir / "json_entries.parquet"
    entries_df.to_parquet(out_path, index=False)

    # 1-based display columns (for printing only)
    if not entries_df.empty:
        entries_df["entry_num_1based"] = entries_df["entry_num"] + 1

    print(f"Saved: {out_path}")
    if entries_df.empty:
        print("No JSON entries produced (records_df empty).")
    else:
        print(
            entries_df[
                ["doc_id", "entry_num_1based", "seg_method", "seg_confidence", "error"]
            ].head(25).to_string(index=False)
        )


if __name__ == "__main__":
    main_cli()
