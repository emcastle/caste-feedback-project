from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class PptxSegmentConfig:
    """
    PPTX segmentation (Option A):
      - 1 entry per slide (universal, deterministic)
      - keep field-like anchors for later parsing/validation

    Inputs (from ingest):
      - pptx_documents.parquet
      - pptx_blocks.parquet

    Output:
      - pptx_entries.parquet
    """
    include_ocr_text: bool = True
    include_notes: bool = True  # if you store notes as blocks
    min_entry_chars: int = 30
    dedupe: bool = True


# These are NOT used to split (we split by slide),
# but they are useful signals inside an entry.
FIELD_ANCHORS: Dict[str, re.Pattern] = {
    "from": re.compile(r"^\s*From\s*:\s*", re.IGNORECASE),
    "to": re.compile(r"^\s*To\s*:\s*", re.IGNORECASE),
    "subject": re.compile(r"^\s*Subject\s*:\s*", re.IGNORECASE),
    "date": re.compile(r"^\s*Date\s*:\s*", re.IGNORECASE),
    "cqas_id": re.compile(r"\bCQAS[-–—]?\s*\d{3,6}\b", re.IGNORECASE),
    "requestor_name": re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*", re.IGNORECASE),
    "incoming": re.compile(r"^\s*Incoming\b", re.IGNORECASE),
    "response": re.compile(r"^\s*Response\b", re.IGNORECASE),
}


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _field_anchor_counts_in_text(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, pat in FIELD_ANCHORS.items():
        counts[name] = len(pat.findall(text or ""))
    return counts


def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def segment_pptx_blocks_to_entries(
    documents_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    cfg: PptxSegmentConfig = PptxSegmentConfig(),
) -> pd.DataFrame:
    """
    Returns entries_df columns:
      doc_id, entry_num, slide_num, start_block, end_block, entry_text,
      seg_method, seg_confidence, flags_json, field_anchor_counts_json, error
    """
    if blocks_df is None or blocks_df.empty:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "entry_num",
                "slide_num",
                "start_block",
                "end_block",
                "entry_text",
                "seg_method",
                "seg_confidence",
                "flags_json",
                "field_anchor_counts_json",
                "error",
            ]
        )

    df = blocks_df.copy()

    if "doc_id" not in df.columns:
        raise KeyError("pptx_blocks.parquet must contain a 'doc_id' column.")

    slide_col = _pick_first_present(df, ["slide_num", "slide_index"])
    block_col = _pick_first_present(df, ["block_num", "shape_num", "idx", "index"])
    text_col = _pick_first_present(df, ["block_text", "text", "content_text"])
    ocr_col = _pick_first_present(df, ["ocr_text", "ocr_text_str", "image_ocr_text"])
    block_type_col = _pick_first_present(df, ["block_type", "type"])

    if slide_col is None:
        raise KeyError("pptx_blocks.parquet must contain slide_num (or slide_index).")

    # Normalize text fields
    df["_txt"] = ""
    if text_col:
        df["_txt"] = df["_txt"] + df[text_col].fillna("").astype(str)
    if cfg.include_ocr_text and ocr_col:
        ocr_piece = df[ocr_col].fillna("").astype(str)
        # Only add OCR when it has content, and keep it labeled so it's debuggable
        df["_txt"] = df["_txt"] + ocr_piece.apply(lambda s: f"\n[OCR]\n{s}" if s.strip() else "")

    # Optionally include notes if your ingest stores them as blocks (you can detect by block_type)
    if not cfg.include_notes and block_type_col:
        df = df[~df[block_type_col].astype(str).str.lower().str.contains("notes", na=False)].copy()

    # Deterministic ordering
    sort_cols = ["doc_id", slide_col]
    if block_col:
        sort_cols.append(block_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Attach doc metadata to flags if present
    doc_meta_cols = [c for c in ["num_slides", "num_blocks", "source_rel_path", "source_file"] if c in documents_df.columns]
    docs_meta = documents_df[["doc_id"] + doc_meta_cols].drop_duplicates("doc_id") if not documents_df.empty else pd.DataFrame({"doc_id": df["doc_id"].unique()})
    df = df.merge(docs_meta, on="doc_id", how="left")

    out_rows: List[Dict] = []

    for (doc_id, slide_num), g in df.groupby(["doc_id", slide_col], sort=False):
        # Determine span of blocks
        start_block = int(g[block_col].min()) if block_col and pd.notna(g[block_col]).any() else 0
        end_block = int(g[block_col].max()) if block_col and pd.notna(g[block_col]).any() else (len(g) - 1)

        entry_text = "\n".join([t for t in g["_txt"].tolist() if isinstance(t, str) and t.strip()])

        # Confidence heuristic: mostly about “is there meaningful text”
        n_chars = len(entry_text)
        seg_conf = 0.85 if n_chars >= cfg.min_entry_chars else (0.55 if n_chars > 0 else 0.25)

        flags = {
            "slide_num": int(slide_num) if pd.notna(slide_num) else None,
            "num_blocks_in_slide": int(len(g)),
            "include_ocr_text": bool(cfg.include_ocr_text),
            "include_notes": bool(cfg.include_notes),
            "doc_num_slides": g["num_slides"].iloc[0] if "num_slides" in g.columns else None,
        }

        out_rows.append(
            {
                "doc_id": doc_id,
                "slide_num": int(slide_num) if pd.notna(slide_num) else None,
                "start_block": start_block,
                "end_block": end_block,
                "entry_text": entry_text,
                "seg_method": "tier1_slide",
                "seg_confidence": float(seg_conf),
                "flags_json": json.dumps(flags, ensure_ascii=False),
                "field_anchor_counts_json": json.dumps(_field_anchor_counts_in_text(entry_text), ensure_ascii=False),
                "error": None,
            }
        )

    entries_df = pd.DataFrame(out_rows)

    # Dedupe safety net
    if cfg.dedupe and not entries_df.empty:
        entries_df["_h"] = (
            entries_df["doc_id"].astype(str)
            + "|"
            + entries_df["slide_num"].astype(str)
            + "|"
            + entries_df["entry_text"].fillna("").astype(str).apply(_stable_hash)
        ).apply(_stable_hash)

        entries_df = (
            entries_df.drop_duplicates(subset=["doc_id", "slide_num", "_h"], keep="first")
            .drop(columns=["_h"])
            .reset_index(drop=True)
        )

    # Deterministic entry_num per doc
    entries_df = entries_df.sort_values(["doc_id", "slide_num", "start_block"]).reset_index(drop=True)
    entries_df["entry_num"] = entries_df.groupby("doc_id").cumcount()

    return entries_df


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing pptx_documents.parquet and pptx_blocks.parquet")
    ap.add_argument("--out_dir", required=True, help="Where to write pptx_entries.parquet")
    ap.add_argument("--no_ocr", action="store_true", help="Exclude OCR text from entry_text")
    ap.add_argument("--no_notes", action="store_true", help="Exclude notes blocks (if present)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents_df = pd.read_parquet(in_dir / "pptx_documents.parquet")
    blocks_df = pd.read_parquet(in_dir / "pptx_blocks.parquet")

    cfg = PptxSegmentConfig(include_ocr_text=not args.no_ocr, include_notes=not args.no_notes)
    entries_df = segment_pptx_blocks_to_entries(documents_df, blocks_df, cfg)

    out_path = out_dir / "pptx_entries.parquet"
    entries_df.to_parquet(out_path, index=False)

    if not entries_df.empty:
        entries_df["entry_num_1based"] = entries_df["entry_num"] + 1
        # slide_num from ingest is typically 0-based; for display only:
        entries_df["slide_num_1based"] = entries_df["slide_num"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)

    print(f"Saved: {out_path}")
    if entries_df.empty:
        print("No PPTX entries produced (blocks_df empty).")
    else:
        print(
            entries_df[
                ["doc_id", "entry_num_1based", "slide_num_1based", "seg_method", "seg_confidence", "error"]
            ].head(25).to_string(index=False)
        )


if __name__ == "__main__":
    main_cli()