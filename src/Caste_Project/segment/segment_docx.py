from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


#################################
# Config
#################################
@dataclass(frozen=True)
class DocxSegmentConfig:
    # segmentation heuristics
    min_boundary_repeat: int = 2        # anchor must appear >= this many times in doc to be "repeatable"
    min_segment_chars: int = 300        # merge segments smaller than this into previous
    max_segments_before_suspect: int = 200
    min_confidence: float = 0.60

    # include table text as lines (if your extraction includes table blocks)
    include_tables: bool = True


#################################
# Anchors
#################################

# These are candidate entry START markers. We will pick ONE strategy per doc.
ENTRY_ANCHORS: Dict[str, re.Pattern] = {
    # CQAS / structured comment style
    "requestor_name": re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*", re.IGNORECASE),
    "date_of_request": re.compile(r"^\s*Date\s+of\s+Request\s*:\s*", re.IGNORECASE),
    "incoming": re.compile(r"^\s*Incoming\b", re.IGNORECASE),

    # Email-like style
    "from": re.compile(r"^\s*From\s*:\s*", re.IGNORECASE),
    "to": re.compile(r"^\s*To\s*:\s*", re.IGNORECASE),
    "subject": re.compile(r"^\s*Subject\s*:\s*", re.IGNORECASE),
    "original_message": re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),

    # ID style (not always)
    "cqas_id": re.compile(r"\bCQAS[-–—]?\s*\d{3,6}\b", re.IGNORECASE),
}

# These are NOT entry boundaries by default; they’re useful later as “field/section” markers.
FIELD_ANCHORS: Dict[str, re.Pattern] = {
    "response": re.compile(r"^\s*Response\b", re.IGNORECASE),
    "sincerely": re.compile(r"^\s*Sincerely\s*,?\s*$", re.IGNORECASE),
    "regards": re.compile(r"^\s*Regards\s*,?\s*$", re.IGNORECASE),
    "respectfully": re.compile(r"^\s*Respectfully\s*,?\s*$", re.IGNORECASE),
    "signed": re.compile(r"^\s*Signed\s*,?\s*$", re.IGNORECASE),
}


#################################
# Helpers
#################################
def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\u00a0", " ").strip()


def _iter_docx_lines(blocks_df: pd.DataFrame, cfg: DocxSegmentConfig) -> pd.DataFrame:
    """
    Flatten docx blocks into deterministic ordered "lines".
    Expected input columns (from your extraction):
      doc_id, block_num, block_type, block_text (or text)
    """
    # tolerate different extraction column names
    text_col = "block_text" if "block_text" in blocks_df.columns else ("text" if "text" in blocks_df.columns else None)
    if text_col is None:
        return pd.DataFrame(columns=["doc_id", "pos", "block_num", "block_type", "line_text"])

    keep_types = {"paragraph"}
    if cfg.include_tables:
        keep_types.add("table")  # only if your extraction emits table blocks w/ text

    rows: List[Dict] = []
    for _, r in blocks_df.sort_values(["doc_id", "block_num"]).iterrows():
        bt = str(r.get("block_type", "paragraph")).lower()
        if bt not in keep_types:
            continue

        doc_id = r["doc_id"]
        block_num = int(r["block_num"]) if pd.notna(r.get("block_num")) else None
        text = _normalize_text(r.get(text_col, "") or "")
        if not text:
            continue

        # Split into lines for anchor matching, but preserve order
        for line in text.splitlines():
            ln = _normalize_text(line)
            if ln:
                rows.append(
                    {
                        "doc_id": doc_id,
                        "block_num": block_num,
                        "block_type": bt,
                        "line_text": ln,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["doc_id", "pos", "block_num", "block_type", "line_text"])

    out = pd.DataFrame(rows)
    out.insert(1, "pos", range(len(out)))  # stable global position per doc after groupby; we’ll rebuild per doc anyway
    return out


def _count_entry_anchor_hits(lines: List[str]) -> Dict[str, int]:
    counts = {k: 0 for k in ENTRY_ANCHORS}
    for t in lines:
        for name, pat in ENTRY_ANCHORS.items():
            if pat.search(t):
                counts[name] += 1
    return counts


def _pick_entry_strategy(anchor_counts: Dict[str, int], cfg: DocxSegmentConfig) -> Tuple[str, List[str]]:
    """
    Pick ONE entry strategy per doc, to avoid mixed anchors creating duplicates.

    Priority:
      - email headers if they repeat
      - structured CQAS-ish if they repeat
      - cqas_id if it repeats
      - else none
    """
    repeatable = {k for k, v in anchor_counts.items() if v >= cfg.min_boundary_repeat}

    email_keys = [k for k in ["from", "to", "subject", "original_message"] if k in repeatable]
    cqas_keys = [k for k in ["requestor_name", "date_of_request", "incoming"] if k in repeatable]

    if email_keys:
        return "email_headers", email_keys
    if cqas_keys:
        return "cqas_headers", cqas_keys
    if "cqas_id" in repeatable:
        return "cqas_id", ["cqas_id"]
    return "none", []


def _find_entry_start_positions(lines_df: pd.DataFrame, entry_keys: List[str]) -> List[int]:
    if not entry_keys or lines_df.empty:
        return []

    pats = [ENTRY_ANCHORS[k] for k in entry_keys]
    starts: List[int] = []
    for _, r in lines_df.iterrows():
        txt = r["line_text"]
        if any(p.search(txt) for p in pats):
            starts.append(int(r["pos"]))

    # dedupe in order
    seen = set()
    out = []
    for p in starts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _segment_by_starts(lines_df: pd.DataFrame, starts: List[int], cfg: DocxSegmentConfig) -> List[Tuple[int, int]]:
    if lines_df.empty:
        return []
    last_pos = int(lines_df["pos"].max())

    # Always include 0 as start
    starts2 = [0] + [p for p in starts if p > 0]

    # Remove noisy near-duplicates (too close)
    cleaned = []
    prev = None
    for s in starts2:
        if prev is None or (s - prev) >= 2:
            cleaned.append(s)
            prev = s

    # Build segments
    segs: List[Tuple[int, int]] = []
    for i, s in enumerate(cleaned):
        e = (cleaned[i + 1] - 1) if i + 1 < len(cleaned) else last_pos
        if e >= s:
            segs.append((s, e))

    # Merge too-small segments into previous
    merged: List[Tuple[int, int]] = []
    for s, e in segs:
        seg_text = "\n".join(lines_df.loc[lines_df["pos"].between(s, e), "line_text"].tolist())
        if merged and len(seg_text) < cfg.min_segment_chars:
            ps, pe = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


def _field_anchor_counts_in_text(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, pat in FIELD_ANCHORS.items():
        counts[name] = len(pat.findall(text or ""))
    return counts


def _segment_confidence(n_segments: int, strategy_name: str, cfg: DocxSegmentConfig) -> float:
    score = 0.55
    if strategy_name != "none":
        score += 0.20
    if n_segments == 1:
        score += 0.10
    if n_segments > 50:
        score -= 0.15
    if n_segments > cfg.max_segments_before_suspect:
        score -= 0.20
    return max(0.0, min(1.0, score))


#################################
# Main segmenter
#################################
def segment_docx_blocks_to_entries(
    blocks_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    cfg: DocxSegmentConfig = DocxSegmentConfig(),
) -> pd.DataFrame:
    out_rows: List[Dict] = []

    for doc_id, doc_blocks in blocks_df.groupby("doc_id"):
        # rebuild deterministic lines per doc
        lines_df = _iter_docx_lines(doc_blocks, cfg)
        if not lines_df.empty:
            lines_df = lines_df.reset_index(drop=True)
            lines_df["pos"] = range(len(lines_df))  # per-doc stable positions

        doc_lines = lines_df["line_text"].tolist() if not lines_df.empty else []
        anchor_counts = _count_entry_anchor_hits(doc_lines)
        strategy_name, entry_keys = _pick_entry_strategy(anchor_counts, cfg)

        starts = _find_entry_start_positions(lines_df, entry_keys) if entry_keys else []
        segments = _segment_by_starts(lines_df, starts, cfg) if not lines_df.empty else []

        if not segments and not lines_df.empty:
            segments = [(0, int(lines_df["pos"].max()))]
            seg_method = "tier1_single"
        elif not segments and lines_df.empty:
            out_rows.append(
                {
                    "doc_id": doc_id,
                    "entry_num": 0,
                    "start_block": None,
                    "end_block": None,
                    "start_pos": None,
                    "end_pos": None,
                    "entry_text": "",
                    "seg_method": "tier1_empty",
                    "seg_confidence": 0.0,
                    "flags_json": json.dumps(
                        {"entry_strategy": "empty", "entry_anchors_used": [], "entry_anchor_counts": anchor_counts},
                        ensure_ascii=False,
                    ),
                    "field_anchor_counts_json": json.dumps({}, ensure_ascii=False),
                    "error": None,
                }
            )
            continue
        else:
            seg_method = f"tier1_{strategy_name}"

        confidence = _segment_confidence(len(segments), strategy_name, cfg)

        flags = {
            "entry_strategy": strategy_name,
            "entry_anchors_used": entry_keys,
            "entry_anchor_counts": anchor_counts,
        }

        for entry_num, (s, e) in enumerate(segments):
            seg_lines = lines_df.loc[lines_df["pos"].between(s, e)]
            entry_text = "\n".join(seg_lines["line_text"].tolist())

            start_block = int(seg_lines["block_num"].min()) if not seg_lines.empty and "block_num" in seg_lines.columns else None
            end_block = int(seg_lines["block_num"].max()) if not seg_lines.empty and "block_num" in seg_lines.columns else None

            field_counts = _field_anchor_counts_in_text(entry_text)

            out_rows.append(
                {
                    "doc_id": doc_id,
                    "entry_num": entry_num,
                    "start_block": start_block,
                    "end_block": end_block,
                    "start_pos": int(s),
                    "end_pos": int(e),
                    "entry_text": entry_text,
                    "seg_method": seg_method,
                    "seg_confidence": float(confidence),
                    "flags_json": json.dumps(flags, ensure_ascii=False),
                    "field_anchor_counts_json": json.dumps(field_counts, ensure_ascii=False),
                    "error": None,
                }
            )

    entries_df = pd.DataFrame(out_rows)

    # Dedupe safety net (stable hash) + deterministic renumber
    if not entries_df.empty:
        entries_df["_entry_text_hash"] = entries_df["entry_text"].fillna("").apply(
            lambda s: hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
        )
        entries_df = (
            entries_df.drop_duplicates(
                subset=["doc_id", "start_block", "end_block", "_entry_text_hash"],
                keep="first",
            )
            .drop(columns=["_entry_text_hash"])
            .sort_values(["doc_id", "start_block", "start_pos"])
            .reset_index(drop=True)
        )
        entries_df["entry_num"] = entries_df.groupby("doc_id").cumcount()

    return entries_df


#################################
# CLI
#################################
def main_cli() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing docx_blocks.parquet and docx_documents.parquet")
    ap.add_argument("--out_dir", required=True, help="Where to write docx_entries.parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks_df = pd.read_parquet(in_dir / "docx_blocks.parquet")
    documents_df = pd.read_parquet(in_dir / "docx_documents.parquet")

    cfg = DocxSegmentConfig()
    entries_df = segment_docx_blocks_to_entries(blocks_df, documents_df, cfg)

    # Add 1-based display columns (like pdf)
    if not entries_df.empty:
        entries_df["entry_num_1based"] = entries_df["entry_num"] + 1
        entries_df["start_block_1based"] = entries_df["start_block"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)
        entries_df["end_block_1based"] = entries_df["end_block"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)

    entries_df.to_parquet(out_dir / "docx_entries.parquet", index=False)

    print(f"Saved: {out_dir / 'docx_entries.parquet'}")
    show_cols = ["doc_id", "entry_num_1based", "start_block_1based", "end_block_1based", "seg_method", "seg_confidence", "error"]
    show_cols = [c for c in show_cols if c in entries_df.columns]
    print(entries_df[show_cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
