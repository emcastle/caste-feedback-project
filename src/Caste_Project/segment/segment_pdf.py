"""
Segmentation handling for pdf files

Logic Flow:
1. Tier 1 = heuristic segmenter
- takes pdf_pages.parquet output from extraction (page order preserved)
- flattens into ordered lines (including OCR text)
- finds repeatable boundary anchors 
- segments into feedback entries (may be one or more per document)
- computes the confience & flags the failure conditions


2. Tier 2 = HF Model 
Handles segmentation when tier one fails to find high confidence anchors 
- 


# Testing command example:
conda run -n feedback python -m Caste_Project.segment.segment_pdf --in_dir data\_test_output --out_dir data\_seg_output

"""


#################################
# Imports
#################################
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


#################################
# Class
#################################
@dataclass(frozen=True)
class PdfSegmentConfig:
    """
    Tier-1 PDF segmentation config.

    Segmentation goal:
      - Decide how many "entries" exist in a PDF
      - Produce entry-level text spans (not field parsing)

    Inputs:
      - pages_df from extraction (doc_id, page_num, page_text, ocr_text, ...)

    Outputs:
      - entries_df: one row per segmented entry
    """

    # Variables used to determine a failed segmentation
    max_pages_tier1: int = 30
    min_doc_chars_for_anchor_failure: int = 10_000
    max_orphan_head_ratio: float = 0.35
    min_boundary_repeat: int = 2
    min_segment_chars: int = 400
    max_segments_before_suspect: int = 80
    min_confidence: float = 0.60

    include_ocr_lines: bool = True

    # Tier-2 (local HF) routing
    enable_local_hf: bool = False
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

"""
# Repeatable, boundary-like anchors (good candidates for splitting multiple entries)
BOUNDARY_ANCHORS: Dict[str, re.Pattern] = {
    # Standardized CQAS-ish forms
    "requestor_name": re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*", re.IGNORECASE),
    "date_of_request": re.compile(r"^\s*Date\s+of\s+Request\s*:\s*", re.IGNORECASE),
    "incoming": re.compile(r"^\s*Incoming\b", re.IGNORECASE),
    "response": re.compile(r"^\s*Response\b", re.IGNORECASE),

    # Email-like headers
    "from": re.compile(r"^\s*From\s*:\s*", re.IGNORECASE),
    "to": re.compile(r"^\s*To\s*:\s*", re.IGNORECASE),
    "subject": re.compile(r"^\s*Subject\s*:\s*", re.IGNORECASE),
    "original_message": re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),

    # CQAS ID (often appears near top of an entry)
    "cqas_id": re.compile(r"\bCQAS[-–—]?\s*\d{3,6}\b", re.IGNORECASE),
}

"""

# Entry boundary anchors: ONLY things that can start a new entry
ENTRY_ANCHORS: Dict[str, re.Pattern] = {
    # CQAS / form entry starts
    "requestor_name": re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*", re.IGNORECASE),
    "cqas_id": re.compile(r"\bCQAS[-–—]?\s*\d{3,6}\b", re.IGNORECASE),

    # Email-like entry starts (use as start markers, not fields)
    "original_message": re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),
    "from": re.compile(r"^\s*From\s*:\s*", re.IGNORECASE),
    "to": re.compile(r"^\s*To\s*:\s*", re.IGNORECASE),
    "subject": re.compile(r"^\s*Subject\s*:\s*", re.IGNORECASE),
}

# Field/section anchors: DO NOT split on these; record them for validation later
FIELD_ANCHORS: Dict[str, re.Pattern] = {
    "date_of_request": re.compile(r"^\s*Date\s+of\s+Request\s*:\s*", re.IGNORECASE),
    "incoming": re.compile(r"^\s*Incoming\b", re.IGNORECASE),
    "response": re.compile(r"^\s*Response\b", re.IGNORECASE),

    # closings/signatures (examples)
    "sincerely": re.compile(r"^\s*Sincerely\s*,?\s*$", re.IGNORECASE),
    "regards": re.compile(r"^\s*Regards\s*,?\s*$", re.IGNORECASE),
    "respectfully": re.compile(r"^\s*Respectfully\s*,?\s*$", re.IGNORECASE),
}


def _field_anchor_counts_in_text(text: str) -> Dict[str, int]:
    counts = {}
    for name, pat in FIELD_ANCHORS.items():
        counts[name] = len(pat.findall(text))
    return counts

# Single-occurrence anchors that are NOT boundaries by default (section markers)
NON_BOUNDARY_SECTION_ANCHORS: List[re.Pattern] = [
    re.compile(r"^\s*Sincerely\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Regards\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Respectfully\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Kindly\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Very Respectfully\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Signed\s*,?\s*$", re.IGNORECASE),
]

#################################
# Tier 1 functions 
#################################

def _normalize_line(s: str) -> str:
    s = s.replace("\u00a0", " ")  # non-breaking space
    return s.strip()


def _iter_doc_lines(pages_df: pd.DataFrame, cfg: PdfSegmentConfig) -> pd.DataFrame:
    """
    Flatten pages into a deterministic ordered table of lines.

    Returns columns:
      doc_id, pos, page_num, source, line_text
    """
    rows: List[Dict] = []
    for _, r in pages_df.sort_values(["doc_id", "page_num"]).iterrows():
        doc_id = r["doc_id"]
        page_num = int(r["page_num"])
        page_text = r.get("page_text") or ""
        ocr_text = r.get("ocr_text") or ""

        for src, text in (("page_text", page_text), ("ocr_text", ocr_text)):
            if src == "ocr_text" and not cfg.include_ocr_lines:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            for line in text.splitlines():
                ln = _normalize_line(line)
                if ln:
                    rows.append(
                        {
                            "doc_id": doc_id,
                            "page_num": page_num,
                            "source": src,
                            "line_text": ln,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["doc_id", "pos", "page_num", "source", "line_text"])

    out = pd.DataFrame(rows)
    out.insert(1, "pos", range(len(out)))  # stable global position
    return out

def _pick_entry_strategy(anchor_counts: Dict[str, int], cfg: PdfSegmentConfig) -> Tuple[str, List[str]]:
    """
    Choose ONE entry segmentation strategy per document.
    Returns: (strategy_name, entry_anchor_keys_used)
    """
    # counts for CQAS/form style
    cqas_hits = {
        "requestor_name": anchor_counts.get("requestor_name", 0),
        "cqas_id": anchor_counts.get("cqas_id", 0),
    }

    # counts for email style
    email_hits = {
        "original_message": anchor_counts.get("original_message", 0),
        "from": anchor_counts.get("from", 0),
        "to": anchor_counts.get("to", 0),
        "subject": anchor_counts.get("subject", 0),
    }

    # Heuristic: CQAS form wins if it repeats
    if cqas_hits["requestor_name"] >= cfg.min_boundary_repeat:
        return ("cqas_requestor_name", ["requestor_name"])
    if cqas_hits["cqas_id"] >= cfg.min_boundary_repeat:
        return ("cqas_id", ["cqas_id"])

    # Email heuristic: require some repeatable “start pattern”
    # Prefer "Original Message" if present/repeatable, else From:
    if email_hits["original_message"] >= cfg.min_boundary_repeat:
        return ("email_original_message", ["original_message"])
    if email_hits["from"] >= cfg.min_boundary_repeat:
        return ("email_from", ["from"])

    # Otherwise: no safe multi-entry boundary
    return ("fallback_single", [])

def _count_anchor_hits(lines_df: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in ENTRY_ANCHORS}
    for txt in lines_df["line_text"].tolist():
        for name, pat in ENTRY_ANCHORS.items():
            if pat.search(txt):
                counts[name] += 1
    return counts


def _pick_repeatable_boundary_anchors(anchor_counts: Dict[str, int], cfg: PdfSegmentConfig) -> List[str]:
    """
    Choose anchors that are actually repeatable (likely indicating multiple entries).
    """
    repeatable = [k for k, v in anchor_counts.items() if v >= cfg.min_boundary_repeat]

    # If CQAS-ID is repeatable but the document also has strong email headers repeatable,
    # treat that as "conflict" later; keep both for now.
    return repeatable

"""
def _find_boundary_positions(lines_df: pd.DataFrame, repeatable_keys: List[str]) -> List[int]:
    
    # Return 'pos' indices where a repeatable boundary anchor hits.
    
    if not repeatable_keys:
        return []

    pats = [BOUNDARY_ANCHORS[k] for k in repeatable_keys]
    boundary_pos: List[int] = []

    for _, r in lines_df.iterrows():
        txt = r["line_text"]
        # Avoid splitting on closings like "Sincerely,"
        if any(p.search(txt) for p in NON_BOUNDARY_SECTION_ANCHORS):
            continue
        if any(p.search(txt) for p in pats):
            boundary_pos.append(int(r["pos"]))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in boundary_pos:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out
"""

def _segment_by_boundaries(
    lines_df: pd.DataFrame,
    boundary_pos: List[int],
    cfg: PdfSegmentConfig,
) -> List[Tuple[int, int]]:
    
    # Return list of (start_pos, end_pos_inclusive) segments.
    
    if lines_df.empty:
        return []

    last_pos = int(lines_df["pos"].max())

    # Always start at 0
    # starts = [0] + [p for p in boundary_pos if p > 0]

    if not boundary_pos:
        return[(0, last_pos)]
    
    starts = boundary_pos[:] # starts are anchor hits

    starts[0] = 0

    # Remove starts that are too close together (noise)
    cleaned = []
    prev = None
    for s in starts:
        if prev is None or (s - prev) >= 3:
            cleaned.append(s)
            prev = s

    # Build segments
    segs: List[Tuple[int, int]] = []
    for i, s in enumerate(cleaned):
        e = (cleaned[i + 1] - 1) if i + 1 < len(cleaned) else last_pos
        if e >= s:
            segs.append((s, e))

    # Merge segments that are too tiny
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
    counts = {}
    for name, pat in FIELD_ANCHORS.items():
        counts[name] = len(pat.findall(text))
    return counts

def _find_entry_start_positions(lines_df: pd.DataFrame, entry_anchor_keys: List[str], min_gap_lines: int = 8) -> List[int]:
    """
    Return pos indices where chosen entry start anchors occur.
    De-dupe hits that are too close together (same header repeated, etc.).
    """
    if not entry_anchor_keys:
        return []

    pats = [ENTRY_ANCHORS[k] for k in entry_anchor_keys]
    hits: List[int] = []

    for _, r in lines_df.iterrows():
        txt = r["line_text"]
        if any(p.search(txt) for p in pats):
            hits.append(int(r["pos"]))

    # de-dupe by proximity window (prevents multiple starts caused by repeated headers)
    hits = sorted(set(hits))
    cleaned: List[int] = []
    for p in hits:
        if not cleaned or (p - cleaned[-1]) >= min_gap_lines:
            cleaned.append(p)

    return cleaned

def _segment_confidence(
    lines_df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    repeatable_keys: List[str],
) -> float:
    """
    Simple confidence heuristic:
      - reward repeatable boundary anchors
      - penalize too many segments
      - penalize extreme orphan head
    """
    if lines_df.empty:
        return 0.0

    doc_text = "\n".join(lines_df["line_text"].tolist())
    doc_chars = max(1, len(doc_text))

    # Orphan head = text before first non-zero boundary
    first_start = segments[0][0] if segments else 0
    orphan_head_chars = len("\n".join(lines_df.loc[lines_df["pos"] < first_start, "line_text"].tolist()))
    orphan_ratio = orphan_head_chars / doc_chars

    score = 0.50

    if repeatable_keys:
        score += 0.25

    # More segments => more risk
    nseg = len(segments)
    if nseg <= 1:
        score += 0.10
    elif nseg <= 10:
        score += 0.05
    elif nseg > 50:
        score -= 0.15

    # Orphan penalty
    if orphan_ratio > 0.35:
        score -= 0.20
    elif orphan_ratio > 0.15:
        score -= 0.10
    else:
        score += 0.05

    return max(0.0, min(1.0, score))


def _detect_conflicted_anchors(anchor_counts: Dict[str, int], cfg: PdfSegmentConfig) -> bool:
    """
    "Conflicted" if multiple boundary families are repeatable.
    """
    repeatable = {k for k, v in anchor_counts.items() if v >= cfg.min_boundary_repeat}

    has_cqas_form = any(k in repeatable for k in ["requestor_name", "incoming", "response", "cqas_id"])
    has_email = any(k in repeatable for k in ["from", "to", "subject", "original_message"])

    return bool(has_cqas_form and has_email)

#################################
# Tier 2 functions
#################################

def _should_escalate_to_hf(
    num_pages: int,
    doc_chars: int,
    repeatable_keys: List[str],
    orphan_head_ratio: float,
    conflicted: bool,
    confidence: float,
    cfg: PdfSegmentConfig,
) -> bool:
    if not cfg.enable_local_hf:
        return False

    if num_pages > cfg.max_pages_tier1:
        return True

    if (not repeatable_keys) and doc_chars >= cfg.min_doc_chars_for_anchor_failure:
        return True

    if orphan_head_ratio > cfg.max_orphan_head_ratio:
        return True

    if conflicted:
        return True

    if confidence < cfg.min_confidence:
        return True

    return False


def _hf_segment_stub(lines_df: pd.DataFrame, cfg: PdfSegmentConfig) -> List[Tuple[int, int]]:
    """
    Tier-2 stub. You can replace this with sentence-transformers embedding + change-point detection.

    For now: raise a clear error if enabled but not implemented.
    """
    raise NotImplementedError(
        "Local HF segmentation is enabled but not implemented yet. "
        "Plumb in sentence-transformers based boundary detection here."
    )

#################################
# Main Function
#################################

def segment_pdf_pages_to_entries(
    pages_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    cfg: PdfSegmentConfig = PdfSegmentConfig(),
) -> pd.DataFrame:
    """
    Segment extracted PDF pages into entry-level rows.

    Returns entries_df columns:
      doc_id, entry_num, start_page, end_page, start_pos, end_pos, entry_text,
      seg_method, seg_confidence, flags_json, field_anchor_counts_json, error
    """
    out_rows: List[Dict] = []

    for doc_id, doc_pages in pages_df.groupby("doc_id"):
        doc_meta = documents_df.loc[documents_df["doc_id"] == doc_id].head(1)
        num_pages = (
            int(doc_meta["num_pages"].iloc[0])
            if (not doc_meta.empty and pd.notna(doc_meta["num_pages"].iloc[0]))
            else int(doc_pages["page_num"].max()) + 1
        )

        lines_df = _iter_doc_lines(doc_pages, cfg)
        doc_text = "\n".join(lines_df["line_text"].tolist()) if not lines_df.empty else ""
        doc_chars = len(doc_text)

        # Count ENTRY anchors only (make sure _count_anchor_hits uses ENTRY_ANCHORS now)
        anchor_counts = _count_anchor_hits(lines_df) if not lines_df.empty else {k: 0 for k in ENTRY_ANCHORS}

        # Choose ONE strategy per doc
        strategy_name, entry_keys = _pick_entry_strategy(anchor_counts, cfg)

        # Find entry start positions using ONLY chosen entry anchors
        boundary_pos = _find_entry_start_positions(lines_df, entry_keys) if (not lines_df.empty and entry_keys) else []

        # If we have starts, segment; else fallback to single
        segments = _segment_by_boundaries(lines_df, boundary_pos, cfg) if not lines_df.empty else []

        # orphan head ratio (text before first start if first start isn't 0)
        if segments and segments[0][0] > 0 and not lines_df.empty:
            orphan_head_chars = len("\n".join(lines_df.loc[lines_df["pos"] < segments[0][0], "line_text"].tolist()))
            orphan_head_ratio = orphan_head_chars / max(1, doc_chars)
        else:
            orphan_head_ratio = 0.0

        # Confidence should be computed using entry_keys (not old repeatable_keys)
        confidence = _segment_confidence(lines_df, segments, entry_keys)

        flags = {
            "num_pages": num_pages,
            "doc_chars": doc_chars,
            "entry_strategy": strategy_name,
            "entry_anchors_used": entry_keys,
            "entry_anchor_counts": anchor_counts,
            "orphan_head_ratio": orphan_head_ratio,
        }

        try:
            # Tier-2 routing (optional). For now you can leave hf disabled.
            if _should_escalate_to_hf(
                num_pages=num_pages,
                doc_chars=doc_chars,
                repeatable_keys=entry_keys,          # reuse param name, but feed entry_keys
                orphan_head_ratio=orphan_head_ratio,
                conflicted=False,                   # conflicted logic removed by single-strategy pick
                confidence=confidence,
                cfg=cfg,
            ):
                segments = _hf_segment_stub(lines_df, cfg)
                seg_method = "tier2_local_hf"
            else:
                seg_method = f"tier1_{strategy_name}"

            # Fallback behavior
            if not segments and not lines_df.empty:
                segments = [(0, int(lines_df["pos"].max()))]
                seg_method = "tier1_single"

            if not segments and lines_df.empty:
                out_rows.append(
                    {
                        "doc_id": doc_id,
                        "entry_num": 0,
                        "start_page": None,
                        "end_page": None,
                        "start_pos": None,
                        "end_pos": None,
                        "entry_text": "",
                        "seg_method": "tier1_empty",
                        "seg_confidence": 0.0,
                        "flags_json": json.dumps(flags, ensure_ascii=False),
                        "field_anchor_counts_json": json.dumps({}, ensure_ascii=False),
                        "error": None,
                    }
                )
                continue

            for entry_num, (s, e) in enumerate(segments):
                seg_lines = lines_df.loc[lines_df["pos"].between(s, e)]
                entry_text = "\n".join(seg_lines["line_text"].tolist())

                start_page = int(seg_lines["page_num"].min()) if not seg_lines.empty else None
                end_page = int(seg_lines["page_num"].max()) if not seg_lines.empty else None

                field_counts = _field_anchor_counts_in_text(entry_text)

                out_rows.append(
                    {
                        "doc_id": doc_id,
                        "entry_num": entry_num,
                        "start_page": start_page,
                        "end_page": end_page,
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

        except Exception as ex:
            out_rows.append(
                {
                    "doc_id": doc_id,
                    "entry_num": 0,
                    "start_page": None,
                    "end_page": None,
                    "start_pos": None,
                    "end_pos": None,
                    "entry_text": "",
                    "seg_method": "error",
                    "seg_confidence": float(confidence),
                    "flags_json": json.dumps(flags, ensure_ascii=False),
                    "field_anchor_counts_json": json.dumps({}, ensure_ascii=False),
                    "error": f"{type(ex).__name__}: {ex}",
                }
            )

    entries_df = pd.DataFrame(out_rows)

    # Optional final dedupe safety net (recommended)
    if not entries_df.empty:
        entries_df["_entry_text_hash"] = entries_df["entry_text"].fillna("").apply(lambda s: hash(s))
        entries_df = entries_df.drop_duplicates(
            subset=["doc_id", "start_page", "end_page", "_entry_text_hash"],
            keep="first",
        ).drop(columns=["_entry_text_hash"])

    return entries_df

#################################
# Main CLI
#################################
def main_cli() -> None:
    """
    Example CLI usage:
      python -m Caste_Project.segment.segment_pdf --in_dir data/_test_output --out_dir data/_seg_out
    """
    import argparse
    import hashlib


    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing pdf_pages.parquet and pdf_documents.parquet")
    ap.add_argument("--out_dir", required=True, help="Where to write pdf_entries.parquet")
    ap.add_argument("--enable_hf", action="store_true", help="Enable Tier-2 local HF segmentation (not implemented)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_df = pd.read_parquet(in_dir / "pdf_pages.parquet")
    documents_df = pd.read_parquet(in_dir / "pdf_documents.parquet")

    cfg = PdfSegmentConfig(enable_local_hf=bool(args.enable_hf))

    entries_df = segment_pdf_pages_to_entries(pages_df, documents_df, cfg)
    # entries_df.to_parquet(out_dir / "pdf_entries.parquet", index=False)

    # Dedupe safety net (stable hash)
    if not entries_df.empty:
        entries_df["_entry_text_hash"] = entries_df["entry_text"].fillna("").apply(
            lambda s: hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
        )
        entries_df = (
            entries_df.drop_duplicates(
                subset=["doc_id", "start_page", "end_page", "_entry_text_hash"],
                keep="first",
            )
            .drop(columns=["_entry_text_hash"])
            .reset_index(drop=True)
        ) 

    # Sanity check 
    bad_docs = []
    for doc_id, g in entries_df.groupby("doc_id"):
        nums = sorted(g["entry_num"].dropna().astype(int).unique().tolist())
        if nums != list(range(len(nums))):
            bad_docs.append((doc_id, nums[:30], len(nums)))
    
    if bad_docs:
        print("WARNING: non-contiguous entry_num detected:")
        for doc_id, sample_nums, n in bad_docs[:10]:
            print(f" {doc_id}: n_unique={n}, nums_head={sample_nums}")

    # Deterministic ordering 
    if not entries_df.empty:
        entries_df = entries_df.sort_values(["doc_id", "start_page", "start_pos"]).reset_index(drop=True)
        entries_df["entry_num"] = entries_df.groupby("doc_id").cumcount()
        #entries_df = (
        #    entries_df.sort_values(["doc_id", "start_page", "start_pos"])
        #    .groupby("doc_id", group_keys=False)
        #    .apply(lambda g: g.assign(entry_num=range(len(g))))
        #    .reset_index(drop=True)
        #)

    # Add human-friendly 1-based display columns (otherwise entry = 0)
    entries_df["entry_num_1based"] = entries_df["entry_num"] + 1

    # start_page/end_page can be None, so handle safely
    entries_df["start_page_1based"] = entries_df["start_page"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)
    entries_df["end_page_1based"] = entries_df["end_page"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)

    # Save (cleaned)
    entries_df.to_parquet(out_dir / "pdf_entries.parquet", index=False)

    print(f"Saved: {out_dir / 'pdf_entries.parquet'}")
    print(
        entries_df[
            ["doc_id", "entry_num_1based", "start_page_1based", "end_page_1based", "seg_method", "seg_confidence", "error"]
        ].head(25).to_string(index=False)
    )


if __name__ == "__main__":
    main_cli()
