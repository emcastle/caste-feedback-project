from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


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

# Single-occurrence anchors that are NOT boundaries by default (section markers)
NON_BOUNDARY_SECTION_ANCHORS: List[re.Pattern] = [
    re.compile(r"^\s*Sincerely\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Regards\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Respectfully\s*,?\s*$", re.IGNORECASE),
]


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


def _count_anchor_hits(lines_df: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in BOUNDARY_ANCHORS}
    for txt in lines_df["line_text"].tolist():
        for name, pat in BOUNDARY_ANCHORS.items():
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


def _find_boundary_positions(lines_df: pd.DataFrame, repeatable_keys: List[str]) -> List[int]:
    """
    Return 'pos' indices where a repeatable boundary anchor hits.
    """
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


def _segment_by_boundaries(
    lines_df: pd.DataFrame,
    boundary_pos: List[int],
    cfg: PdfSegmentConfig,
) -> List[Tuple[int, int]]:
    """
    Return list of (start_pos, end_pos_inclusive) segments.
    """
    if lines_df.empty:
        return []

    last_pos = int(lines_df["pos"].max())

    # Always start at 0
    starts = [0] + [p for p in boundary_pos if p > 0]

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


def segment_pdf_pages_to_entries(
    pages_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    cfg: PdfSegmentConfig = PdfSegmentConfig(),
) -> pd.DataFrame:
    """
    Segment extracted PDF pages into entry-level rows.

    Returns entries_df columns:
      doc_id, entry_num, start_page, end_page, start_pos, end_pos, entry_text,
      seg_method, seg_confidence, flags_json, error
    """
    out_rows: List[Dict] = []

    for doc_id, doc_pages in pages_df.groupby("doc_id"):
        doc_meta = documents_df.loc[documents_df["doc_id"] == doc_id].head(1)
        num_pages = int(doc_meta["num_pages"].iloc[0]) if (not doc_meta.empty and pd.notna(doc_meta["num_pages"].iloc[0])) else int(doc_pages["page_num"].max()) + 1

        lines_df = _iter_doc_lines(doc_pages, cfg)
        doc_text = "\n".join(lines_df["line_text"].tolist()) if not lines_df.empty else ""
        doc_chars = len(doc_text)

        anchor_counts = _count_anchor_hits(lines_df) if not lines_df.empty else {k: 0 for k in BOUNDARY_ANCHORS}
        repeatable_keys = _pick_repeatable_boundary_anchors(anchor_counts, cfg)
        conflicted = _detect_conflicted_anchors(anchor_counts, cfg)

        boundary_pos = _find_boundary_positions(lines_df, repeatable_keys) if not lines_df.empty else []
        segments = _segment_by_boundaries(lines_df, boundary_pos, cfg) if not lines_df.empty else []

        # orphan head ratio (for escalation and flags)
        if segments and segments[0][0] > 0 and not lines_df.empty:
            orphan_head_chars = len("\n".join(lines_df.loc[lines_df["pos"] < segments[0][0], "line_text"].tolist()))
            orphan_head_ratio = orphan_head_chars / max(1, doc_chars)
        else:
            orphan_head_ratio = 0.0

        confidence = _segment_confidence(lines_df, segments, repeatable_keys)

        flags = {
            "num_pages": num_pages,
            "doc_chars": doc_chars,
            "repeatable_anchors": repeatable_keys,
            "anchor_counts": anchor_counts,
            "conflicted_anchors": conflicted,
            "orphan_head_ratio": orphan_head_ratio,
        }

        try:
            if _should_escalate_to_hf(
                num_pages=num_pages,
                doc_chars=doc_chars,
                repeatable_keys=repeatable_keys,
                orphan_head_ratio=orphan_head_ratio,
                conflicted=conflicted,
                confidence=confidence,
                cfg=cfg,
            ):
                segments = _hf_segment_stub(lines_df, cfg)
                seg_method = "tier2_local_hf"
            else:
                seg_method = "tier1_anchors"

            if not segments and not lines_df.empty:
                # fallback: single entry covering all
                segments = [(0, int(lines_df["pos"].max()))]
                seg_method = "tier1_single"

            if not segments and lines_df.empty:
                # no text at all
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
                        "error": None,
                    }
                )
                continue

            for entry_num, (s, e) in enumerate(segments):
                seg_lines = lines_df.loc[lines_df["pos"].between(s, e)]
                entry_text = "\n".join(seg_lines["line_text"].tolist())

                start_page = int(seg_lines["page_num"].min()) if not seg_lines.empty else None
                end_page = int(seg_lines["page_num"].max()) if not seg_lines.empty else None

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
                    "error": f"{type(ex).__name__}: {ex}",
                }
            )

    return pd.DataFrame(out_rows)


def main_cli() -> None:
    """
    Example CLI usage:
      python -m Caste_Project.segment.segment_pdf --in_dir data/_test_output --out_dir data/_seg_out
    """
    import argparse

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
    entries_df.to_parquet(out_dir / "pdf_entries.parquet", index=False)
    print(f"Saved: {out_dir / 'pdf_entries.parquet'}")
    print(entries_df[["doc_id", "entry_num", "start_page", "end_page", "seg_method", "seg_confidence", "error"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
