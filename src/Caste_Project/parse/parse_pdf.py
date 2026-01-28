from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class PdfEntryValidationConfig:
    # content checks
    min_chars_primary: int = 200
    min_words_primary: int = 30

    # header-heavy heuristic
    header_heavy_ratio: float = 0.45  # if > this fraction of lines are "header-like", flag it
    max_signature_tail_lines: int = 12  # used for optional signature-dominance heuristic

    # OCR/image checks
    low_ocr_yield_chars: int = 50
    low_ocr_yield_image_pages: int = 1

    # rollup decision
    needs_vision_if_missing_and_has_images: bool = True


HEADER_LINE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*From\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*To\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*Subject\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*Date\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*Date\s+of\s+Request\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*Incoming\b", re.IGNORECASE),
    re.compile(r"^\s*Response\b", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"\bCQAS[-–—]?\s*\d{3,6}\b", re.IGNORECASE),
]

SIGNATURE_LINE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*Sincerely\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Regards\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Respectfully\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Very Respectfully\s*,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Thank you\s*,?\s*$", re.IGNORECASE),
]


def _safe_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _normalize_text(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\u00a0", " ").strip()


def _line_stats(text: str, cfg: PdfEntryValidationConfig) -> Dict[str, float]:
    """
    Compute simple heuristics about the entry_text.
    """
    txt = _normalize_text(text)
    if not txt:
        return {
            "num_lines": 0,
            "num_header_like_lines": 0,
            "header_like_ratio": 0.0,
            "signature_tail_hits": 0,
        }

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return {
            "num_lines": 0,
            "num_header_like_lines": 0,
            "header_like_ratio": 0.0,
            "signature_tail_hits": 0,
        }

    header_like = 0
    for ln in lines:
        if any(p.search(ln) for p in HEADER_LINE_PATTERNS):
            header_like += 1

    # signature-tail heuristic: count signature markers in the last N lines
    tail = lines[-min(len(lines), cfg.max_signature_tail_lines) :]
    sig_hits = sum(1 for ln in tail if any(p.search(ln) for p in SIGNATURE_LINE_PATTERNS))

    return {
        "num_lines": float(len(lines)),
        "num_header_like_lines": float(header_like),
        "header_like_ratio": float(header_like) / max(1.0, float(len(lines))),
        "signature_tail_hits": float(sig_hits),
    }


def _aggregate_pages_for_entry(pages_df: pd.DataFrame, doc_id: str, start_page: int, end_page: int) -> Dict[str, int]:
    """
    Aggregate extraction signals across all pages in an entry span.
    Assumes start_page/end_page are 0-based and inclusive.
    """
    p = pages_df[
        (pages_df["doc_id"] == doc_id)
        & (pages_df["page_num"] >= start_page)
        & (pages_df["page_num"] <= end_page)
    ].copy()

    if p.empty:
        return {
            "num_pages_in_entry": 0,
            "num_image_pages_in_entry": 0,
            "ocr_text_total": 0,
            "page_text_total": 0,
            "has_images_in_entry": 0,
            "ocr_present_in_entry": 0,
        }

    p["has_images"] = p["has_images"].apply(_safe_bool)
    p["ocr_text_len"] = p["ocr_text_len"].apply(lambda x: _safe_int(x, 0))
    p["page_text_len"] = p["page_text_len"].apply(lambda x: _safe_int(x, 0))

    num_pages = int(p["page_num"].nunique())
    num_image_pages = int(p["has_images"].sum())
    ocr_total = int(p["ocr_text_len"].sum())
    page_text_total = int(p["page_text_len"].sum())

    has_images = 1 if num_image_pages > 0 else 0
    ocr_present = 1 if ocr_total > 0 else 0

    return {
        "num_pages_in_entry": num_pages,
        "num_image_pages_in_entry": num_image_pages,
        "ocr_text_total": ocr_total,
        "page_text_total": page_text_total,
        "has_images_in_entry": has_images,
        "ocr_present_in_entry": ocr_present,
    }


def validate_pdf_entries(
    entries_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    cfg: PdfEntryValidationConfig = PdfEntryValidationConfig(),
) -> pd.DataFrame:
    """
    Returns one row per entry with validation flags + summary metrics.
    """
    required_entry_cols = {"doc_id", "entry_num", "start_page", "end_page", "entry_text"}
    missing = required_entry_cols - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries_df missing required columns: {sorted(missing)}")

    required_pages_cols = {"doc_id", "page_num", "page_text_len", "ocr_text_len", "has_images"}
    missing_p = required_pages_cols - set(pages_df.columns)
    if missing_p:
        raise ValueError(f"pages_df missing required columns: {sorted(missing_p)}")

    out_rows: List[Dict] = []

    for _, r in entries_df.iterrows():
        doc_id = str(r["doc_id"])
        entry_num = _safe_int(r["entry_num"], 0)

        start_page = r["start_page"]
        end_page = r["end_page"]
        start_page_i = _safe_int(start_page, 0)
        end_page_i = _safe_int(end_page, start_page_i)

        entry_text = _normalize_text(r.get("entry_text"))

        # Content metrics
        content_len = len(entry_text)
        word_count = len([w for w in re.split(r"\s+", entry_text) if w])

        stats = _line_stats(entry_text, cfg)

        # Page/image/OCR metrics
        page_aggs = _aggregate_pages_for_entry(pages_df, doc_id, start_page_i, end_page_i)

        # Flags
        missing_content_primary = content_len == 0
        too_short_content_primary = (content_len > 0) and (
            content_len < cfg.min_chars_primary or word_count < cfg.min_words_primary
        )
        content_is_mostly_headers = (stats["num_lines"] > 0) and (stats["header_like_ratio"] >= cfg.header_heavy_ratio)

        has_images_in_entry = bool(page_aggs["has_images_in_entry"])
        ocr_present_in_entry = bool(page_aggs["ocr_present_in_entry"])

        low_ocr_yield = (
            has_images_in_entry
            and page_aggs["ocr_text_total"] < cfg.low_ocr_yield_chars
            and page_aggs["num_image_pages_in_entry"] >= cfg.low_ocr_yield_image_pages
        )

        likely_nontext_graphics = has_images_in_entry and low_ocr_yield

        needs_vision_review = False
        if cfg.needs_vision_if_missing_and_has_images and missing_content_primary and has_images_in_entry:
            needs_vision_review = True
        if low_ocr_yield and page_aggs["num_image_pages_in_entry"] >= 2:
            needs_vision_review = True
        if too_short_content_primary and has_images_in_entry and low_ocr_yield:
            needs_vision_review = True

        flags = {
            "missing_content_primary": bool(missing_content_primary),
            "too_short_content_primary": bool(too_short_content_primary),
            "content_is_mostly_headers": bool(content_is_mostly_headers),
            "has_images_in_entry": bool(has_images_in_entry),
            "ocr_present_in_entry": bool(ocr_present_in_entry),
            "low_ocr_yield": bool(low_ocr_yield),
            "likely_nontext_graphics": bool(likely_nontext_graphics),
            "needs_vision_review": bool(needs_vision_review),
        }

        # Optional numeric quality score (0..100) for sorting/debug
        score = 100
        if missing_content_primary:
            score -= 60
        if too_short_content_primary:
            score -= 25
        if content_is_mostly_headers:
            score -= 15
        if needs_vision_review:
            score -= 10
        score = max(0, min(100, score))

        out_rows.append(
            {
                "doc_id": doc_id,
                "entry_num": entry_num,
                "start_page": start_page if pd.notna(start_page) else None,
                "end_page": end_page if pd.notna(end_page) else None,
                "content_primary_len": int(content_len),
                "content_primary_words": int(word_count),
                "num_lines": int(stats["num_lines"]),
                "num_header_like_lines": int(stats["num_header_like_lines"]),
                "header_like_ratio": float(stats["header_like_ratio"]),
                "signature_tail_hits": int(stats["signature_tail_hits"]),
                **page_aggs,
                "needs_vision_review": bool(needs_vision_review),
                "quality_score": int(score),
                "flags_json": json.dumps(flags, ensure_ascii=False),
                "error": None,
            }
        )

    out = pd.DataFrame(out_rows)

    # deterministic ordering + ensure entry_num contiguous per doc
    if not out.empty:
        out = out.sort_values(["doc_id", "start_page", "entry_num"]).reset_index(drop=True)
        out["entry_num"] = out.groupby("doc_id").cumcount()

    return out


def main_cli() -> None:
    """
    Usage:
      conda run -n feedback python -m Caste_Project.validate.validate_pdf_entries ^
        --extract_dir data\\_test_output ^
        --seg_dir data\\_seg_output ^
        --out_dir data\\_val_output
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_dir", required=True, help="Folder containing pdf_pages.parquet (extraction output)")
    ap.add_argument("--seg_dir", required=True, help="Folder containing pdf_entries.parquet (segmentation output)")
    ap.add_argument("--out_dir", required=True, help="Where to write pdf_entry_validation.parquet")
    args = ap.parse_args()

    extract_dir = Path(args.extract_dir)
    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_path = extract_dir / "pdf_pages.parquet"
    entries_path = seg_dir / "pdf_entries.parquet"

    if not pages_path.exists():
        raise FileNotFoundError(f"Missing: {pages_path}")
    if not entries_path.exists():
        raise FileNotFoundError(f"Missing: {entries_path}")

    pages_df = pd.read_parquet(pages_path)
    entries_df = pd.read_parquet(entries_path)

    cfg = PdfEntryValidationConfig()

    val_df = validate_pdf_entries(entries_df=entries_df, pages_df=pages_df, cfg=cfg)
    out_path = out_dir / "pdf_entry_validation.parquet"
    val_df.to_parquet(out_path, index=False)

    # human-friendly print (1-based pages + entry nums)
    show = val_df.copy()
    show["entry_num_1based"] = show["entry_num"] + 1
    show["start_page_1based"] = show["start_page"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)
    show["end_page_1based"] = show["end_page"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)

    print(f"Saved: {out_path}")
    cols = [
        "doc_id",
        "entry_num_1based",
        "start_page_1based",
        "end_page_1based",
        "content_primary_len",
        "num_image_pages_in_entry",
        "ocr_text_total",
        "needs_vision_review",
        "quality_score",
        "error",
    ]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main_cli()