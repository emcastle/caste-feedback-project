"""
document_pdf.py

Purpose
-------
PDF-specific document ingestion utilities.

This handler extracts page-level text from PDFs, optionally runs OCR on embedded images,
and (optionally) parses the PDF into "entries" using marker-based segmentation.

Design
------
There are two layers:

1) Text extraction (template-agnostic):
   - Extract page text with PyMuPDF (fitz)
   - Detect pages containing images
   - Optionally OCR embedded images (pytesseract)

2) Entry parsing (template-dependent):
   - Segment content into entries using configurable markers (e.g., "Requestor’s Name:")
   - If no parser is configured or detection fails, fallback to a single record per document.

Outputs
-------
The primary entry point returns a records DataFrame with (at minimum):
- record_id
- source_file
- source_rel_path
- source_ext
- source_type
- text
- page_start, page_end (nullable)
- parser_used
- needs_ocr (nullable)
- error (nullable)
"""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


IMAGE_TEXT_FLAG = "[IMAGE_TEXT_HERE]"


@dataclass(frozen=True)
class PdfEntryMarkers:
    """
    Marker strings for parsing entries inside a PDF.
    These are template dependent.
    """
    entry_start: str = "Requestor’s Name:"
    response_header: str = "Response"
    incoming_hint: str = "Incoming"


@dataclass(frozen=True)
class PdfOcrConfig:
    """
    OCR configuration.
    """
    enable_ocr: bool = True
    sparse_text_threshold: int = 20  # if page text shorter than this and images exist, mark as needs OCR


@dataclass(frozen=True)
class PdfParseConfig:
    """
    Parsing configuration.

    parse_mode:
      - "entries": attempt marker-based entry segmentation
      - "pages": emit one record per page (merged text)
      - "document": emit one record for whole document (merged text)
    """
    parse_mode: str = "entries"
    markers: PdfEntryMarkers = PdfEntryMarkers()


def _new_record_id() -> str:
    return str(uuid.uuid4())


def extract_pdf_page_texts(pdf_path: Path) -> Dict[int, str]:
    import fitz  # PyMuPDF

    pdf_path = pdf_path.resolve()
    doc = fitz.open(str(pdf_path))
    try:
        out: Dict[int, str] = {}
        for page_num, page in enumerate(doc):
            out[page_num] = page.get_text("text") or ""
        return out
    finally:
        doc.close()


def pages_with_images(pdf_path: Path) -> Dict[int, bool]:
    import fitz  # PyMuPDF

    pdf_path = pdf_path.resolve()
    doc = fitz.open(str(pdf_path))
    try:
        out: Dict[int, bool] = {}
        for page_num, page in enumerate(doc):
            out[page_num] = len(page.get_images(full=True)) > 0
        return out
    finally:
        doc.close()


def extract_pdf_image_ocr_by_page(pdf_path: Path) -> Dict[int, str]:
    """
    OCR embedded images by page.
    Returns: {page_num: ocr_text}
    """
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract

    pdf_path = pdf_path.resolve()
    doc = fitz.open(str(pdf_path))
    try:
        ocr_texts: Dict[int, str] = {}
        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            if not images:
                continue

            parts: List[str] = []
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue

                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image).strip()
                if text:
                    parts.append(text)

            if parts:
                ocr_texts[page_num] = "\n".join(parts)

        return ocr_texts
    finally:
        doc.close()


def merge_page_text_and_ocr(
    page_texts: Dict[int, str],
    ocr_texts: Dict[int, str],
) -> Dict[int, str]:
    """
    Merge OCR text into page text (page-level).
    """
    merged: Dict[int, str] = {}
    for p in sorted(page_texts.keys()):
        t = (page_texts.get(p) or "").strip()
        o = (ocr_texts.get(p) or "").strip()
        if t and o:
            merged[p] = t + "\n\nOCR Extracted Text:\n" + o
        elif o and not t:
            merged[p] = "OCR Extracted Text:\n" + o
        else:
            merged[p] = t
    return merged


def parse_entries_from_page_texts(
    page_texts: Dict[int, str],
    markers: PdfEntryMarkers,
) -> pd.DataFrame:
    """
    Marker-based segmentation into entries.

    Returns a DataFrame:
      - text
      - start_page
      - end_page
    """
    entries: List[dict] = []
    current_lines: List[str] = []
    collecting = False
    start_page: Optional[int] = None

    page_nums = sorted(page_texts.keys())
    for page_num in page_nums:
        lines = [ln.strip() for ln in (page_texts.get(page_num) or "").split("\n")]

        for line in lines:
            if not line:
                continue

            if markers.entry_start in line:
                if current_lines:
                    entries.append(
                        {
                            "text": "\n".join(current_lines),
                            "start_page": start_page,
                            "end_page": page_num,
                        }
                    )
                current_lines = [line]
                start_page = page_num
                collecting = True
                continue

            if markers.response_header and markers.response_header in line:
                collecting = True
                current_lines.append("\n" + line)
                continue

            if collecting:
                current_lines.append(line)

    if current_lines:
        last_page = page_nums[-1] if page_nums else 0
        entries.append(
            {"text": "\n".join(current_lines), "start_page": start_page, "end_page": last_page}
        )

    return pd.DataFrame(entries)


def mark_entries_needing_ocr(
    df_entries: pd.DataFrame,
    page_texts: Dict[int, str],
    page_has_images: Dict[int, bool],
    markers: PdfEntryMarkers,
    ocr_cfg: PdfOcrConfig,
) -> pd.DataFrame:
    """
    Flag entries likely missing content due to image-based text.
    Uses two signals:
    - pages contain images AND page text is sparse
    - entry text contains incoming_hint AND spans image pages
    """
    if df_entries.empty:
        df_entries = df_entries.copy()
        df_entries["needs_ocr"] = False
        return df_entries

    needs_list: List[bool] = []
    updated_texts: List[str] = []

    for _, row in df_entries.iterrows():
        entry_text = row.get("text") or ""
        sp = int(row["start_page"]) if pd.notna(row.get("start_page")) else 0
        ep = int(row["end_page"]) if pd.notna(row.get("end_page")) else sp

        needs = False

        # Signal 1: sparse page text on an image page within entry range
        for p in range(sp, ep + 1):
            if not page_has_images.get(p, False):
                continue
            page_txt = (page_texts.get(p) or "").strip()
            if len(page_txt) < ocr_cfg.sparse_text_threshold:
                needs = True
                break

        # Signal 2: incoming hint + image pages
        if not needs and markers.incoming_hint and markers.incoming_hint in entry_text:
            for p in range(sp, ep + 1):
                if page_has_images.get(p, False):
                    needs = True
                    break

        needs_list.append(needs)

        t = entry_text
        if needs and IMAGE_TEXT_FLAG not in t:
            t = t + "\n" + IMAGE_TEXT_FLAG
        updated_texts.append(t)

    out = df_entries.copy()
    out["needs_ocr"] = needs_list
    out["text"] = updated_texts
    return out


def merge_entry_ocr_by_page_range(
    df_entries: pd.DataFrame,
    ocr_texts: Dict[int, str],
) -> pd.DataFrame:
    """
    If IMAGE_TEXT_FLAG is present, replace it with OCR text from pages in [start_page, end_page].
    """
    if df_entries.empty:
        return df_entries

    merged_texts: List[str] = []
    for _, row in df_entries.iterrows():
        t = row.get("text") or ""
        sp = int(row["start_page"]) if pd.notna(row.get("start_page")) else 0
        ep = int(row["end_page"]) if pd.notna(row.get("end_page")) else sp

        relevant: List[str] = []
        for p in range(sp, ep + 1):
            txt = (ocr_texts.get(p) or "").strip()
            if txt:
                relevant.append(txt)

        if IMAGE_TEXT_FLAG in t and relevant:
            t = t.replace(IMAGE_TEXT_FLAG, "OCR Extracted Text:\n" + "\n".join(relevant))

        merged_texts.append(t)

    out = df_entries.copy()
    out["text"] = merged_texts
    return out


def _records_from_entries(
    df_entries: pd.DataFrame,
    source_file: str,
    source_rel_path: str,
    parser_used: str,
) -> pd.DataFrame:
    """
    Convert entry rows to unified records schema.
    """
    if df_entries.empty:
        return pd.DataFrame(
            [
                {
                    "record_id": _new_record_id(),
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": ".pdf",
                    "source_type": "pdf_document",
                    "text": "",
                    "page_start": None,
                    "page_end": None,
                    "parser_used": parser_used,
                    "needs_ocr": None,
                    "error": "no_entries_parsed",
                }
            ]
        )

    records = []
    for _, r in df_entries.iterrows():
        records.append(
            {
                "record_id": _new_record_id(),
                "source_file": source_file,
                "source_rel_path": source_rel_path,
                "source_ext": ".pdf",
                "source_type": "pdf_entry",
                "text": r.get("text") or "",
                "page_start": int(r["start_page"]) if pd.notna(r.get("start_page")) else None,
                "page_end": int(r["end_page"]) if pd.notna(r.get("end_page")) else None,
                "parser_used": parser_used,
                "needs_ocr": bool(r.get("needs_ocr")) if "needs_ocr" in df_entries.columns else None,
                "error": None,
            }
        )
    return pd.DataFrame(records)


def _records_from_pages(
    merged_page_texts: Dict[int, str],
    source_file: str,
    source_rel_path: str,
    parser_used: str,
) -> pd.DataFrame:
    records = []
    for p in sorted(merged_page_texts.keys()):
        records.append(
            {
                "record_id": _new_record_id(),
                "source_file": source_file,
                "source_rel_path": source_rel_path,
                "source_ext": ".pdf",
                "source_type": "pdf_page",
                "text": merged_page_texts.get(p) or "",
                "page_start": p,
                "page_end": p,
                "parser_used": parser_used,
                "needs_ocr": None,
                "error": None,
            }
        )
    return pd.DataFrame(records)


def _records_from_document(
    merged_page_texts: Dict[int, str],
    source_file: str,
    source_rel_path: str,
    parser_used: str,
) -> pd.DataFrame:
    full_text = "\n\n".join([merged_page_texts[p] for p in sorted(merged_page_texts.keys())]).strip()
    return pd.DataFrame(
        [
            {
                "record_id": _new_record_id(),
                "source_file": source_file,
                "source_rel_path": source_rel_path,
                "source_ext": ".pdf",
                "source_type": "pdf_document",
                "text": full_text,
                "page_start": 0 if merged_page_texts else None,
                "page_end": (max(merged_page_texts.keys()) if merged_page_texts else None),
                "parser_used": parser_used,
                "needs_ocr": None,
                "error": None,
            }
        ]
    )


def pdf_to_records(
    pdf_path: Path,
    source_rel_path: str,
    parse_cfg: PdfParseConfig = PdfParseConfig(),
    ocr_cfg: PdfOcrConfig = PdfOcrConfig(),
) -> pd.DataFrame:
    """
    Public handler function called by the orchestrator.

    Parameters
    ----------
    pdf_path:
        Absolute path to the PDF.
    source_rel_path:
        Path relative to raw root (stored for traceability).
    parse_cfg:
        Controls parse mode and markers.
    ocr_cfg:
        Controls OCR behavior.

    Returns
    -------
    pd.DataFrame in unified records schema.
    """
    pdf_path = pdf_path.resolve()
    source_file = pdf_path.name

    try:
        page_texts = extract_pdf_page_texts(pdf_path)
        img_pages = pages_with_images(pdf_path)

        ocr_texts: Dict[int, str] = {}
        if ocr_cfg.enable_ocr:
            ocr_texts = extract_pdf_image_ocr_by_page(pdf_path)

        merged_page_texts = merge_page_text_and_ocr(page_texts, ocr_texts)

        mode = (parse_cfg.parse_mode or "entries").lower().strip()
        parser_used = f"pdf:{mode}"

        if mode == "pages":
            return _records_from_pages(merged_page_texts, source_file, source_rel_path, parser_used)

        if mode == "document":
            return _records_from_document(merged_page_texts, source_file, source_rel_path, parser_used)

        # mode == "entries"
        df_entries = parse_entries_from_page_texts(page_texts, parse_cfg.markers)
        df_entries = mark_entries_needing_ocr(df_entries, page_texts, img_pages, parse_cfg.markers, ocr_cfg)

        if ocr_cfg.enable_ocr and ocr_texts:
            df_entries = merge_entry_ocr_by_page_range(df_entries, ocr_texts)

        return _records_from_entries(df_entries, source_file, source_rel_path, parser_used + ":markers")

    except Exception as e:
        return pd.DataFrame(
            [
                {
                    "record_id": _new_record_id(),
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": ".pdf",
                    "source_type": "pdf_document",
                    "text": "",
                    "page_start": None,
                    "page_end": None,
                    "parser_used": "pdf:error",
                    "needs_ocr": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )