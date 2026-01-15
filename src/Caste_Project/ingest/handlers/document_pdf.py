"""
document_pdf.py

Purpose
-------
PDF ingestion handler focused on extraction (not template parsing).

This module:
1) Reads PDF page text using PyMuPDF (fitz)
2) Detects embedded images per page
3) Optionally OCRs embedded images per page (pytesseract)
4) Produces two relational outputs:
   - documents_df: 1 row per PDF (identity + metadata + doc-level errors)
   - pages_df:     1 row per page (page order + extracted text + OCR text)

Key Design Decisions
--------------------
- Extraction-only: no marker-based entry segmentation here.
- Preserve ordering: page_num + doc_id allow later segmentation/parsing to reconstruct
  full-document text deterministically.
- Stable identity: doc_id is computed as sha256 of PDF bytes so it persists across runs
  even if the file is renamed or moved.

Downstream Usage
----------------
To reconstruct full document text later:
- group pages_df by doc_id
- sort by page_num
- join merged_text with page delimiters

Example downstream reconstruction (conceptual):
full_text = "\n\n".join(f"=== PAGE {p} ===\n{t}" for p, t in ordered_pages)

Outputs (schemas)
-----------------
documents_df columns:
- doc_id
- source_file
- source_rel_path
- source_ext
- source_type
- num_pages
- num_image_pages
- extractor_used
- error

pages_df columns:
- doc_id
- page_num
- has_images
- page_text
- ocr_text
- merged_text
- page_text_len
- ocr_text_len
- error
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class PdfOcrConfig:
    """
    OCR configuration for embedded images inside PDFs.
    """
    enable_ocr: bool = True


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 for a file. Used as stable doc_id.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extract_page_texts(pdf_path: Path) -> Dict[int, str]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    try:
        out: Dict[int, str] = {}
        for page_num, page in enumerate(doc):
            out[page_num] = page.get_text("text") or ""
        return out
    finally:
        doc.close()


def _pages_with_images(pdf_path: Path) -> Dict[int, bool]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    try:
        out: Dict[int, bool] = {}
        for page_num, page in enumerate(doc):
            out[page_num] = len(page.get_images(full=True)) > 0
        return out
    finally:
        doc.close()


def _ocr_images_by_page(pdf_path: Path) -> Dict[int, str]:
    """
    OCR embedded images in the PDF, grouped by page number.
    """
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract

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

                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    text = pytesseract.image_to_string(image).strip()
                    if text:
                        parts.append(text)
                except Exception:
                    # Skip this image only; page-level error handled separately if desired
                    continue

            if parts:
                ocr_texts[page_num] = "\n".join(parts)

        return ocr_texts
    finally:
        doc.close()


def _merge_page_text_and_ocr(page_text: str, ocr_text: str) -> str:
    t = (page_text or "").strip()
    o = (ocr_text or "").strip()

    if t and o:
        return t + "\n\nOCR Extracted Text:\n" + o
    if o and not t:
        return "OCR Extracted Text:\n" + o
    return t


def pdf_extract_to_relational(
    pdf_path: Path,
    source_rel_path: str,
    ocr_cfg: PdfOcrConfig = PdfOcrConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract a PDF into two relational tables: documents_df and pages_df.

    Parameters
    ----------
    pdf_path:
        Absolute path to the PDF.
    source_rel_path:
        Path relative to your raw root (for traceability).
    ocr_cfg:
        OCR configuration (enable/disable).

    Returns
    -------
    (documents_df, pages_df)
    """
    pdf_path = pdf_path.resolve()
    doc_id = sha256_file(pdf_path)

    source_file = pdf_path.name
    source_ext = ".pdf"
    source_type = "pdf_document"
    extractor_used = "pymupdf" + ("+tesseract" if ocr_cfg.enable_ocr else "")

    try:
        page_texts = _extract_page_texts(pdf_path)
        has_images = _pages_with_images(pdf_path)

        ocr_texts: Dict[int, str] = {}
        if ocr_cfg.enable_ocr:
            ocr_texts = _ocr_images_by_page(pdf_path)

        pages_rows: List[dict] = []
        for page_num in sorted(page_texts.keys()):
            pt = page_texts.get(page_num) or ""
            ot = ocr_texts.get(page_num) or ""
            merged = _merge_page_text_and_ocr(pt, ot)

            pages_rows.append(
                {
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "has_images": bool(has_images.get(page_num, False)),
                    "page_text": pt,
                    "ocr_text": ot,
                    "merged_text": merged,
                    "page_text_len": len(pt.strip()),
                    "ocr_text_len": len(ot.strip()),
                    "error": None,
                }
            )

        pages_df = pd.DataFrame(pages_rows)

        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_pages": int(len(page_texts)),
                    "num_image_pages": int(sum(1 for v in has_images.values() if v)),
                    "extractor_used": extractor_used,
                    "error": None,
                }
            ]
        )

        return documents_df, pages_df

    except Exception as e:
        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_pages": None,
                    "num_image_pages": None,
                    "extractor_used": extractor_used,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        pages_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "page_num": None,
                    "has_images": None,
                    "page_text": "",
                    "ocr_text": "",
                    "merged_text": "",
                    "page_text_len": 0,
                    "ocr_text_len": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        return documents_df, pages_df


def build_full_text_from_pages(pages_df: pd.DataFrame, include_page_delimiters: bool = True) -> pd.DataFrame:
    """
    Convenience helper: derive a doc-level full_text from pages_df.

    Returns a DataFrame with:
    - doc_id
    - full_text

    Notes
    -----
    This is deterministic and rebuildable; do not treat it as canonical storage
    unless you explicitly want a cached copy.
    """
    if pages_df.empty:
        return pd.DataFrame(columns=["doc_id", "full_text"])

    required = {"doc_id", "page_num", "merged_text"}
    missing = required - set(pages_df.columns)
    if missing:
        raise ValueError(f"pages_df missing required columns: {sorted(missing)}")

    rows = []
    for doc_id, grp in pages_df.groupby("doc_id", sort=False):
        g = grp.sort_values("page_num")
        if include_page_delimiters:
            parts = [f"=== PAGE {int(r.page_num)} ===\n{r.merged_text}".strip() for r in g.itertuples(index=False)]
        else:
            parts = [str(r.merged_text).strip() for r in g.itertuples(index=False)]
        full_text = "\n\n".join([p for p in parts if p])
        rows.append({"doc_id": doc_id, "full_text": full_text})

    return pd.DataFrame(rows)