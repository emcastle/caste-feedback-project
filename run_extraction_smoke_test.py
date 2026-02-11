"""
Docstring for Feedback_Caste.run_extraction_smoke_test

Runs extraction for ALL files of a specified extension type under a test input folder,
and writes combined parquet outputs to an output folder.

Examples:
  conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .pdf
  conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .docx
  conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .csv
  conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .xlsx
  conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .json
  conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .pptx

Optional:
  --limit 3   # only process first 3 matching files
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

# Make repo_root/src importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from Caste_Project.ingest.discover import build_manifest

from Caste_Project.ingest.handlers.document_pdf import pdf_extract_to_relational
from Caste_Project.ingest.handlers.document_docx import extract_docx_to_relational
from Caste_Project.ingest.handlers.tabular_csv import extract_csv_to_relational
from Caste_Project.ingest.handlers.tabular_xlsx import extract_excel_to_relational
from Caste_Project.ingest.handlers.structured_json import extract_json_to_relational
from Caste_Project.ingest.handlers.document_txt import extract_txt_to_relational
from Caste_Project.ingest.handlers.presentation_pptx import extract_pptx_to_relational, PptxReadConfig


def _pick_files(manifest: pd.DataFrame, ext: str, limit: int | None = None) -> pd.DataFrame:
    ext = ext.lower()
    m = manifest[manifest["ext"].str.lower() == ext].copy()
    if limit is not None:
        m = m.head(limit)
    return m


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder to discover files under (test input)")
    ap.add_argument("--out", required=True, help="Output folder for parquet results")
    ap.add_argument(
        "--ext",
        required=True,
        choices=[".pdf", ".docx", ".csv", ".xlsx", ".json", ".txt", ".pptx"],
        help="Which file type to test",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of files processed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) Discovery
    manifest = build_manifest(root)
    print(f"Discovered {len(manifest)} files total.")
    if {"rel_path", "ext"}.issubset(manifest.columns):
        print(manifest[["rel_path", "ext"]].head(20).to_string(index=False))

    # 2) Select matching files
    subset = _pick_files(manifest, args.ext, args.limit)
    if subset.empty:
        print(f"No files found for ext {args.ext}. Nothing to do.")
        return

    all_docs: list[pd.DataFrame] = []
    all_blocks: list[pd.DataFrame] = []
    all_sheets: list[pd.DataFrame] = []  # only for xlsx

    # 3) Run extraction on all matching files
    for i, r in subset.reset_index(drop=True).iterrows():
        abs_path = Path(r["abs_path"]).resolve()
        rel_path = str(r["rel_path"])

        print("\nTesting file:")
        print(f"  [{i+1}/{len(subset)}] ext: {r['ext']}")
        print(f"  abs: {abs_path}")
        print(f"  rel: {rel_path}")

        if args.ext == ".pdf":
            docs_df, pages_df = pdf_extract_to_relational(abs_path, rel_path)
            all_docs.append(docs_df)
            all_blocks.append(pages_df)

        elif args.ext == ".docx":
            docs_df, blocks_df = extract_docx_to_relational(abs_path, rel_path)
            all_docs.append(docs_df)
            all_blocks.append(blocks_df)

        elif args.ext == ".csv":
            docs_df, rows_df = extract_csv_to_relational(abs_path, rel_path)
            all_docs.append(docs_df)
            all_blocks.append(rows_df)

        elif args.ext == ".xlsx":
            docs_df, sheets_df, rows_df = extract_excel_to_relational(abs_path, rel_path)
            all_docs.append(docs_df)
            all_sheets.append(sheets_df)
            all_blocks.append(rows_df)

        elif args.ext == ".json":
            docs_df, records_df = extract_json_to_relational(abs_path, rel_path)
            all_docs.append(docs_df)
            all_blocks.append(records_df)

        elif args.ext == ".txt":
            docs_df, lines_df = extract_txt_to_relational(abs_path, rel_path)
            all_docs.append(docs_df)
            all_blocks.append(lines_df)

        elif args.ext == ".pptx":
            docs_df, blocks_df = extract_pptx_to_relational(
                abs_path,
                rel_path,
                PptxReadConfig(ocr_images=True),
            )
            all_docs.append(docs_df)
            all_blocks.append(blocks_df)

        else:
            raise ValueError(f"Unsupported ext {args.ext}")

    docs_all = _concat_or_empty(all_docs)
    blocks_all = _concat_or_empty(all_blocks)
    sheets_all = _concat_or_empty(all_sheets)

    # 4) Save + quick checks per ext
    if args.ext == ".pdf":
        docs_all.to_parquet(out / "pdf_documents.parquet", index=False)
        blocks_all.to_parquet(out / "pdf_pages.parquet", index=False)
        print(f"\nSaved: {out / 'pdf_documents.parquet'}")
        print(f"Saved: {out / 'pdf_pages.parquet'}")

        print("\nPDF quick checks:")
        cols = [c for c in ["doc_id", "num_pages", "num_image_pages", "error"] if c in docs_all.columns]
        print(docs_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "page_num", "page_text_len", "ocr_text_len", "has_images", "error"] if c in blocks_all.columns]
        print(blocks_all[cols].head(20).to_string(index=False))

    elif args.ext == ".docx":
        docs_all.to_parquet(out / "docx_documents.parquet", index=False)
        blocks_all.to_parquet(out / "docx_blocks.parquet", index=False)
        print(f"\nSaved: {out / 'docx_documents.parquet'}")
        print(f"Saved: {out / 'docx_blocks.parquet'}")

        print("\nDOCX quick checks:")
        cols = [c for c in ["doc_id", "num_blocks", "num_tables", "error"] if c in docs_all.columns]
        print(docs_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "block_num", "block_type", "has_equation", "error"] if c in blocks_all.columns]
        print(blocks_all[cols].head(20).to_string(index=False))

    elif args.ext == ".csv":
        docs_all.to_parquet(out / "csv_documents.parquet", index=False)
        blocks_all.to_parquet(out / "csv_rows.parquet", index=False)
        print(f"\nSaved: {out / 'csv_documents.parquet'}")
        print(f"Saved: {out / 'csv_rows.parquet'}")

        print("\nCSV quick checks:")
        cols = [c for c in ["doc_id", "num_rows", "num_cols", "error"] if c in docs_all.columns]
        print(docs_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "row_num", "row_text", "error"] if c in blocks_all.columns]
        print(blocks_all[cols].head(20).to_string(index=False))

    elif args.ext == ".xlsx":
        docs_all.to_parquet(out / "excel_documents.parquet", index=False)
        sheets_all.to_parquet(out / "excel_sheets.parquet", index=False)
        blocks_all.to_parquet(out / "excel_rows.parquet", index=False)

        print(f"\nSaved: {out / 'excel_documents.parquet'}")
        print(f"Saved: {out / 'excel_sheets.parquet'}")
        print(f"Saved: {out / 'excel_rows.parquet'}")

        print("\nEXCEL quick checks:")
        cols = [c for c in ["doc_id", "num_sheets", "total_rows", "error"] if c in docs_all.columns]
        print(docs_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "sheet_name", "sheet_index", "num_rows", "num_cols", "error"] if c in sheets_all.columns]
        print(sheets_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "sheet_name", "row_num", "row_text", "error"] if c in blocks_all.columns]
        print(blocks_all[cols].head(20).to_string(index=False))

    elif args.ext == ".json":
        docs_all.to_parquet(out / "json_documents.parquet", index=False)
        blocks_all.to_parquet(out / "json_records.parquet", index=False)

        print(f"\nSaved: {out / 'json_documents.parquet'}")
        print(f"Saved: {out / 'json_records.parquet'}")

        print("\nJSON quick checks:")
        cols = [c for c in ["doc_id", "top_level_type", "num_records", "error"] if c in docs_all.columns]
        print(docs_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "record_num", "record_type", "record_text", "error"] if c in blocks_all.columns]
        print(blocks_all[cols].head(20).to_string(index=False))

    elif args.ext == ".txt":
        docs_all.to_parquet(out / "txt_documents.parquet", index=False)
        blocks_all.to_parquet(out / "txt_lines.parquet", index=False)

        print(f"\nSaved: {out / 'txt_documents.parquet'}")
        print(f"Saved: {out / 'txt_lines.parquet'}")

        print("\nTXT quick checks:")
        cols = [c for c in ["doc_id", "num_lines", "error"] if c in docs_all.columns]
        print(docs_all[cols].head(20).to_string(index=False))
        cols = [c for c in ["doc_id", "line_num", "line_text", "error"] if c in blocks_all.columns]
        print(blocks_all[cols].head(20).to_string(index=False))

    elif args.ext == ".pptx":
        docs_all.to_parquet(out / "pptx_documents.parquet", index=False)
        blocks_all.to_parquet(out / "pptx_blocks.parquet", index=False)

        print(f"\nSaved: {out / 'pptx_documents.parquet'}")
        print(f"Saved: {out / 'pptx_blocks.parquet'}")

        print("\nPPTX quick checks:")
        cols = [
            c for c in [
                "doc_id",
                "num_slides",
                "num_blocks",
                "num_images",
                "num_images_ocr_success",
                "num_images_ocr_failed",
                "error",
            ]
            if c in docs_all.columns
        ]
        print(docs_all[cols].head(20).to_string(index=False))

        cols = [
            c for c in [
                "doc_id",
                "slide_num",
                "block_num",
                "block_type",
                "has_image",
                "ocr_text_len",
                "image_content_type",
                "error",
            ]
            if c in blocks_all.columns
        ]
        print(blocks_all[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
