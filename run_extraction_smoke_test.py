"""
Docstring for Feedback_Caste.run_extraction_smoke_test

test run commands 
conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .pdf
conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .docx
conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Import your discovery + handlers
# Update these imports to match your actual module paths.
from Caste_Project.ingest.discovery import build_manifest  # expected: build_manifest(root: Path) -> pd.DataFrame
from Caste_Project.handlers.document_pdf import pdf_extract_to_relational
from Caste_Project.handlers.document_docx import extract_docx_to_relational
from Caste_Project.handlers.document_csv import extract_csv_to_relational


def _pick_first_file(manifest: pd.DataFrame, ext: str) -> pd.Series:
    m = manifest[manifest["ext"].str.lower() == ext.lower()]
    if m.empty:
        raise FileNotFoundError(f"No files with ext={ext} found in manifest.")
    return m.sort_values(["source_rel_path"]).iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder to discover files under (test input)")
    ap.add_argument("--out", required=True, help="Output folder for parquet results")
    ap.add_argument("--ext", required=True, choices=[".pdf", ".docx", ".csv"], help="Which file type to test")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) Discovery
    manifest = build_manifest(root)
    print(f"Discovered {len(manifest)} files total.")
    print(manifest[["source_rel_path", "ext"]].head(20).to_string(index=False))

    # 2) Pick one file of the requested type
    row = _pick_first_file(manifest, args.ext)
    abs_path = Path(row["abs_path"]).resolve()
    rel_path = str(row["source_rel_path"])

    print(f"\nTesting one file:")
    print(f"  ext: {args.ext}")
    print(f"  abs: {abs_path}")
    print(f"  rel: {rel_path}")

    # 3) Run handler + save outputs
    if args.ext == ".pdf":
        docs_df, pages_df = pdf_extract_to_relational(abs_path, rel_path)
        docs_df.to_parquet(out / "pdf_documents.parquet", index=False)
        pages_df.to_parquet(out / "pdf_pages.parquet", index=False)
        print(f"Saved: {out / 'pdf_documents.parquet'}")
        print(f"Saved: {out / 'pdf_pages.parquet'}")

        print("\nPDF quick checks:")
        print(docs_df[["doc_id", "num_pages", "num_image_pages", "error"]].to_string(index=False))
        print(pages_df[["doc_id", "page_num", "page_text_len", "ocr_text_len", "has_images", "error"]].head(10).to_string(index=False))

    elif args.ext == ".docx":
        docs_df, blocks_df = extract_docx_to_relational(abs_path, rel_path)
        docs_df.to_parquet(out / "docx_documents.parquet", index=False)
        blocks_df.to_parquet(out / "docx_blocks.parquet", index=False)
        print(f"Saved: {out / 'docx_documents.parquet'}")
        print(f"Saved: {out / 'docx_blocks.parquet'}")

        print("\nDOCX quick checks:")
        print(docs_df[["doc_id", "num_blocks", "num_tables", "error"]].to_string(index=False))
        print(blocks_df[["doc_id", "block_num", "block_type", "has_equation", "error"]].head(15).to_string(index=False))

    elif args.ext == ".csv":
        docs_df, rows_df = extract_csv_to_relational(abs_path, rel_path)
        docs_df.to_parquet(out / "csv_documents.parquet", index=False)
        rows_df.to_parquet(out / "csv_rows.parquet", index=False)
        print(f"Saved: {out / 'csv_documents.parquet'}")
        print(f"Saved: {out / 'csv_rows.parquet'}")

        print("\nCSV quick checks:")
        print(docs_df[["doc_id", "num_rows", "num_cols", "error"]].to_string(index=False))
        cols = ["doc_id", "row_num", "row_text", "error"]
        cols = [c for c in cols if c in rows_df.columns]
        print(rows_df[cols].head(10).to_string(index=False))

    else:
        raise ValueError(f"Unsupported ext {args.ext}")


if __name__ == "__main__":
    main()