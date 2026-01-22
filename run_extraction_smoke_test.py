"""
Docstring for Feedback_Caste.run_extraction_smoke_test


Use this script to test one single file from a specified extension type


test run command examples:
conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .pdf
conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .docx
conda run -n feedback python run_extraction_smoke_test.py --root data\_test_input --out data\_test_output --ext .csv
"""


from __future__ import annotations
import sys
import argparse
from pathlib import Path

import pandas as pd

# Modify sys path to have folder searchable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Import your discovery + handlers
# Update these imports to match your actual module paths.
from Caste_Project.ingest.discover import build_manifest  # expected: build_manifest(root: Path) -> pd.DataFrame
from Caste_Project.ingest.handlers.document_pdf import pdf_extract_to_relational
from Caste_Project.ingest.handlers.document_docx import extract_docx_to_relational
from Caste_Project.ingest.handlers.tabular_csv import extract_csv_to_relational
from Caste_Project.ingest.handlers.tabular_excel import extract_excel_to_relational
from Caste_Project.ingest.handlers.structured_json import extract_json_to_relational
from Caste_Project.ingest.handlers.document_txt import extract_txt_to_relational
from Caste_Project.ingest.handlers.presentation_pptx import (extract_pptx_to_relational, 
                                                             PptxReadConfig)
# from Caste_Project.ingest.handlers.tabular_parquet import extract_parquet_to_relational

"""
def _pick_first_file(manifest: pd.DataFrame, ext: str) -> pd.Series:
    m = manifest[manifest["ext"].str.lower() == ext.lower()]
    if m.empty:
        raise FileNotFoundError(f"No files with ext={ext} found in manifest.")
    return m.sort_values(["rel_path"]).iloc[1]
"""
def _pick_files(manifest: pd.DataFrame, ext: str, limit: int | None = None) -> pd.DataFrame:
    ext = ext.lower()
    m = manifest[manifest["ext"].str.lower() == ext].copy()
    if m.empty:
        raise FileNotFoundError(f"No files with ext={ext} found in manifest.")
    if limit is not None:
        m = m.head(limit)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder to discover files under (test input)")
    ap.add_argument("--out", required=True, help="Output folder for parquet results")
    ap.add_argument("--ext", required=True, choices=[".pdf",
                                                      ".docx",
                                                      ".csv",
                                                      ".xlsx",
                                                      ".json",
                                                      ".txt",
                                                      ".pptx"], help="Which file type to test")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) Discovery
    manifest = build_manifest(root)
    print(f"Discovered {len(manifest)} files total.")
    print(manifest[["rel_path", "ext"]].head(20).to_string(index=False))

    """
    # 2) Pick one file of the requested type
    row = _pick_first_file(manifest, args.ext)
    abs_path = Path(row["abs_path"]).resolve()
    rel_path = str(row["rel_path"])
    """

    # 2) Iterate through all files 
    subset = _pick_files(manifest, args.ext, args.limit)

    all_docs = []
    all_blocks = []  # pages/rows/blocks depending on ext

    for i, r in subset.reset_index(drop=True).iterrows():
        abs_path = Path(r["abs_path"])
        rel_path = r["rel_path"]

    print("\nTesting file:")
    print(f"  [{i+1}/{len(subset)}] ext: {r['ext']}")
    print(f"  abs: {abs_path}")
    print(f"  rel: {rel_path}")

    if args.ext == ".pdf":
        docs_df, pages_df = extract_pdf_to_relational(abs_path, rel_path)
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

    elif args.ext in (".xlsx", ".xls"):
        docs_df, rows_df = extract_excel_to_relational(abs_path, rel_path)
        all_docs.append(docs_df)
        all_blocks.append(rows_df)

    elif args.ext in (".json", ".json5"):
        docs_df, rows_df = extract_json_to_relational(abs_path, rel_path)
        all_docs.append(docs_df)
        all_blocks.append(rows_df)

    elif args.ext == ".pptx":
        docs_df, blocks_df = extract_pptx_to_relational(abs_path, rel_path, PptxReadConfig(ocr_images=True))
        all_docs.append(docs_df)
        all_blocks.append(blocks_df)

    else:
        raise ValueError(f"Unsupported ext {args.ext}")


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

    elif args.ext == ".xlsx":
        
        docs_df, sheets_df, rows_df = extract_excel_to_relational(abs_path, rel_path)

        docs_df.to_parquet(out / "excel_documents.parquet", index=False)
        sheets_df.to_parquet(out / "excel_sheets.parquet", index=False)
        rows_df.to_parquet(out / "excel_rows.parquet", index=False)

        print(f"Saved: {out / 'excel_documents.parquet'}")
        print(f"Saved: {out / 'excel_sheets.parquet'}")
        print(f"Saved: {out / 'excel_rows.parquet'}")

        print("\nEXCEL quick checks:")
        print(docs_df[["doc_id", "num_sheets", "total_rows", "error"]].to_string(index=False))

        sheet_cols = ["doc_id", "sheet_name", "sheet_index", "num_rows", "num_cols", "error"]
        sheet_cols = [c for c in sheet_cols if c in sheets_df.columns]
        print(sheets_df[sheet_cols].head(20).to_string(index=False))

        row_cols = ["doc_id", "sheet_name", "row_num", "row_text", "error"]
        row_cols = [c for c in row_cols if c in rows_df.columns]
        print(rows_df[row_cols].head(10).to_string(index=False))


    elif args.ext == ".json":
        docs_df, records_df = extract_json_to_relational(abs_path, rel_path)
        docs_df.to_parquet(out / "json_documents.parquet", index=False)
        records_df.to_parquet(out / "json_records.parquet", index=False)

        print(f"Saved: {out / 'json_documents.parquet'}")
        print(f"Saved: {out / 'json_records.parquet'}")

        print("\nJSON quick checks:")
        print(docs_df[["doc_id", "top_level_type", "num_records", "error"]].to_string(index=False))
        cols = ["doc_id", "record_num", "record_type", "record_text", "error"]
        cols = [c for c in cols if c in records_df.columns]
        print(records_df[cols].head(10).to_string(index=False))

    elif args.ext == ".txt":
        docs_df, lines_df = extract_txt_to_relational(abs_path, rel_path)
        docs_df.to_parquet(out / "txt_documents.parquet", index=False)
        lines_df.to_parquet(out / "txt_lines.parquet", index=False)

        print(f"Saved: {out / 'txt_documents.parquet'}")
        print(f"Saved: {out / 'txt_lines.parquet'}")

        print("\nTXT quick checks:")
        print(docs_df[["doc_id", "num_lines", "error"]].to_string(index=False))
        cols = ["doc_id", "line_num", "line_text", "error"]
        cols = [c for c in cols if c in lines_df.columns]
        print(lines_df[cols].head(20).to_string(index=False))


    elif args.ext == ".pptx":
        docs_df, blocks_df = extract_pptx_to_relational(abs_path, rel_path, PptxReadConfig(ocr_images=False))
        docs_df.to_parquet(out / "pptx_documents.parquet", index=False)
        blocks_df.to_parquet(out / "pptx_blocks.parquet", index=False)
        print(f"Saved: {out / 'pptx_documents.parquet'}")
        print(f"Saved: {out / 'pptx_blocks.parquet'}")

        print("\nPPTX quick checks:")
        print(docs_df[["doc_id", "num_slides", "num_blocks", "num_images", "num_images_ocr_success", "num_images_ocr_failed", "error"]].to_string(index=False))
        cols = ["doc_id", "slide_num", "block_num", "block_type", "has_image", "ocr_text_len", "image_content_type", "error"]
        cols = [c for c in cols if c in blocks_df.columns]
        print(blocks_df[cols].head(20).to_string(index=False))


    else:
        raise ValueError(f"Unsupported ext {args.ext}")


if __name__ == "__main__":
    main()