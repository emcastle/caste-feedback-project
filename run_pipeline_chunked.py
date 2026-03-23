"""
Docstring for run_pipeline_chunked

to run: (ensure feedback is activated )
    $env:PYTHONUNBUFFERED="1"; python -u run_pipeline_chunked.py --root "data\_nearly_full_input\Master_Data" --stage_out "data\_stage_master" --parse_out "data\_parse_master" --final_out "data\_final_master" --chunk_size 10

to run parts utilizing flags:

# RUN SCRIPT BUT SKIP INGEST (INGEST ALREADY RAN)
$env:PYTHONUNBUFFERED="1"; python -u run_pipeline_chunked.py `
  --root "data\_nearly_full_input\Master_Data" `
  --stage_out "data\_stage_master" `
  --parse_out "data\_parse_master" `
  --final_out "data\_final_master" `
  --skip_ingest



# SKIP INGEST & SEGMENT (INGEST + SEGMENT ALREADY RAN AND PARQUET FILES EXIST)
$env:PYTHONUNBUFFERED="1"; python -u run_pipeline_chunked.py `
  --root "data\_nearly_full_input\Master_Data" `
  --stage_out "data\_stage_master" `
  --parse_out "data\_parse_master" `
  --final_out "data\_final_master" `
  --skip_ingest `
  --skip_segment

"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

# Make repo_root/src importable (same trick as run_extraction_smoke_test.py)
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from Caste_Project.ingest.discover import build_manifest

from Caste_Project.ingest.handlers.document_pdf import pdf_extract_to_relational
from Caste_Project.ingest.handlers.document_docx import extract_docx_to_relational
from Caste_Project.ingest.handlers.tabular_csv import extract_csv_to_relational
from Caste_Project.ingest.handlers.tabular_xlsx import extract_excel_to_relational
from Caste_Project.ingest.handlers.structured_json import extract_json_to_relational
from Caste_Project.ingest.handlers.document_txt import extract_txt_to_relational
from Caste_Project.ingest.handlers.presentation_pptx import extract_pptx_to_relational, PptxReadConfig


def _path_hash_id(p: Path) -> str:
    # Stable-ish ID even if file is unreadable (lock file, perms)
    return hashlib.sha256(str(p).encode("utf-8")).hexdigest()


def _append_parquet(path: Path, df: pd.DataFrame, dedupe_subset: Optional[List[str]] = None) -> None:
    """
    Append by read+concat+write. Works fine for ~260 files.
    dedupe_subset lets you avoid duplicates if you re-run.
    """
    if df is None or df.empty:
        return

    if path.exists():
        old = pd.read_parquet(path)
        df2 = pd.concat([old, df], ignore_index=True)
        if dedupe_subset:
            # Keep last occurrence (newest)
            df2 = df2.drop_duplicates(subset=dedupe_subset, keep="last")
        df2.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)


def _filter_manifest(manifest: pd.DataFrame, exts: List[str], skip_lockfiles: bool = True) -> pd.DataFrame:
    m = manifest.copy()
    m["ext"] = m["ext"].astype(str).str.lower()
    exts = [e.lower() for e in exts]
    m = m[m["ext"].isin(exts)].copy()

    if skip_lockfiles:
        # Skip Excel lock/temp files like "~$foo.xlsx"
        # We can also skip other temp artifacts similarly if needed.
        rp = m["rel_path"].astype(str)
        m = m[~rp.str.contains(r"\\~\$|/~\$", regex=True)].copy()

    return m


def _chunk_indices(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        out.append((i, j))
        i = j
    return out


def _run_subprocess(cmd: List[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)  # ensure -m modules can import from src
    print("\nRUN:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

"""
def ingest_in_chunks(root: Path, stage_out: Path, exts: List[str], chunk_size: int, limit_total: Optional[int]) -> None:
    stage_out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root)
    print(f"Discovered {len(manifest)} files total under {root}")

    subset = _filter_manifest(manifest, exts=exts, skip_lockfiles=True)

    if subset.empty:
        print("No matching files for requested types. Nothing to ingest.")
        return

    if limit_total is not None:
        subset = subset.head(limit_total).copy()

    total = len(subset)
    print(f"Will ingest {total} files (types: {', '.join(exts)}), chunk_size={chunk_size}")

    # Process in chunks
    chunks = _chunk_indices(total, chunk_size)

    # For pptx OCR config (you used ocr_images=True in smoke test)
    pptx_cfg = PptxReadConfig(ocr_images=True)

    for ci, (start, end) in enumerate(chunks, start=1):
        chunk = subset.iloc[start:end].copy()
        print(f"\n=== Chunk {ci}/{len(chunks)}: files {start+1}-{end} of {total} ===")

        # collect per chunk
        all_docs: List[pd.DataFrame] = []
        all_blocks: List[pd.DataFrame] = []
        all_sheets: List[pd.DataFrame] = []  # xlsx only

        for idx, r in chunk.reset_index(drop=True).iterrows():
            abs_path = Path(r["abs_path"]).resolve()
            rel_path = str(r["rel_path"])
            ext = str(r["ext"]).lower()

            global_i = start + idx + 1
            print(f"[{global_i}/{total}] ingest {ext}  rel={rel_path}")

            try:
                if ext == ".pdf":
                    docs_df, pages_df = pdf_extract_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_blocks.append(pages_df)

                elif ext == ".docx":
                    docs_df, blocks_df = extract_docx_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_blocks.append(blocks_df)

                elif ext == ".csv":
                    docs_df, rows_df = extract_csv_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_blocks.append(rows_df)

                elif ext == ".xlsx":
                    docs_df, sheets_df, rows_df = extract_excel_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_sheets.append(sheets_df)
                    all_blocks.append(rows_df)

                elif ext == ".json":
                    docs_df, records_df = extract_json_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_blocks.append(records_df)

                elif ext == ".txt":
                    docs_df, lines_df = extract_txt_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_blocks.append(lines_df)

                elif ext == ".pptx":
                    docs_df, blocks_df = extract_pptx_to_relational(abs_path, rel_path, pptx_cfg)
                    all_docs.append(docs_df)
                    all_blocks.append(blocks_df)

                else:
                    # shouldn't happen due to filtering
                    raise ValueError(f"Unsupported ext: {ext}")

            except Exception as e:
                # Per-file failure: keep going; write an error doc row so you can audit.
                doc_id = _path_hash_id(abs_path)
                err = f"{type(e).__name__}: {e}"
                print(f"  ERROR: {err}")

                # Minimal docs row, aligned with your other docs parquet conventions
                docs_df = pd.DataFrame([{
                    "doc_id": doc_id,
                    "source_file": abs_path.name,
                    "source_rel_path": rel_path,
                    "ext": ext,
                    "error": err,
                }])
                all_docs.append(docs_df)

        # write chunk results by ext type into stage_out
        # We split by ext because each handler produces a different schema.
        # Instead of trying to merge them now, we append to the correct parquet files.
        if all_docs:
            docs_all = pd.concat(all_docs, ignore_index=True)
        else:
            docs_all = pd.DataFrame()

        # Append documents by ext group
        # We use rel_path+doc_id as dedupe keys (doc_id should be stable for readable files).
        # If doc_id schema differs per handler, this is still safe enough for your scale.
        for ext in exts:
            ext_low = ext.lower()

            # fix attempt 
            if "source_ext" in docs_all.columns:
                docs_ext = docs_all[
                    docs_all["source_ext"].astype(str).str.lower() == ext_low
                ].copy()

                # fallback 
            elif "source_rel_path" in docs_all_columns:
                docs_ext = docs_all[
                    docs_all["source_rel_path"].astype(str).str.lower().str.endswith(ext_low)
                ].copy()
            else:
                docs_ext = pd.DataFrame()
    
            
            # previously commented out when degbugging parse issue 
            #docs_ext = docs_all[docs_all.get("ext", pd.Series([""] * len(docs_all))).astype(str).str.lower() == ext_low].copy() if "ext" in docs_all.columns else pd.DataFrame()
            # docs from successful extractions likely don't have "ext" column; so we also route by rel_path suffix:
            #if docs_ext.empty and not docs_all.empty and "source_rel_path" in docs_all.columns:
            #    docs_ext = docs_all[docs_all["source_rel_path"].astype(str).str.lower().str.endswith(ext_low)].copy()
            
            if docs_ext.empty:
                continue

            if ext_low == ".csv":
                _append_parquet(stage_out / "csv_documents.parquet", docs_ext, dedupe_subset=["doc_id"])
            elif ext_low == ".xlsx":
                _append_parquet(stage_out / "excel_documents.parquet", docs_ext, dedupe_subset=["doc_id"])
            elif ext_low == ".pdf":
                _append_parquet(stage_out / "pdf_documents.parquet", docs_ext, dedupe_subset=["doc_id"])
            elif ext_low == ".docx":
                _append_parquet(stage_out / "docx_documents.parquet", docs_ext, dedupe_subset=["doc_id"])
            elif ext_low == ".json":
                _append_parquet(stage_out / "json_documents.parquet", docs_ext, dedupe_subset=["doc_id"])
            elif ext_low == ".txt":
                _append_parquet(stage_out / "txt_documents.parquet", docs_ext, dedupe_subset=["doc_id"])
            elif ext_low == ".pptx":
                _append_parquet(stage_out / "pptx_documents.parquet", docs_ext, dedupe_subset=["doc_id"])

        # Append block-level outputs
        if all_blocks:
            blocks_all = pd.concat(all_blocks, ignore_index=True)
        else:
            blocks_all = pd.DataFrame()

        # Route blocks by presence of expected columns
        # CSV rows
        if not blocks_all.empty and {"row_num", "row_text"}.issubset(blocks_all.columns) and "sheet_name" not in blocks_all.columns:
            # this could include csv or txt lines; route by whether source_rel_path endswith .csv if available
            if "source_rel_path" in blocks_all.columns:
                csv_rows = blocks_all[blocks_all["source_rel_path"].astype(str).str.lower().str.endswith(".csv")].copy()
                txt_lines = blocks_all[blocks_all["source_rel_path"].astype(str).str.lower().str.endswith(".txt")].copy()
            else:
                csv_rows = blocks_all.copy()
                txt_lines = pd.DataFrame()

            if not csv_rows.empty:
                _append_parquet(stage_out / "csv_rows.parquet", csv_rows, dedupe_subset=["doc_id", "row_num"])
            if not txt_lines.empty:
                _append_parquet(stage_out / "txt_lines.parquet", txt_lines, dedupe_subset=["doc_id", "line_num"] if "line_num" in txt_lines.columns else None)

        # Excel rows (has sheet_name)
        if not blocks_all.empty and "sheet_name" in blocks_all.columns and {"doc_id", "row_num"}.issubset(blocks_all.columns):
            _append_parquet(stage_out / "excel_rows.parquet", blocks_all, dedupe_subset=["doc_id", "sheet_name", "row_num"])

        # PDF pages
        if not blocks_all.empty and "page_num" in blocks_all.columns:
            _append_parquet(stage_out / "pdf_pages.parquet", blocks_all, dedupe_subset=["doc_id", "page_num"])

        # DOCX blocks
        if not blocks_all.empty and "block_num" in blocks_all.columns and "block_type" in blocks_all.columns:
            _append_parquet(stage_out / "docx_blocks.parquet", blocks_all, dedupe_subset=["doc_id", "block_num"])

        # JSON records
        if not blocks_all.empty and "record_num" in blocks_all.columns:
            _append_parquet(stage_out / "json_records.parquet", blocks_all, dedupe_subset=["doc_id", "record_num"])

        # PPTX blocks
        if not blocks_all.empty and "slide_num" in blocks_all.columns:
            _append_parquet(stage_out / "pptx_blocks.parquet", blocks_all, dedupe_subset=["doc_id", "slide_num", "block_num"] if "block_num" in blocks_all.columns else None)

        # Excel sheets
        if all_sheets:
            sheets_all = pd.concat(all_sheets, ignore_index=True)
            _append_parquet(stage_out / "excel_sheets.parquet", sheets_all, dedupe_subset=["doc_id", "sheet_name"] if "sheet_name" in sheets_all.columns else ["doc_id"])

    print("\nINGEST COMPLETE.")
    print(f"Stage outputs in: {stage_out}")
"""

# new ingest_in_chuncks 
# previous function error appended all block filetypes together which results in mismatches in doc_ids counts

def ingest_in_chunks(root: Path, stage_out: Path, exts: List[str], chunk_size: int, limit_total: Optional[int]) -> None:
    stage_out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root)
    print(f"Discovered {len(manifest)} files total under {root}")

    subset = _filter_manifest(manifest, exts=exts, skip_lockfiles=True)

    if subset.empty:
        print("No matching files for requested types. Nothing to ingest.")
        return

    if limit_total is not None:
        subset = subset.head(limit_total).copy()

    total = len(subset)
    print(f"Will ingest {total} files (types: {', '.join(exts)}), chunk_size={chunk_size}")

    chunks = _chunk_indices(total, chunk_size)
    pptx_cfg = PptxReadConfig(ocr_images=True)

    for ci, (start, end) in enumerate(chunks, start=1):
        chunk = subset.iloc[start:end].copy()
        print(f"\n=== Chunk {ci}/{len(chunks)}: files {start+1}-{end} of {total} ===")

        # -------------------------
        # Per-chunk collectors
        # -------------------------
        all_docs: List[pd.DataFrame] = []

        docx_blocks_list: List[pd.DataFrame] = []
        csv_rows_list: List[pd.DataFrame] = []
        excel_rows_list: List[pd.DataFrame] = []
        pdf_pages_list: List[pd.DataFrame] = []
        json_records_list: List[pd.DataFrame] = []
        txt_lines_list: List[pd.DataFrame] = []
        pptx_blocks_list: List[pd.DataFrame] = []
        all_sheets: List[pd.DataFrame] = []

        # -------------------------
        # Ingest loop
        # -------------------------
        for idx, r in chunk.reset_index(drop=True).iterrows():
            abs_path = Path(r["abs_path"]).resolve()
            rel_path = str(r["rel_path"])
            ext = str(r["ext"]).lower()

            global_i = start + idx + 1
            print(f"[{global_i}/{total}] ingest {ext}  rel={rel_path}")

            try:
                if ext == ".pdf":
                    docs_df, pages_df = pdf_extract_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    pdf_pages_list.append(pages_df)

                elif ext == ".docx":
                    docs_df, blocks_df = extract_docx_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    docx_blocks_list.append(blocks_df)

                elif ext == ".csv":
                    docs_df, rows_df = extract_csv_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    csv_rows_list.append(rows_df)

                elif ext == ".xlsx":
                    docs_df, sheets_df, rows_df = extract_excel_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    all_sheets.append(sheets_df)
                    excel_rows_list.append(rows_df)

                elif ext == ".json":
                    docs_df, records_df = extract_json_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    json_records_list.append(records_df)

                elif ext == ".txt":
                    docs_df, lines_df = extract_txt_to_relational(abs_path, rel_path)
                    all_docs.append(docs_df)
                    txt_lines_list.append(lines_df)

                elif ext == ".pptx":
                    docs_df, blocks_df = extract_pptx_to_relational(abs_path, rel_path, pptx_cfg)
                    all_docs.append(docs_df)
                    pptx_blocks_list.append(blocks_df)

                else:
                    raise ValueError(f"Unsupported ext: {ext}")

            except Exception as e:
                doc_id = _path_hash_id(abs_path)
                err = f"{type(e).__name__}: {e}"
                print(f"  ERROR: {err}")

                docs_df = pd.DataFrame([{
                    "doc_id": doc_id,
                    "source_file": abs_path.name,
                    "source_rel_path": rel_path,
                    "source_ext": ext,
                    "error": err,
                }])
                all_docs.append(docs_df)

        # -------------------------
        # DOCUMENTS WRITE (fixed)
        # -------------------------
        if all_docs:
            docs_all = pd.concat(all_docs, ignore_index=True)
        else:
            docs_all = pd.DataFrame()

        if not docs_all.empty and "source_ext" in docs_all.columns:
            for ext in exts:
                ext_low = ext.lower()

                docs_ext = docs_all[
                    docs_all["source_ext"].astype(str).str.lower() == ext_low
                ].copy()

                if docs_ext.empty:
                    continue

                if ext_low == ".csv":
                    _append_parquet(stage_out / "csv_documents.parquet", docs_ext, ["doc_id"])
                elif ext_low == ".xlsx":
                    _append_parquet(stage_out / "excel_documents.parquet", docs_ext, ["doc_id"])
                elif ext_low == ".pdf":
                    _append_parquet(stage_out / "pdf_documents.parquet", docs_ext, ["doc_id"])
                elif ext_low == ".docx":
                    _append_parquet(stage_out / "docx_documents.parquet", docs_ext, ["doc_id"])
                elif ext_low == ".json":
                    _append_parquet(stage_out / "json_documents.parquet", docs_ext, ["doc_id"])
                elif ext_low == ".txt":
                    _append_parquet(stage_out / "txt_documents.parquet", docs_ext, ["doc_id"])
                elif ext_low == ".pptx":
                    _append_parquet(stage_out / "pptx_documents.parquet", docs_ext, ["doc_id"])

        # -------------------------
        # BLOCK WRITES (FIXED)
        # -------------------------
        if docx_blocks_list:
            _append_parquet(
                stage_out / "docx_blocks.parquet",
                pd.concat(docx_blocks_list, ignore_index=True),
                ["doc_id", "block_num"]
            )

        if csv_rows_list:
            _append_parquet(
                stage_out / "csv_rows.parquet",
                pd.concat(csv_rows_list, ignore_index=True),
                ["doc_id", "row_num"]
            )

        if excel_rows_list:
            _append_parquet(
                stage_out / "excel_rows.parquet",
                pd.concat(excel_rows_list, ignore_index=True),
                ["doc_id", "sheet_name", "row_num"]
            )

        if pdf_pages_list:
            _append_parquet(
                stage_out / "pdf_pages.parquet",
                pd.concat(pdf_pages_list, ignore_index=True),
                ["doc_id", "page_num"]
            )

        if json_records_list:
            _append_parquet(
                stage_out / "json_records.parquet",
                pd.concat(json_records_list, ignore_index=True),
                ["doc_id", "record_num"]
            )

        if txt_lines_list:
            _append_parquet(
                stage_out / "txt_lines.parquet",
                pd.concat(txt_lines_list, ignore_index=True),
                ["doc_id", "line_num"]
            )

        if pptx_blocks_list:
            _append_parquet(
                stage_out / "pptx_blocks.parquet",
                pd.concat(pptx_blocks_list, ignore_index=True),
                ["doc_id", "slide_num", "block_num"]
            )

        if all_sheets:
            _append_parquet(
                stage_out / "excel_sheets.parquet",
                pd.concat(all_sheets, ignore_index=True),
                ["doc_id", "sheet_name"]
            )

    print("\nINGEST COMPLETE.")
    print(f"Stage outputs in: {stage_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing raw files")
    ap.add_argument("--stage_out", required=True, help="Where to write ingest relational parquet outputs")
    ap.add_argument("--parse_out", required=True, help="Where to write parsed *_entry_fields.parquet outputs")
    ap.add_argument("--final_out", required=True, help="Where to write unified feedback table")
    ap.add_argument("--chunk_size", type=int, default=10)
    ap.add_argument("--limit_total", type=int, default=None, help="Optional cap for testing")
    ap.add_argument(
        "--types",
        default=".csv,.xlsx,.pdf,.docx,.json,.txt,.pptx",
        help="Comma-separated list of extensions, e.g. .csv,.xlsx",
    )
    ap.add_argument("--skip_ingest", action="store_true", help="Skip ingest step")
    ap.add_argument("--skip_segment", action="store_true", help="Skip segment step")
    ap.add_argument("--skip_parse", action="store_true", help="Skip parse step")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    stage_out = Path(args.stage_out).resolve()
    parse_out = Path(args.parse_out).resolve()
    final_out = Path(args.final_out).resolve()
    parse_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)

    exts = [e.strip() for e in args.types.split(",") if e.strip()]
    # ingest_in_chunks(root, stage_out, exts=exts, chunk_size=args.chunk_size, limit_total=args.limit_total)

    if not args.skip_ingest:
        print("\n== INGEST ===")
        ingest_in_chunks(root, stage_out, exts=exts, chunk_size=args.chunk_size, limit_total=args.limit_total)
    else:
        print("\n== SKIPPING INGEST ===")

    # ---- SEGMENT ----
    # Segmenters generally expect the ingest outputs in stage_out.
    # Only run if the expected inputs exist.
    """
    # csv segment
    if (stage_out / "csv_documents.parquet").exists() and (stage_out / "csv_rows.parquet").exists():
        _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_csv",
                         "--in_dir", str(stage_out), "--out_dir", str(stage_out)])

    # xlsx (excel) segment
    if (stage_out / "excel_documents.parquet").exists() and (stage_out / "excel_rows.parquet").exists():
        _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_xlsx",
                         "--in_dir", str(stage_out), "--out_dir", str(stage_out)])

    # pdf segment
    if(stage_out / "pdf_documents.parquet").exists() and (stage_out / "pdf_entries.parquet").exists():
        _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_pdf",
                         "--in_dir", str(stage_out), "--out_dir", str(stage_out)])
        
    # docx segment
    if(stage_out / "docx_documents.parquet").exists() and (stage_out / "docx_entries.parquet").exists():
        _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_docx",
                         "--in_dir", str(stage_out), "--out_dir", str(stage_out)])
    
    # json segment
    if(stage_out / "json_documents.parquet").exists() and (stage_out / "json_entries.parquet").exists():
        _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_pptx",
                         "--in_dir", str(stage_out), "--out_dir", str(stage_out)])
        
    # pptx segment
    if(stage_out / "pptx_documents.parquet").exists() and (stage_out / "pptx_entries.parquet").exists():
        _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_pptx",
                         "--in_dir", str(stage_out), "--out_dir", str(stage_out)])
    """

    # ---- SEGMENT ----
    # Segment ALL file types into entries
    # wrap segment block 
    if not args.skip_segment:
        print("\n=== SEGMENT ===")
        # CSV
        if (stage_out / "csv_rows.parquet").exists():
            _run_subprocess([
                "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_csv",
                "--in_dir", str(stage_out),
                "--out_dir", str(stage_out)
            ])

        # XLSX
        if (stage_out / "excel_rows.parquet").exists():
            _run_subprocess([
                "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_xlsx",
                "--in_dir", str(stage_out),
                "--out_dir", str(stage_out)
            ])

        # DOCX
        if (stage_out / "docx_blocks.parquet").exists():
            _run_subprocess([
                "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_docx",
                "--in_dir", str(stage_out),
                "--out_dir", str(stage_out)
            ])

        # PDF
        if (stage_out / "pdf_pages.parquet").exists():
            _run_subprocess([
                "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_pdf",
                "--in_dir", str(stage_out),
                "--out_dir", str(stage_out)
            ])

        # PPTX
        if (stage_out / "pptx_blocks.parquet").exists():
            _run_subprocess([
                "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_pptx",
                "--in_dir", str(stage_out),
                "--out_dir", str(stage_out)
            ])

        # JSON
        if (stage_out / "json_records.parquet").exists():
            _run_subprocess([
                "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.segment.segment_json",
                "--in_dir", str(stage_out),
                "--out_dir", str(stage_out)
            ])
    else: 
        print("\n=== SKIPPING SEGMENT ===")

    print("\n=== SEGMENT OUTPUT CHECK ===")
    for f in [
        "csv_entries.parquet",
        "excel_entries.parquet",
        "docx_entries.parquet",
        "pdf_entries.parquet",
        "pptx_entries.parquet",
        "json_entries.parquet"
    ]:
        print(f, (stage_out / f).exists())

    # ---- PARSE ----

    # wrap the parse 
    if not args.skip_parse:
        print("\n=== PARSE ===")

        # CSV parse (expects csv_entries.parquet by default in --in_dir)
        if (stage_out / "csv_entries.parquet").exists():
            _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.parse.parse_csv",
                            "--in_dir", str(stage_out),
                            "--out_dir", str(parse_out),
                            "--docs_dir", str(stage_out),
                            "--granularity", "auto",
                            "--keep_summaries"])

        # XLSX parse (we parse from excel_rows.parquet; parse_xlsx supports this)
        # maybe the comment above is incorrect
        if (stage_out / "excel_entries.parquet").exists():
            _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.parse.parse_xlsx",
                            "--in_dir", str(stage_out),
                            "--out_dir", str(parse_out),
                            "--docs_dir", str(stage_out),
                            # "--rows_name", "excel_rows.parquet",
                            "--documents_name", "excel_documents.parquet",
                            "--granularity", "auto",
                            "--keep_summaries"])

        # PDF parse 
        if(stage_out / "pdf_entries.parquet").exists():
            _run_subprocess(["conda", "run", "-n", "feedback", "python","-m", "Caste_Project.parse.parse_pdf",
                            "--in_dir", str(stage_out),
                            "--out_dir", str(parse_out),
                            "--docs_dir", str(stage_out)])
        
        # docx parse 
        if(stage_out / "docx_entries.parquet").exists():
            _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.parse.parse_docx",
                        "--in_dir", str(stage_out),
                            "--out_dir", str(parse_out),
                            "--docs_dir", str(stage_out)])
        
        # pptx parse 
        if(stage_out / "pptx_entries.parquet").exists():
            _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.parse.parse_pptx",
                            "--in_dir", str(stage_out),
                            "--out_dir", str(parse_out),
                            "--docs_dir", str(stage_out),
                            "--strip_anchor_lines"])
            
        # json parse
        if(stage_out / "json_entries.parquet").exists():
            _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.parse.parse_json",
                            "--in_dir", str(stage_out),
                            "--out_dir", str(parse_out),
                            "--docs_dir", str(stage_out)])

    else: 
        print("\n=== SKIPPING PARSE ===")

    # ---- CONSOLIDATE ----
    # created Caste_Project.curate.build_feedback_table

    if not args.skip_parse:
        print("\n=== CONSOLIDATE ===")

        _run_subprocess([
            "conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.curate.build_feedback_table",
            "--parse_out_dir", str(parse_out),
            "--out_path", str(final_out / "feedback_entries.parquet")
        ])
    else:
        print("\n=== SKIPPING CONSOLIDATE (parse skipped) ===")
        

    """
    _run_subprocess(["conda", "run", "-n", "feedback", "python", "-m", "Caste_Project.curate.build_feedback_table",
                     "--parse_out_dir", str(parse_out),
                     "--out_path", str(final_out / "feedback_entries.parquet")])

    print("\nDONE.")
    print("Stage:", stage_out)
    print("Parsed:", parse_out)
    print("Final:", final_out / "feedback_entries.parquet")
    """


if __name__ == "__main__":
    main()