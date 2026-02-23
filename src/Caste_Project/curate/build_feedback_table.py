from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    return str(x).strip()


def _pick_first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        s = _safe_str(v)
        if s:
            return s
    return None


def _split_name(full: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Conservative name split:
      - Keep full as-is.
      - If "Last, First" => split on comma.
      - Else if 2+ tokens => first token = first, last token = last.
      - Else leave first/last null.
    """
    full_s = _safe_str(full)
    if not full_s:
        return {"sender_full_name": None, "sender_first_name": None, "sender_last_name": None}

    # Remove extra whitespace
    full_s = re.sub(r"\s+", " ", full_s).strip()

    if "," in full_s:
        parts = [p.strip() for p in full_s.split(",", 1)]
        last = parts[0] if parts and parts[0] else None
        first = parts[1].split(" ", 1)[0].strip() if len(parts) > 1 and parts[1].strip() else None
        return {"sender_full_name": full_s, "sender_first_name": first, "sender_last_name": last}

    toks = full_s.split()
    if len(toks) >= 2:
        return {"sender_full_name": full_s, "sender_first_name": toks[0], "sender_last_name": toks[-1]}

    return {"sender_full_name": full_s, "sender_first_name": None, "sender_last_name": None}


def _extract_org_from_structured(raw_json_str: Optional[str]) -> Optional[str]:
    """
    Optional: try to find 'organization' or similar keys from raw row(s) json.
    Keep it conservative; return None if not found.
    """
    s = _safe_str(raw_json_str)
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None

    # raw_rows_json is list[dict], raw_row_json is dict
    candidates: List[Dict] = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else []
    keys = ["organization", "org", "agency", "company", "affiliation"]

    for d in candidates:
        if not isinstance(d, dict):
            continue
        # case-insensitive key match
        for k, v in d.items():
            nk = str(k).strip().lower()
            if any(t in nk for t in keys):
                val = _safe_str(v)
                if val:
                    return val
    return None


def _standardize_entry_fields(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """
    Map a parsed *_entry_fields.parquet into the canonical schema.
    """
    out = pd.DataFrame()

    # core identity / traceability
    out["doc_id"] = df.get("doc_id")
    out["sheet_name"] = df.get("sheet_name")  # exists for excel; ok if missing
    out["entry_num"] = df.get("entry_num")

    out["file_type"] = file_type

    file_path = df.get("source_rel_path")
    if file_path is None:
        file_path = df.get("source_file")
    out["file_path"] = file_path

    out["entry_type"] = df.get("entry_type")
    out["parse_reason"] = df.get("parse_reason")

    # date: prefer parsed date, then filename-derived file_date
    out["feedback_date"] = df.apply(lambda r: _pick_first_nonempty(r.get("date"), r.get("file_date")), axis=1)

    # sender: prefer parsed sender, then file_sender
    sender_full = df.apply(lambda r: _pick_first_nonempty(r.get("sender"), r.get("file_sender")), axis=1)
    name_parts = sender_full.map(_split_name)
    out["sender_full_name"] = name_parts.map(lambda d: d["sender_full_name"])
    out["sender_first_name"] = name_parts.map(lambda d: d["sender_first_name"])
    out["sender_last_name"] = name_parts.map(lambda d: d["sender_last_name"])

    # CQAS
    out["cqas_no"] = df.get("cqas_id")

    # incoming content
    out["incoming_feedback_content"] = df.get("feedback_text")

    # response (if present in parser output; else None)
    out["response"] = df.get("response") if "response" in df.columns else None

    # structured payload: keep raw json if available (especially for csv/xlsx)
    if "raw_rows_json" in df.columns:
        out["incoming_structured_json"] = df["raw_rows_json"]
    elif "raw_row_json" in df.columns:
        out["incoming_structured_json"] = df["raw_row_json"]
    else:
        out["incoming_structured_json"] = None

    # organization: if parser already extracted it, use it; else try to infer from structured json
    if "organization" in df.columns:
        out["organization"] = df["organization"]
    else:
        out["organization"] = out["incoming_structured_json"].map(_extract_org_from_structured)

    # subject is often useful even if you didn't ask—keep it optional
    out["subject"] = df.get("subject")

    return out


def build_feedback_table(parse_out_dir: Path, out_path: Path) -> pd.DataFrame:
    """
    Look for known parsed outputs in parse_out_dir and unify them.
    """
    sources = [
        ("csv", parse_out_dir / "csv_entry_fields.parquet"),
        ("xlsx", parse_out_dir / "excel_entry_fields.parquet"),
        ("pdf", parse_out_dir / "pdf_entry_fields.parquet"),
        ("docx", parse_out_dir / "docx_entry_fields.parquet"),
        ("pptx", parse_out_dir / "pptx_entry_fields.parquet"),
        ("json", parse_out_dir / "json_entry_fields.parquet"),
        ("txt", parse_out_dir / "txt_entry_fields.parquet"),
    ]

    parts: List[pd.DataFrame] = []
    for ftype, path in sources:
        if path.exists():
            df = pd.read_parquet(path)
            parts.append(_standardize_entry_fields(df, ftype))

    unified = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Drop summary sheets unless you want them included
    if "entry_type" in unified.columns:
        unified = unified[unified["entry_type"].fillna("").astype(str).str.lower() != "summary_sheet"].copy()

    unified.to_parquet(out_path, index=False)
    return unified


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parse_out_dir", required=True, help="Folder containing *_entry_fields.parquet files")
    ap.add_argument("--out_path", required=True, help="Where to write unified feedback_entries.parquet")
    args = ap.parse_args()

    parse_out_dir = Path(args.parse_out_dir)
    out_path = Path(args.out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_feedback_table(parse_out_dir, out_path)

    print(f"Saved: {out_path}")
    cols = [c for c in ["file_type", "file_path", "feedback_date", "sender_full_name", "organization", "cqas_no", "entry_type"] if c in df.columns]
    if not df.empty and cols:
        print(df[cols].head(30).to_string(index=False))