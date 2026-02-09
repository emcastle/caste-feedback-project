from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class CsvParseConfig:
    """
    Parsing goal for CSV:
      - One CSV *row* = one "entry"
      - Extract best-effort fields (date/sender/receiver/subject/cqas)
      - Provide feedback_text for downstream NLP

    Expects:
    in_dir/csv_entries.parquet
    docs_dir/csv_documents.parquet (recommended)

    Notes:
      - CSV schemas vary widely, so we use heuristics + optional column hints.
    """
    # how many characters to allow in key fields
    max_sender_chars: int = 300
    max_receiver_chars: int = 600
    max_subject_chars: int = 600
    max_date_chars: int = 160
    max_feedback_chars: int = 80_000

    # Column name hints (case-insensitive) you can extend later
    date_col_hints: Tuple[str, ...] = ("date", "submitted", "timestamp", "created", "time")
    sender_col_hints: Tuple[str, ...] = ("sender", "from", "name", "requestor", "submitter", "contact")
    receiver_col_hints: Tuple[str, ...] = ("receiver", "to", "agency", "office", "recipient")
    subject_col_hints: Tuple[str, ...] = ("subject", "topic", "title", "re", "regarding")
    feedback_col_hints: Tuple[str, ...] = ("feedback", "comment", "message", "text", "body", "description", "content")


# -----------------------------
# Patterns
# -----------------------------
RE_CQAS = re.compile(r"\bCQAS[-–—]?\s*(\d{3,10})\b", re.IGNORECASE)

RE_DATE_TEXT = re.compile(
    r"\b("
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}"
    r"|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r")\b",
    re.IGNORECASE,
)


# -----------------------------
# Helpers
# -----------------------------
def _norm_col(s: str) -> str:
    return (s or "").strip().lower()

def _pick_first_matching_col(columns: List[str], hints: Tuple[str, ...]) -> Optional[str]:
    """
    Choose the first column whose normalized name contains any hint token.
    """
    norm = {c: _norm_col(c) for c in columns}
    for hint in hints:
        h = hint.lower()
        for c, nc in norm.items():
            if h in nc:
                return c
    return None

def _safe_str(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def _all_cqas_ids(text: str) -> List[str]:
    out = []
    for m in RE_CQAS.finditer(text or ""):
        num = (m.group(1) or "").strip()
        if num:
            out.append(f"CQAS-{num}")
    # dedupe preserve order
    seen, deduped = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped

def _best_date_from_row(row: pd.Series, date_col: Optional[str]) -> Optional[str]:
    # 1) explicit date column
    if date_col and date_col in row.index:
        v = _safe_str(row[date_col])
        if v:
            return v
    # 2) scan whole row text
    joined = " ".join(_safe_str(v) for v in row.values)
    m = RE_DATE_TEXT.search(joined)
    return m.group(1).strip() if m else None

def _best_text_from_columns(
    row: pd.Series,
    preferred_col: Optional[str],
    fallback_cols: Optional[List[str]] = None,
) -> Optional[str]:
    if preferred_col and preferred_col in row.index:
        v = _safe_str(row[preferred_col])
        if v:
            return v
    if fallback_cols:
        for c in fallback_cols:
            if c in row.index:
                v = _safe_str(row[c])
                if v:
                    return v
    return None

def _make_feedback_text(row: pd.Series) -> str:
    # Prefer your row_text if present
    if "row_text" in row.index:
        rt = _safe_str(row["row_text"])
        if rt:
            return rt
    # Otherwise build a compact k=v string
    parts = []
    for k, v in row.items():
        if k in ("doc_id", "row_num", "error"):
            continue
        sv = _safe_str(v)
        if not sv:
            continue
        parts.append(f"{k}={sv}")
    return " | ".join(parts)


# -----------------------------
# Filename/title metadata patterns
# -----------------------------
RE_DATE_FILENAME = re.compile(
    r"(?P<mdy>\b\d{1,2}[ _/-]\d{1,2}[ _/-]\d{2,4}\b)"          # 03 28 2025, 03-28-2025, 03/28/25
    r"|(?P<compact>\b\d{8}\b)",                                # 12142021 (MMDDYYYY)
    re.IGNORECASE,
)

RE_FEEDBACK_TOKEN = re.compile(r"\bfeedback\b", re.IGNORECASE)


def _clean_title_for_parse(s: str) -> str:
    """
    Normalize a filename/title into a space-separated string.
    - drops file extension
    - replaces underscores with spaces
    - collapses whitespace
    """
    s = _safe_str(s)
    if not s:
        return ""
    # If it's a path/filename, drop extension and directories
    try:
        s = Path(s).name
        s = Path(s).stem
    except Exception:
        pass
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_date_from_title(title: str) -> Optional[str]:
    """
    Best-effort date extraction from filename/title.
    Returns a string; does not attempt strict normalization beyond compact MMDDYYYY.
    """
    t = _clean_title_for_parse(title)
    if not t:
        return None

    m = RE_DATE_FILENAME.search(t)
    if not m:
        return None

    if m.group("mdy"):
        # Keep as-is (normalized spacing) so downstream can normalize if needed
        return m.group("mdy").replace("_", " ").strip()

    if m.group("compact"):
        s = m.group("compact")
        # Assume MMDDYYYY
        mm, dd, yyyy = s[0:2], s[2:4], s[4:8]
        return f"{mm}/{dd}/{yyyy}"

    return None


def _remove_substring_case_insensitive(haystack: str, needle: str) -> str:
    if not haystack or not needle:
        return haystack
    return re.sub(re.escape(needle), "", haystack, flags=re.IGNORECASE).strip()


def _extract_sender_subject_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristics:
      - If "feedback" appears, text before it often resembles a sender/org name.
      - Remaining text becomes subject/event.
      - If no "feedback", treat entire title as subject.
    """
    t = _clean_title_for_parse(title)
    if not t:
        return None, None

    sender: Optional[str] = None
    subject: Optional[str] = None

    if RE_FEEDBACK_TOKEN.search(t):
        # Split once on "feedback" (case-insensitive)
        parts = RE_FEEDBACK_TOKEN.split(t, maxsplit=1)
        left = (parts[0] or "").strip(" -_")
        right = (parts[1] or "").strip(" -_")

        sender = left if left else None
        subject = right if right else None
    else:
        subject = t

    # Strip a detected date out of sender/subject (common in filenames)
    dt = _extract_date_from_title(t)
    if dt:
        if sender:
            sender = _remove_substring_case_insensitive(sender, dt) or sender
        if subject:
            subject = _remove_substring_case_insensitive(subject, dt) or subject

    # Light cleanup: collapse whitespace again
    if sender:
        sender = re.sub(r"\s+", " ", sender).strip()
        if len(sender) < 2:
            sender = None
    if subject:
        subject = re.sub(r"\s+", " ", subject).strip()
        if len(subject) < 2:
            subject = None

    return sender, subject


def _file_level_defaults_from_row(r: pd.Series) -> Dict[str, Optional[str]]:
    """
    Use source_file/source_rel_path (if present) to infer metadata defaults.
    Returns dict with keys: file_sender, file_subject, file_date
    """
    source = _safe_str(r.get("source_file")) or _safe_str(r.get("source_rel_path")) or ""
    if not source:
        return {"file_sender": None, "file_subject": None, "file_date": None}

    file_sender, file_subject = _extract_sender_subject_from_title(source)
    file_date = _extract_date_from_title(source)

    return {"file_sender": file_sender, "file_subject": file_subject, "file_date": file_date}



# -----------------------------
# Core parse
# -----------------------------
def parse_csv_rows_to_fields(
    rows_df: pd.DataFrame,
    documents_df: Optional[pd.DataFrame] = None,
    cfg: CsvParseConfig = CsvParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert ingested CSV rows into entry fields.

    Required columns in rows_df:
      - doc_id
      - row_num
      - (other columns vary; row_text optional)

    documents_df recommended columns:
      - doc_id, source_file, source_rel_path
    """
    required = {"doc_id", "row_num"}
    missing = required - set(rows_df.columns)
    if missing:
        raise ValueError(f"rows_df missing required cols: {sorted(missing)}")

    df = rows_df.copy()

    # Attach filename/path if available
    if documents_df is not None and "source_file" in documents_df.columns:
        df = df.merge(
            documents_df[["doc_id", "source_file", "source_rel_path"]].drop_duplicates("doc_id"),
            on="doc_id",
            how="left",
        )
    else:
        if "source_file" not in df.columns:
            df["source_file"] = None
        if "source_rel_path" not in df.columns:
            df["source_rel_path"] = None

    # Identify likely schema columns (per file, but we do it globally here)
    cols = [c for c in df.columns if c not in ("doc_id", "row_num", "error")]
    date_col = _pick_first_matching_col(cols, cfg.date_col_hints)
    sender_col = _pick_first_matching_col(cols, cfg.sender_col_hints)
    receiver_col = _pick_first_matching_col(cols, cfg.receiver_col_hints)
    subject_col = _pick_first_matching_col(cols, cfg.subject_col_hints)
    feedback_col = _pick_first_matching_col(cols, cfg.feedback_col_hints)

    wide_rows: List[Dict] = []
    long_rows: List[Dict] = []

    for _, r in df.iterrows():
        doc_id = r["doc_id"]
        row_num = int(r["row_num"]) if pd.notna(r["row_num"]) else 0

        # entry_num mirrors row_num (this is your “segmentation” for CSV)
        entry_num = row_num

        # apply metadata labels
        file_defaults = _file_level_defaults_from_row(r)

        # Build a big text blob for CQAS scan / fallbacks
        feedback_text = _safe_str(r.get("row_text")) or ""
        if not feedback_text:
            feedback_text = _make_feedback_text(r)

        cqas_ids = _all_cqas_ids(feedback_text)
        cqas_id = cqas_ids[0] if cqas_ids else None

        # Best-effort fields
        date = _best_date_from_row(r, date_col)
        sender = _best_text_from_columns(r, sender_col)
        receiver = _best_text_from_columns(r, receiver_col)
        subject = _best_text_from_columns(r, subject_col)
        # fallback to meta data 
        if not sender:
            sender = file_defaults.get("file_sender")
        if not subject:
            subject = file_defaults.get("file_subject")
        if not date:
            date = file_defaults.get("file_date")

        # If there is a dedicated feedback/comment column, prefer that as feedback_text
        fb = _best_text_from_columns(r, feedback_col)
        if fb:
            feedback_text = fb

        # entry_type for CSV is usually “tabular_row”
        entry_type = "tabular_row"

        # caps
        if sender:
            sender = sender[: cfg.max_sender_chars]
        if receiver:
            receiver = receiver[: cfg.max_receiver_chars]
        if subject:
            subject = subject[: cfg.max_subject_chars]
        if date:
            date = date[: cfg.max_date_chars]
        if feedback_text:
            feedback_text = feedback_text[: cfg.max_feedback_chars]

        wide = {
            "doc_id": doc_id,
            "entry_num": entry_num,
            "row_num": row_num,
            "source_file": r.get("source_file"),
            "source_rel_path": r.get("source_rel_path"),
            "file_sender": file_defaults.get("file_sender"),
            "file_subject": file_defaults.get("file_subject"),
            "file_date": file_defaults.get("file_date"),


            "entry_type": entry_type,

            "cqas_id": cqas_id,
            "cqas_ids_json": json.dumps(cqas_ids, ensure_ascii=False),

            "sender": sender,
            "receiver": receiver,
            "date": date,
            "subject": subject,
        

            "feedback_text": feedback_text,
            "raw_row_json": json.dumps(
                {k: (None if pd.isna(v) else v) for k, v in r.items()},
                ensure_ascii=False,
                default=str,
            ),
            "error": r.get("error"),
        }
        wide_rows.append(wide)

        def emit(field_name: str, field_value: Optional[str]) -> None:
            long_rows.append(
                {"doc_id": doc_id, "entry_num": entry_num, "field_name": field_name, "field_value": field_value}
            )

        emit("entry_type", entry_type)
        emit("cqas_id", cqas_id)
        emit("sender", sender)
        emit("receiver", receiver)
        emit("date", date)
        emit("subject", subject)
        emit("feedback_text", feedback_text)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)

    if not wide_df.empty:
        wide_df = wide_df.sort_values(["doc_id", "entry_num"]).reset_index(drop=True)
    if not long_df.empty:
        long_df = long_df.sort_values(["doc_id", "entry_num", "field_name"]).reset_index(drop=True)

    return wide_df, long_df


# -----------------------------
# CLI
# -----------------------------
def main_cli() -> None:
    """
    Usage:
      conda run -n feedback python -m Caste_Project.parse.parse_csv --in_dir data\\_test_output --out_dir data\\_parse_output --docs_dir data\\_test_output

    Expects:
      in_dir/csv_rows.parquet
      docs_dir/csv_documents.parquet (recommended)
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows_name", default="csv_entries.parquet")
    ap.add_argument("--docs_dir", required=False, help="Folder containing csv_documents.parquet (recommended)")
    ap.add_argument("--documents_name", default="csv_documents.parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = in_dir / args.rows_name
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing {rows_path}. Pass --rows_name if different.")

    rows_df = pd.read_parquet(rows_path)

    documents_df = None
    if args.docs_dir:
        docs_path = Path(args.docs_dir) / args.documents_name
        if docs_path.exists():
            documents_df = pd.read_parquet(docs_path)

    wide_df, long_df = parse_csv_rows_to_fields(rows_df, documents_df, CsvParseConfig())

    wide_df.to_parquet(out_dir / "csv_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "csv_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'csv_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'csv_entry_fields_long.parquet'}")

    cols = ["doc_id", "entry_num", "sender", "date", "subject", "cqas_id"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
