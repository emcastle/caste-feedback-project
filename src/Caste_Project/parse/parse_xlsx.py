# src/Caste_Project/parse/parse_xlsx.py
from __future__ import annotations

import argparse
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
class XlsxParseConfig:
    """
    Parsing goal for XLSX:
      - One Excel *row* = one "entry"
      - Extract best-effort fields (date/sender/receiver/subject/cqas)
      - Provide feedback_text for downstream NLP

    Notes:
      - Excel schemas vary widely, so we use heuristics + optional column hints.
      - Works with either excel_rows.parquet (from ingest) or excel_entries.parquet (from segment).
    """
    # how many characters to allow in key fields
    max_sender_chars: int = 300
    max_receiver_chars: int = 600
    max_subject_chars: int = 600
    max_date_chars: int = 160
    max_feedback_chars: int = 80_000

    # Column name hints (case-insensitive) can extend later
    date_col_hints: Tuple[str, ...] = ("date", "submitted", "timestamp", "created", "time")
    sender_col_hints: Tuple[str, ...] = ("sender", "from", "name", "requestor", "submitter", "contact")
    receiver_col_hints: Tuple[str, ...] = ("receiver", "to", "agency", "office", "recipient")
    subject_col_hints: Tuple[str, ...] = ("subject", "topic", "title", "re", "regarding")
    feedback_col_hints: Tuple[str, ...] = (
        "feedback",
        "comment",
        "message",
        "text",
        "body",
        "description",
        "content",
        "notes",
        "why",
        "criteria",
        "use case",
        "detail",
    )

    # When no obvious feedback column exists, choose the "most text-like" column by length
    enable_best_text_col_fallback: bool = True


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

# filename/title date patterns (for file-level defaults)
RE_DATE_FILENAME = re.compile(
    r"(?P<mdy>\b\d{1,2}[ _/-]\d{1,2}[ _/-]\d{2,4}\b)"          # 03 28 2025, 03-28-2025, 03/28/25
    r"|(?P<compact>\b\d{8}\b)",                                # 12142021 (MMDDYYYY)
    re.IGNORECASE,
)
RE_FEEDBACK_TOKEN = re.compile(r"\bfeedback\b", re.IGNORECASE)


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
    if "entry_text" in row.index:
        et = _safe_str(row["entry_text"])
        if et:
            return et

    # Otherwise build a compact k=v string
    parts = []
    for k, v in row.items():
        if k in ("doc_id", "row_num", "entry_num", "error"):
            continue
        sv = _safe_str(v)
        if not sv:
            continue
        parts.append(f"{k}={sv}")
    return " | ".join(parts)


def _pick_best_text_col_by_length(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for c in df.columns:
        if c in exclude:
            continue
        # object/string-ish columns only
        if df[c].dtype == "object":
            lens = df[c].dropna().astype(str).map(len)
            if len(lens) == 0:
                continue
            candidates.append((float(lens.mean()), c))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


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
    try:
        s = Path(s).name
        s = Path(s).stem
    except Exception:
        pass
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_date_from_title(title: str) -> Optional[str]:
    t = _clean_title_for_parse(title)
    if not t:
        return None
    m = RE_DATE_FILENAME.search(t)
    if not m:
        return None
    if m.group("mdy"):
        return m.group("mdy").replace("_", " ").strip()
    if m.group("compact"):
        s = m.group("compact")
        mm, dd, yyyy = s[0:2], s[2:4], s[4:8]
        return f"{mm}/{dd}/{yyyy}"
    return None


def _remove_substring_case_insensitive(haystack: str, needle: str) -> str:
    if not haystack or not needle:
        return haystack
    return re.sub(re.escape(needle), "", haystack, flags=re.IGNORECASE).strip()


def _extract_sender_subject_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    t = _clean_title_for_parse(title)
    if not t:
        return None, None

    sender: Optional[str] = None
    subject: Optional[str] = None

    if RE_FEEDBACK_TOKEN.search(t):
        parts = RE_FEEDBACK_TOKEN.split(t, maxsplit=1)
        left = (parts[0] or "").strip(" -_")
        right = (parts[1] or "").strip(" -_")
        sender = left if left else None
        subject = right if right else None
    else:
        subject = t

    dt = _extract_date_from_title(t)
    if dt:
        if sender:
            sender = _remove_substring_case_insensitive(sender, dt) or sender
        if subject:
            subject = _remove_substring_case_insensitive(subject, dt) or subject

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
    source = _safe_str(r.get("source_file")) or _safe_str(r.get("source_rel_path")) or ""
    if not source:
        return {"file_sender": None, "file_subject": None, "file_date": None}
    file_sender, file_subject = _extract_sender_subject_from_title(source)
    file_date = _extract_date_from_title(source)
    return {"file_sender": file_sender, "file_subject": file_subject, "file_date": file_date}


def _ensure_row_num(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize row index field:
      - If row_num exists, keep it.
      - Else if entry_num exists, alias it to row_num.
      - Else create row_num as cumcount per doc_id.
    """
    out = df.copy()
    if "row_num" in out.columns:
        return out
    if "entry_num" in out.columns:
        out["row_num"] = pd.to_numeric(out["entry_num"], errors="coerce").fillna(0).astype(int)
        return out
    if "doc_id" in out.columns:
        out = out.sort_values(["doc_id"]).reset_index(drop=True)
        out["row_num"] = out.groupby("doc_id").cumcount()
        return out
    out["row_num"] = range(len(out))
    return out


# -----------------------------
# Core parse
# -----------------------------
def parse_xlsx_rows_to_fields(
    rows_df: pd.DataFrame,
    documents_df: Optional[pd.DataFrame] = None,
    cfg: XlsxParseConfig = XlsxParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert ingested XLSX rows into entry fields.

    Required columns in rows_df:
      - doc_id
      - row_num   (or entry_num, which will be aliased)

    documents_df recommended columns:
      - doc_id, source_file, source_rel_path
    """
    required = {"doc_id"}
    missing = required - set(rows_df.columns)
    if missing:
        raise ValueError(f"rows_df missing required cols: {sorted(missing)}")

    df = _ensure_row_num(rows_df)

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

    # Identify schema columns per doc_id (XLSX is highly heterogeneous)
    wide_rows: List[Dict] = []
    long_rows: List[Dict] = []

    for doc_id, g in df.groupby("doc_id", sort=False):
        # exclude meta-ish columns
        cols = [c for c in g.columns if c not in ("doc_id", "row_num", "entry_num", "error", "source_file", "source_rel_path")]
        date_col = _pick_first_matching_col(cols, cfg.date_col_hints)
        sender_col = _pick_first_matching_col(cols, cfg.sender_col_hints)
        receiver_col = _pick_first_matching_col(cols, cfg.receiver_col_hints)
        subject_col = _pick_first_matching_col(cols, cfg.subject_col_hints)
        feedback_col = _pick_first_matching_col(cols, cfg.feedback_col_hints)

        if cfg.enable_best_text_col_fallback and feedback_col is None:
            feedback_col = _pick_best_text_col_by_length(
                g,
                exclude=set(["doc_id", "row_num", "entry_num", "error", "source_file", "source_rel_path"]),
            )

        for _, r in g.iterrows():
            row_num = int(r["row_num"]) if pd.notna(r["row_num"]) else 0
            entry_num = row_num

            file_defaults = _file_level_defaults_from_row(r)

            # Build a big text blob for CQAS scan / fallbacks
            feedback_text = _safe_str(r.get("row_text")) or _safe_str(r.get("entry_text")) or ""
            if not feedback_text:
                feedback_text = _make_feedback_text(r)

            cqas_ids = _all_cqas_ids(feedback_text)
            cqas_id = cqas_ids[0] if cqas_ids else None

            # Best-effort fields
            date = _best_date_from_row(r, date_col)
            sender = _best_text_from_columns(r, sender_col)
            receiver = _best_text_from_columns(r, receiver_col)
            subject = _best_text_from_columns(r, subject_col)

            # If there is a dedicated feedback/comment column, prefer that as feedback_text
            fb = _best_text_from_columns(r, feedback_col)
            if fb:
                structured = _make_feedback_text(r)
                # keep structured tail so we don't lose important context fields
                if structured and structured not in fb:
                    feedback_text = fb + " | " + structured
                else:
                    feedback_text = fb

            # Fallback to filename/title-derived defaults if row-level fields are missing
            if not sender:
                sender = file_defaults.get("file_sender")
            if not subject:
                subject = file_defaults.get("file_subject")
            if not date:
                date = file_defaults.get("file_date")

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

                # file-level inferred metadata (audit columns)
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

def parse_xlsx_sheets_to_fields(
    rows_df: pd.DataFrame,
    documents_df: Optional[pd.DataFrame] = None,
    cfg: XlsxParseConfig = XlsxParseConfig(),
    entry_type: str = "document",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Document-level parse for XLSX:
      - One (doc_id, sheet_name) => one entry (entry_num=0 within that sheet group)
      - feedback_text is concatenation of row text in row_num order
    """
    required = {"doc_id"}
    missing = required - set(rows_df.columns)
    if missing:
        raise ValueError(f"rows_df missing required cols: {sorted(missing)}")

    df = _ensure_row_num(rows_df).copy()

    # Ensure sheet_name exists (ingest should provide it; if not, treat as single sheet)
    if "sheet_name" not in df.columns:
        df["sheet_name"] = None

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

    wide_rows: List[Dict] = []
    long_rows: List[Dict] = []

    group_cols = ["doc_id", "sheet_name"]

    for (doc_id, sheet_name), g in df.groupby(group_cols, sort=False, dropna=False):
        g = g.sort_values("row_num", kind="stable")
        r0 = g.iloc[0]
        file_defaults = _file_level_defaults_from_row(r0)

        # Determine hint columns within this sheet (more accurate than global)
        cols = [c for c in g.columns if c not in (
            "doc_id", "sheet_name", "row_num", "entry_num", "error", "source_file", "source_rel_path"
        )]
        date_col = _pick_first_matching_col(cols, cfg.date_col_hints)
        sender_col = _pick_first_matching_col(cols, cfg.sender_col_hints)
        receiver_col = _pick_first_matching_col(cols, cfg.receiver_col_hints)
        subject_col = _pick_first_matching_col(cols, cfg.subject_col_hints)
        feedback_col = _pick_first_matching_col(cols, cfg.feedback_col_hints)

        if cfg.enable_best_text_col_fallback and feedback_col is None:
            feedback_col = _pick_best_text_col_by_length(
                g,
                exclude=set(["doc_id", "sheet_name", "row_num", "entry_num", "error", "source_file", "source_rel_path"]),
            )

        # Build document text by concatenating all rows
        texts: List[str] = []
        for _, r in g.iterrows():
            fb = _best_text_from_columns(r, feedback_col)
            if fb:
                texts.append(fb)
            else:
                rt = _safe_str(r.get("row_text")) or _safe_str(r.get("entry_text"))
                texts.append(rt if rt else _make_feedback_text(r))

        feedback_text = "\n".join([t for t in texts if t.strip()])
        feedback_text = feedback_text[: cfg.max_feedback_chars] if feedback_text else ""

        cqas_ids = _all_cqas_ids(feedback_text)
        cqas_id = cqas_ids[0] if cqas_ids else None

        # Best-effort metadata: first non-empty in hinted columns across rows
        sender: Optional[str] = None
        receiver: Optional[str] = None
        subject: Optional[str] = None
        date: Optional[str] = None

        if sender_col and sender_col in g.columns:
            s = g[sender_col].fillna("").astype(str).str.strip()
            sender = s[s != ""].iloc[0] if (s != "").any() else None
        if receiver_col and receiver_col in g.columns:
            s = g[receiver_col].fillna("").astype(str).str.strip()
            receiver = s[s != ""].iloc[0] if (s != "").any() else None
        if subject_col and subject_col in g.columns:
            s = g[subject_col].fillna("").astype(str).str.strip()
            subject = s[s != ""].iloc[0] if (s != "").any() else None
        if date_col and date_col in g.columns:
            s = g[date_col].fillna("").astype(str).str.strip()
            date = s[s != ""].iloc[0] if (s != "").any() else None

        # Filename defaults
        if not sender:
            sender = file_defaults.get("file_sender")
        if not subject:
            subject = file_defaults.get("file_subject")
        if not date:
            date = file_defaults.get("file_date")

        # Last resort: scan doc text for date
        if not date and feedback_text:
            m = RE_DATE_TEXT.search(feedback_text)
            if m:
                date = m.group(1).strip()

        # caps
        if sender:
            sender = str(sender)[: cfg.max_sender_chars]
        if receiver:
            receiver = str(receiver)[: cfg.max_receiver_chars]
        if subject:
            subject = str(subject)[: cfg.max_subject_chars]
        if date:
            date = str(date)[: cfg.max_date_chars]

        wide = {
            "doc_id": doc_id,
            "sheet_name": sheet_name,
            "entry_num": 0,
            "row_num": 0,
            "source_file": r0.get("source_file"),
            "source_rel_path": r0.get("source_rel_path"),
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
            "raw_rows_json": json.dumps(
                [{k: (None if pd.isna(v) else v) for k, v in r.items()} for _, r in g.iterrows()],
                ensure_ascii=False,
                default=str,
            ),
            "error": None,
        }
        wide_rows.append(wide)

        def emit(field_name: str, field_value: Optional[str]) -> None:
            long_rows.append(
                {
                    "doc_id": doc_id,
                    "sheet_name": sheet_name,
                    "entry_num": 0,
                    "field_name": field_name,
                    "field_value": field_value,
                }
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
        wide_df = wide_df.sort_values(["doc_id", "sheet_name", "entry_num"]).reset_index(drop=True)
    if not long_df.empty:
        long_df = long_df.sort_values(["doc_id", "sheet_name", "entry_num", "field_name"]).reset_index(drop=True)

    return wide_df, long_df


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _has_any_col(columns: List[str], tokens: List[str]) -> bool:
    cols = [_norm(c) for c in columns]
    for t in tokens:
        tt = t.lower()
        if any(tt in c for c in cols):
            return True
    return False


def _nunique_nonempty(series: pd.Series) -> int:
    s = series.fillna("").astype(str).str.strip()
    s = s[s != ""]
    return int(s.nunique()) if not s.empty else 0


def _should_parse_sheet_as_rows(df: pd.DataFrame, cfg: XlsxParseConfig) -> bool:
    cols = list(df.columns)

    # Marker columns
    has_cqas = _has_any_col(cols, ["cqas", "cqas number"])
    has_submitter_identity = (
        _has_any_col(cols, ["submitter first", "submitter last", "submitter", "sender"])
        and _has_any_col(cols, ["organization", "email", "name", "contact"])
    )

    if not (has_cqas or has_submitter_identity):
        return False

    # Evidence of multiple records
    if has_cqas:
        cqas_col = next((c for c in cols if "cqas" in _norm(c)), None)
        if cqas_col is not None and _nunique_nonempty(df[cqas_col]) >= 3:
            return True

    email_col = next((c for c in cols if "email" in _norm(c)), None)
    if email_col is not None and _nunique_nonempty(df[email_col]) >= 3:
        return True

    fb_col = _pick_first_matching_col(cols, cfg.feedback_col_hints)
    if fb_col and fb_col in df.columns:
        nonempty_fb = (df[fb_col].fillna("").astype(str).str.strip() != "").sum()
        if nonempty_fb >= 5:
            return True

    return False


def _is_likely_summary_sheet(df: pd.DataFrame, cfg: XlsxParseConfig) -> bool:
    cols = list(df.columns)

    # Guardrail: narrative text anywhere => not summary
    for v in df.fillna("").astype(str).to_numpy().ravel():
        s = str(v).strip()
        if len(s) >= 200:
            return False

    # If a feedback-like column exists and has any content => not summary
    fb_col = _pick_first_matching_col(cols, cfg.feedback_col_hints)
    if fb_col and fb_col in df.columns:
        nonempty_fb = (df[fb_col].fillna("").astype(str).str.strip() != "").sum()
        if nonempty_fb >= 1:
            return False

    agg_header = _has_any_col(cols, ["count", "total", "percent", "%", "frequency", "sum", "avg", "mean"])

    tmp = df.fillna("")
    numeric_like = 0
    nonempty = 0
    lengths: List[int] = []

    for v in tmp.to_numpy().ravel():
        s = str(v).strip()
        if not s:
            continue
        nonempty += 1
        lengths.append(len(s))
        try:
            float(s.replace(",", ""))
            numeric_like += 1
        except Exception:
            pass

    pct_numeric = (numeric_like / nonempty) if nonempty else 0.0
    med_len = sorted(lengths)[len(lengths) // 2] if lengths else 0

    if agg_header and not fb_col:
        return True

    if nonempty >= 50 and pct_numeric > 0.70 and med_len < 25:
        return True

    return False




# -----------------------------
# CLI
# -----------------------------
def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows_name", default="excel_rows.parquet")
    ap.add_argument("--docs_dir", required=False, help="Folder containing excel_documents.parquet (recommended)")
    ap.add_argument("--documents_name", default="excel_documents.parquet")

    ap.add_argument("--granularity", choices=["row", "document", "auto"], default="auto")
    ap.add_argument("--keep_summaries", action="store_true")

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

    cfg = XlsxParseConfig()

    if args.granularity == "row":
        wide_df, long_df = parse_xlsx_rows_to_fields(rows_df, documents_df, cfg)

    elif args.granularity == "document":
        # one entry per (doc_id, sheet_name)
        wide_df, long_df = parse_xlsx_sheets_to_fields(rows_df, documents_df, cfg, entry_type="document")

    else:
        # AUTO: decide per (doc_id, sheet_name)
        df = _ensure_row_num(rows_df).copy()
        if "sheet_name" not in df.columns:
            df["sheet_name"] = None

        wide_parts: List[pd.DataFrame] = []
        long_parts: List[pd.DataFrame] = []

        for (doc_id, sheet_name), g in df.groupby(["doc_id", "sheet_name"], sort=False, dropna=False):
            g = g.copy()

            # 1) summary?
            if _is_likely_summary_sheet(g, cfg):
                if not args.keep_summaries:
                    continue
                w, l = parse_xlsx_sheets_to_fields(g, documents_df, cfg, entry_type="summary_sheet")
                if not w.empty:
                    w["parse_reason"] = "summary_detected"
                if not l.empty:
                    l.loc[l["field_name"] == "entry_type", "field_value"] = "summary_sheet"
                wide_parts.append(w)
                long_parts.append(l)
                continue

            # 2) multi-entry?
            if _should_parse_sheet_as_rows(g, cfg):
                w, l = parse_xlsx_rows_to_fields(g, documents_df, cfg)
                if not w.empty:
                    w["parse_reason"] = "multi_entry_markers"
                wide_parts.append(w)
                long_parts.append(l)
            else:
                # 3) default document
                w, l = parse_xlsx_sheets_to_fields(g, documents_df, cfg, entry_type="document")
                if not w.empty:
                    w["parse_reason"] = "default_document"
                wide_parts.append(w)
                long_parts.append(l)

        wide_df = pd.concat(wide_parts, ignore_index=True) if wide_parts else pd.DataFrame()
        long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    wide_df.to_parquet(out_dir / "excel_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "excel_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'excel_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'excel_entry_fields_long.parquet'}")

    cols = ["doc_id", "sheet_name", "entry_num", "entry_type", "parse_reason", "sender", "date", "subject", "cqas_id"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    if not wide_df.empty and cols:
        print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
