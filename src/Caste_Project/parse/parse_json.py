# src/Caste_Project/parse/parse_json.py
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
class JsonParseConfig:
    """
    Parsing goal for JSON:
      - One record/entry = one "entry" (from json_entries.parquet)
      - Extract best-effort fields (date/sender/receiver/subject/cqas)
      - Provide feedback_text for downstream NLP

    Notes:
      - JSON schemas vary widely; we use:
        1) key-hint search on parsed JSON (if available),
        2) anchored string search inside entry_text,
        3) filename/title-derived defaults (sender/subject/date),
        4) final fallbacks.
    """
    max_sender_chars: int = 300
    max_receiver_chars: int = 600
    max_subject_chars: int = 800
    max_date_chars: int = 200
    max_feedback_chars: int = 120_000

    # key name hints in JSON (case-insensitive; substring match)
    date_key_hints: Tuple[str, ...] = ("date", "submitted", "timestamp", "created", "time")
    sender_key_hints: Tuple[str, ...] = ("sender", "from", "name", "requestor", "submitter", "contact", "author")
    receiver_key_hints: Tuple[str, ...] = ("receiver", "to", "agency", "office", "recipient")
    subject_key_hints: Tuple[str, ...] = ("subject", "topic", "title", "regarding", "re", "theme")
    feedback_key_hints: Tuple[str, ...] = ("feedback", "comment", "message", "text", "body", "description", "content", "notes")

    # If feedback_key_hints fail, include entry_text as feedback_text regardless.
    always_use_entry_text_as_feedback: bool = True


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

# anchored lines inside entry_text (some JSON record_text may be a blob)
RE_FROM = re.compile(r"^\s*From\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_TO = re.compile(r"^\s*To\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_SUBJECT = re.compile(r"^\s*Subject\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_DATE_ANCHOR = re.compile(r"^\s*Date\s*:\s*(.+)\s*$", re.IGNORECASE)

# filename/title date patterns (file-level defaults)
RE_DATE_FILENAME = re.compile(
    r"(?P<mdy>\b\d{1,2}[ _/-]\d{1,2}[ _/-]\d{2,4}\b)"  # 03 28 2025, 03-28-2025, 03/28/25
    r"|(?P<compact>\b\d{8}\b)",  # 12142021 (MMDDYYYY)
    re.IGNORECASE,
)
RE_FEEDBACK_TOKEN = re.compile(r"\bfeedback\b", re.IGNORECASE)


# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    return str(x).strip()


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _all_cqas_ids(text: str) -> List[str]:
    out: List[str] = []
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


def _clean_title_for_parse(s: str) -> str:
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


def _best_date_from_text(text: str) -> Optional[str]:
    m = RE_DATE_TEXT.search(text or "")
    return m.group(1).strip() if m else None


def _extract_anchored_fields(entry_text: str) -> Dict[str, Optional[str]]:
    sender = receiver = subject = date = None
    for raw in (entry_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = RE_FROM.match(line)
        if m:
            sender = (m.group(1) or "").strip() or sender
            continue
        m = RE_TO.match(line)
        if m:
            receiver = (m.group(1) or "").strip() or receiver
            continue
        m = RE_SUBJECT.match(line)
        if m:
            subject = (m.group(1) or "").strip() or subject
            continue
        m = RE_DATE_ANCHOR.match(line)
        if m:
            date = (m.group(1) or "").strip() or date
            continue
    return {"sender": sender, "receiver": receiver, "subject": subject, "date": date}


def _safe_json_loads(s: str) -> Optional[object]:
    s = _safe_str(s)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _walk_kv(obj: object, prefix: str = "", max_nodes: int = 2000) -> List[Tuple[str, object]]:
    """
    Flatten JSON into (path, value) pairs (bounded).
    """
    out: List[Tuple[str, object]] = []
    stack: List[Tuple[str, object]] = [(prefix or "$", obj)]
    seen = 0

    while stack:
        p, cur = stack.pop()
        seen += 1
        if seen > max_nodes:
            break

        if isinstance(cur, dict):
            for k, v in cur.items():
                kp = f"{p}.{k}"
                stack.append((kp, v))
        elif isinstance(cur, list):
            for i, v in enumerate(cur[:200]):  # bound list expansion
                kp = f"{p}[{i}]"
                stack.append((kp, v))
        else:
            out.append((p, cur))
    return out


def _find_value_by_key_hints(obj: object, hints: Tuple[str, ...]) -> Optional[str]:
    """
    Search flattened (path, value) for first match where the final key contains any hint.
    """
    if obj is None:
        return None
    pairs = _walk_kv(obj)
    for hint in hints:
        h = hint.lower()
        for path, val in pairs:
            # last token of dict path: $.a.b.c -> c
            last = path.split(".")[-1]
            # remove array brackets if present: field[0] -> field
            last = re.sub(r"\[\d+\]$", "", last)
            if h in last.lower():
                sv = _safe_str(val)
                if sv:
                    return sv
    return None


def _make_feedback_text(entry_text: str, entry_json: Optional[str]) -> str:
    """
    Preference order:
      1) entry_text if present
      2) entry_json string if present
      3) empty string
    """
    et = _safe_str(entry_text)
    if et:
        return et
    ej = _safe_str(entry_json)
    if ej:
        return ej
    return ""


# -----------------------------
# Core parse
# -----------------------------
def parse_json_entries_to_fields(
    entries_df: pd.DataFrame,
    documents_df: Optional[pd.DataFrame] = None,
    cfg: JsonParseConfig = JsonParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert JSON entries into entry fields.

    Required columns in entries_df:
      - doc_id
      - entry_num
      - entry_text (recommended) OR entry_json
    """
    required = {"doc_id", "entry_num"}
    missing = required - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries_df missing required cols: {sorted(missing)}")

    df = entries_df.copy()

    if "entry_text" not in df.columns and "entry_json" not in df.columns:
        raise ValueError("entries_df must contain at least one of: entry_text, entry_json")

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

    for _, r in df.iterrows():
        doc_id = r["doc_id"]
        entry_num = int(r["entry_num"]) if pd.notna(r["entry_num"]) else 0

        entry_text = _safe_str(r.get("entry_text"))
        entry_json_str = _safe_str(r.get("entry_json"))

        file_defaults = _file_level_defaults_from_row(r)

        # 1) Try parse JSON and pick hinted keys
        obj = _safe_json_loads(entry_json_str) if entry_json_str else None

        sender = _find_value_by_key_hints(obj, cfg.sender_key_hints)
        receiver = _find_value_by_key_hints(obj, cfg.receiver_key_hints)
        subject = _find_value_by_key_hints(obj, cfg.subject_key_hints)
        date = _find_value_by_key_hints(obj, cfg.date_key_hints)

        feedback_from_json = _find_value_by_key_hints(obj, cfg.feedback_key_hints)

        # 2) Anchored fields in entry_text (overrides JSON hints if present)
        anchored = _extract_anchored_fields(entry_text)
        sender = anchored.get("sender") or sender
        receiver = anchored.get("receiver") or receiver
        subject = anchored.get("subject") or subject
        date = anchored.get("date") or date

        # 3) CQAS extraction (scan both)
        cqas_ids = _all_cqas_ids((entry_text or "") + "\n" + (entry_json_str or ""))
        cqas_id = cqas_ids[0] if cqas_ids else None

        # 4) Date scan fallback
        if not date:
            date = _best_date_from_text(entry_text) or _best_date_from_text(entry_json_str)

        # 5) Filename/title fallbacks
        if not sender:
            sender = file_defaults.get("file_sender")
        if not subject:
            subject = file_defaults.get("file_subject")
        if not date:
            date = file_defaults.get("file_date")

        # feedback_text
        feedback_text = ""
        if feedback_from_json:
            # keep both if entry_text adds additional context
            if cfg.always_use_entry_text_as_feedback and entry_text and entry_text not in feedback_from_json:
                feedback_text = feedback_from_json + "\n\n" + entry_text
            else:
                feedback_text = feedback_from_json
        else:
            feedback_text = _make_feedback_text(entry_text, entry_json_str)

        entry_type = "json_record"

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

            # optional passthroughs from segmenter
            "json_path": r.get("json_path"),
            "record_type": r.get("record_type"),
            "seg_method": r.get("seg_method"),
            "seg_confidence": r.get("seg_confidence"),
            "flags_json": r.get("flags_json"),

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

            # keep raw json/text for traceability
            "raw_entry_text": entry_text if entry_text else None,
            "raw_entry_json": entry_json_str if entry_json_str else None,

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
      conda run -n feedback python -m Caste_Project.parse.parse_json --in_dir data\\_seg_output --out_dir data\\_parse_output --docs_dir data\\_test_output

    Expects:
      in_dir/json_entries.parquet
      docs_dir/json_documents.parquet (recommended)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--entries_name", default="json_entries.parquet")
    ap.add_argument("--docs_dir", required=False, help="Folder containing json_documents.parquet (recommended)")
    ap.add_argument("--documents_name", default="json_documents.parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries_path = in_dir / args.entries_name
    if not entries_path.exists():
        raise FileNotFoundError(f"Missing {entries_path}. Pass --entries_name if different.")

    entries_df = pd.read_parquet(entries_path)

    documents_df = None
    if args.docs_dir:
        docs_path = Path(args.docs_dir) / args.documents_name
        if docs_path.exists():
            documents_df = pd.read_parquet(docs_path)

    wide_df, long_df = parse_json_entries_to_fields(entries_df, documents_df, JsonParseConfig())

    wide_df.to_parquet(out_dir / "json_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "json_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'json_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'json_entry_fields_long.parquet'}")

    cols = ["doc_id", "entry_num", "sender", "date", "subject", "cqas_id"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
