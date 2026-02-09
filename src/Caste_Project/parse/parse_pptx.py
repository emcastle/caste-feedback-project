# src/Caste_Project/parse/parse_pptx.py
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
class PptxParseConfig:
    """
    Parsing goal for PPTX:
      - One "entry" = one slide (from pptx_entries.parquet)
      - Extract best-effort fields (date/sender/receiver/subject/cqas)
      - Provide feedback_text for downstream NLP

    Inputs:
      - pptx_entries.parquet (recommended, output of segment_pptx.py)
      - pptx_documents.parquet (recommended for file metadata)

    Outputs:
      - pptx_entry_fields.parquet (wide)
      - pptx_entry_fields_long.parquet (long)
    """
    max_sender_chars: int = 300
    max_receiver_chars: int = 600
    max_subject_chars: int = 800
    max_date_chars: int = 200
    max_feedback_chars: int = 120_000

    # When we find anchored fields like "From:" or "Subject:", optionally remove
    # those lines from feedback_text to avoid duplicating metadata.
    strip_anchor_lines_from_feedback: bool = False


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

# Slide-internal field anchors (common in feedback decks)
RE_FROM = re.compile(r"^\s*From\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_TO = re.compile(r"^\s*To\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_SUBJECT = re.compile(r"^\s*Subject\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_DATE_ANCHOR = re.compile(r"^\s*Date\s*:\s*(.+)\s*$", re.IGNORECASE)
RE_REQUESTOR = re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*(.+)\s*$", re.IGNORECASE)

# filename/title date patterns (for file-level defaults)
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
        # assume MMDDYYYY
        mm, dd, yyyy = s[0:2], s[2:4], s[4:8]
        return f"{mm}/{dd}/{yyyy}"
    return None


def _remove_substring_case_insensitive(haystack: str, needle: str) -> str:
    if not haystack or not needle:
        return haystack
    return re.sub(re.escape(needle), "", haystack, flags=re.IGNORECASE).strip()


def _extract_sender_subject_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic:
      - If "feedback" appears, text before it often resembles sender/org.
      - Remaining text becomes subject/event.
      - If no "feedback", treat entire title as subject.
    """
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

    # strip date out if present
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
    """
    Extracts simple "Label: value" lines from the slide text.
    Returns: sender, receiver, subject, date, requestor_name, and a set of consumed lines.
    """
    sender = receiver = subject = date = requestor_name = None
    consumed_lines: List[str] = []

    lines = (entry_text or "").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = RE_FROM.match(line)
        if m:
            sender = (m.group(1) or "").strip() or sender
            consumed_lines.append(raw)
            continue

        m = RE_TO.match(line)
        if m:
            receiver = (m.group(1) or "").strip() or receiver
            consumed_lines.append(raw)
            continue

        m = RE_SUBJECT.match(line)
        if m:
            subject = (m.group(1) or "").strip() or subject
            consumed_lines.append(raw)
            continue

        m = RE_DATE_ANCHOR.match(line)
        if m:
            date = (m.group(1) or "").strip() or date
            consumed_lines.append(raw)
            continue

        m = RE_REQUESTOR.match(line)
        if m:
            requestor_name = (m.group(1) or "").strip() or requestor_name
            consumed_lines.append(raw)
            continue

    return {
        "sender": sender,
        "receiver": receiver,
        "subject": subject,
        "date": date,
        "requestor_name": requestor_name,
        "consumed_lines": consumed_lines,
    }


def _strip_lines(text: str, lines_to_strip: List[str]) -> str:
    if not text or not lines_to_strip:
        return text or ""
    # Remove exact raw lines (preserving other formatting)
    # Do it line-by-line to avoid accidental substring removals.
    strip_set = set(lines_to_strip)
    kept = []
    for raw in (text or "").splitlines():
        if raw in strip_set:
            continue
        kept.append(raw)
    return "\n".join(kept).strip()


# -----------------------------
# Core parse
# -----------------------------
def parse_pptx_entries_to_fields(
    entries_df: pd.DataFrame,
    documents_df: Optional[pd.DataFrame] = None,
    cfg: PptxParseConfig = PptxParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert PPTX slide entries into entry fields.

    Required columns in entries_df:
      - doc_id
      - entry_num
      - entry_text

    Recommended:
      - slide_num, start_block, end_block, seg_method, seg_confidence

    documents_df recommended columns:
      - doc_id, source_file, source_rel_path
    """
    required = {"doc_id", "entry_num", "entry_text"}
    missing = required - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries_df missing required cols: {sorted(missing)}")

    df = entries_df.copy()

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

        file_defaults = _file_level_defaults_from_row(r)

        # Pull anchored fields from slide text
        anchored = _extract_anchored_fields(entry_text)

        sender = anchored.get("sender")
        receiver = anchored.get("receiver")
        subject = anchored.get("subject")
        date = anchored.get("date")

        # Prefer Requestor's Name if sender missing
        if not sender and anchored.get("requestor_name"):
            sender = anchored.get("requestor_name")

        # CQAS ids: scan full slide text
        cqas_ids = _all_cqas_ids(entry_text)
        cqas_id = cqas_ids[0] if cqas_ids else None

        # If date not anchored, try regex scan
        if not date:
            date = _best_date_from_text(entry_text)

        # Fallback to filename/title-derived defaults
        if not sender:
            sender = file_defaults.get("file_sender")
        if not subject:
            subject = file_defaults.get("file_subject")
        if not date:
            date = file_defaults.get("file_date")

        # feedback_text: slide entry_text (optionally stripped of anchor lines)
        feedback_text = entry_text
        if cfg.strip_anchor_lines_from_feedback:
            feedback_text = _strip_lines(feedback_text, anchored.get("consumed_lines") or [])

        entry_type = "slide"

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

            # slide segmentation metadata if present
            "slide_num": r.get("slide_num"),
            "start_block": r.get("start_block"),
            "end_block": r.get("end_block"),
            "seg_method": r.get("seg_method"),
            "seg_confidence": r.get("seg_confidence"),
            "field_anchor_counts_json": r.get("field_anchor_counts_json"),
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
            "raw_entry_json": json.dumps(
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
      conda run -n feedback python -m Caste_Project.parse.parse_pptx --in_dir data\\_seg_output --out_dir data\\_parse_output --docs_dir data\\_test_output

    Expects:
      in_dir/pptx_entries.parquet
      docs_dir/pptx_documents.parquet (recommended)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--entries_name", default="pptx_entries.parquet")
    ap.add_argument("--docs_dir", required=False, help="Folder containing pptx_documents.parquet (recommended)")
    ap.add_argument("--documents_name", default="pptx_documents.parquet")
    ap.add_argument("--strip_anchor_lines", action="store_true", help="Remove From/To/Subject/Date lines from feedback_text")
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

    cfg = PptxParseConfig(strip_anchor_lines_from_feedback=bool(args.strip_anchor_lines))
    wide_df, long_df = parse_pptx_entries_to_fields(entries_df, documents_df, cfg)

    wide_df.to_parquet(out_dir / "pptx_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "pptx_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'pptx_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'pptx_entry_fields_long.parquet'}")

    cols = ["doc_id", "entry_num", "slide_num", "sender", "date", "subject", "cqas_id"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()
