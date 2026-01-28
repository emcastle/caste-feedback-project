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
class DocxParseConfig:
    """
    Parses segmented DOCX entries into structured fields.

    Input:
      - docx_entries.parquet produced by segment_docx.py
        must have: doc_id, entry_num, entry_text
        optionally: start_block, end_block, seg_method, seg_confidence, error

    Output:
      - docx_entry_fields.parquet (wide): one row per entry
      - docx_entry_fields_long.parquet (long): one row per (entry, field)
    """
    max_header_lines: int = 60
    max_subject_chars: int = 600
    max_name_chars: int = 400
    max_date_chars: int = 160

    prefer_incoming_section: bool = True
    trim_signature_tail: bool = True


# -----------------------------
# Regex anchors
# -----------------------------
RE_CQAS = re.compile(r"\bCQAS[-–—]?\s*(\d{3,10})\b", re.IGNORECASE)

RE_FROM = re.compile(r"^\s*From\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_TO = re.compile(r"^\s*To\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_SUBJECT = re.compile(r"^\s*Subject\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_DATE = re.compile(r"^\s*Date\s*:\s*(.+?)\s*$", re.IGNORECASE)

# CQAS-style headers (seen in some DOCX exports)
RE_REQUESTOR = re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_DATE_OF_REQUEST = re.compile(r"^\s*Date\s+of\s+Request\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Section markers
RE_INCOMING = re.compile(r"^\s*Incoming\b\s*:?$", re.IGNORECASE)
RE_RESPONSE = re.compile(r"^\s*Response\b\s*:?$", re.IGNORECASE)
RE_RESPONSE_FOLLOWUP = re.compile(r"^\s*Response\s+Follow[- ]?Up\b\s*:?$", re.IGNORECASE)

# Signature-ish closings (optional trim)
RE_SIGNATURE_LINE = re.compile(
    r"^\s*(Sincerely|Regards|Respectfully|Very Respectfully|Thank you|Thanks)\b[, ]*\s*$",
    re.IGNORECASE,
)


# -----------------------------
# Helpers
# -----------------------------
def _split_lines(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [ln.strip() for ln in text.splitlines()]

def _first_match(lines: List[str], pat: re.Pattern, max_lines: int) -> Optional[str]:
    for ln in lines[:max_lines]:
        m = pat.match(ln)
        if m:
            return (m.group(1) or "").strip()
    return None

def _all_cqas_ids(text: str) -> List[str]:
    out = []
    for m in RE_CQAS.finditer(text or ""):
        num = (m.group(1) or "").strip()
        if num:
            out.append(f"CQAS-{num}")
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped

def _find_section_span(lines: List[str], start_pat: re.Pattern, end_pat: re.Pattern) -> Optional[Tuple[int, int]]:
    start_idx = None
    for i, ln in enumerate(lines):
        if start_pat.match(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    end_idx = None
    for j in range(start_idx + 1, len(lines)):
        if end_pat.match(lines[j]):
            end_idx = j
            break

    content_start = start_idx + 1
    content_end = (end_idx - 1) if end_idx is not None else (len(lines) - 1)

    if content_end < content_start:
        return None
    return (content_start, content_end)

def _strip_leading_headers(lines: List[str], cfg: DocxParseConfig) -> Tuple[List[str], Dict[str, Optional[str]]]:
    fields: Dict[str, Optional[str]] = {
        "from": None,
        "to": None,
        "subject": None,
        "date": None,
        "requestor_name": None,
        "date_of_request": None,
    }

    fields["from"] = _first_match(lines, RE_FROM, cfg.max_header_lines)
    fields["to"] = _first_match(lines, RE_TO, cfg.max_header_lines)
    fields["subject"] = _first_match(lines, RE_SUBJECT, cfg.max_header_lines)
    fields["date"] = _first_match(lines, RE_DATE, cfg.max_header_lines)

    fields["requestor_name"] = _first_match(lines, RE_REQUESTOR, cfg.max_header_lines)
    fields["date_of_request"] = _first_match(lines, RE_DATE_OF_REQUEST, cfg.max_header_lines)

    def headerish(ln: str) -> bool:
        if not ln:
            return True
        return bool(
            RE_FROM.match(ln)
            or RE_TO.match(ln)
            or RE_SUBJECT.match(ln)
            or RE_DATE.match(ln)
            or RE_REQUESTOR.match(ln)
            or RE_DATE_OF_REQUEST.match(ln)
        )

    k = 0
    while k < len(lines) and k < cfg.max_header_lines and headerish(lines[k]):
        k += 1

    return lines[k:], fields

def _trim_signature_tail(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    last_sig = None
    for i, ln in enumerate(lines):
        if RE_SIGNATURE_LINE.match(ln):
            last_sig = i
    if last_sig is None:
        return lines
    if last_sig >= max(0, len(lines) - 30):
        return lines[:last_sig]
    return lines

def _classify_entry(lines: List[str]) -> str:
    has_incoming = any(RE_INCOMING.match(ln) for ln in lines)
    has_response = any(RE_RESPONSE.match(ln) for ln in lines)
    has_followup = any(RE_RESPONSE_FOLLOWUP.match(ln) for ln in lines)

    if has_followup and not has_incoming:
        return "response_followup"
    if has_response and not has_incoming:
        return "response_only"
    if has_incoming and has_response:
        return "incoming_plus_response"
    if has_incoming:
        return "incoming_only"
    return "unknown"


# -----------------------------
# Core parse
# -----------------------------
def parse_docx_entries_to_fields(
    entries_df: pd.DataFrame,
    cfg: DocxParseConfig = DocxParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"doc_id", "entry_num", "entry_text"}
    missing = required - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries_df missing required cols: {sorted(missing)}")

    wide_rows: List[Dict] = []
    long_rows: List[Dict] = []

    for _, r in entries_df.iterrows():
        doc_id = r["doc_id"]
        entry_num = int(r["entry_num"]) if pd.notna(r["entry_num"]) else 0
        entry_text = r.get("entry_text") or ""

        lines_all = _split_lines(entry_text)

        cqas_ids = _all_cqas_ids(entry_text)
        cqas_id = cqas_ids[0] if cqas_ids else None

        body_lines, header_fields = _strip_leading_headers(lines_all, cfg)

        parsed_date = header_fields.get("date_of_request") or header_fields.get("date")

        feedback_text = ""
        if cfg.prefer_incoming_section:
            span = _find_section_span(lines_all, RE_INCOMING, RE_RESPONSE)
            if span:
                s, e = span
                feedback_lines = lines_all[s : e + 1]
                if cfg.trim_signature_tail:
                    feedback_lines = _trim_signature_tail(feedback_lines)
                feedback_text = "\n".join([ln for ln in feedback_lines if ln.strip()]).strip()

        if not feedback_text:
            cleaned = body_lines
            if cfg.trim_signature_tail:
                cleaned = _trim_signature_tail(cleaned)
            feedback_text = "\n".join([ln for ln in cleaned if ln.strip()]).strip()

        entry_type = _classify_entry(lines_all)

        sender = header_fields.get("from") or header_fields.get("requestor_name")
        receiver = header_fields.get("to")
        subject = header_fields.get("subject")
        if subject:
            subject = subject[: cfg.max_subject_chars]
        if sender:
            sender = sender[: cfg.max_name_chars]
        if receiver:
            receiver = receiver[: cfg.max_name_chars]
        if parsed_date:
            parsed_date = parsed_date[: cfg.max_date_chars]

        wide = {
            "doc_id": doc_id,
            "entry_num": entry_num,
            "start_block": r.get("start_block"),
            "end_block": r.get("end_block"),
            "seg_method": r.get("seg_method"),
            "seg_confidence": r.get("seg_confidence"),
            "entry_type": entry_type,

            "cqas_id": cqas_id,
            "cqas_ids_json": json.dumps(cqas_ids, ensure_ascii=False),

            "sender": sender,
            "receiver": receiver,
            "date": parsed_date,
            "subject": subject,

            "feedback_text": feedback_text,
            "raw_entry_text": entry_text,
            "error": r.get("error"),
        }
        wide_rows.append(wide)

        def emit(field_name: str, field_value: Optional[str]) -> None:
            long_rows.append(
                {
                    "doc_id": doc_id,
                    "entry_num": entry_num,
                    "field_name": field_name,
                    "field_value": field_value,
                }
            )

        emit("cqas_id", cqas_id)
        emit("sender", sender)
        emit("receiver", receiver)
        emit("date", parsed_date)
        emit("subject", subject)
        emit("entry_type", entry_type)
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
      conda run -n feedback python -m Caste_Project.parse.parse_docx --in_dir data\\_seg_output --out_dir data\\_parse_output

    Expects:
      in_dir/docx_entries.parquet

    Writes:
      out_dir/docx_entry_fields.parquet
      out_dir/docx_entry_fields_long.parquet
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--entries_name", default="docx_entries.parquet", help="Filename of segmented DOCX entries parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries_path = in_dir / args.entries_name
    if not entries_path.exists():
        raise FileNotFoundError(f"Missing {entries_path}. If your segmenter wrote a different name, pass --entries_name")

    entries_df = pd.read_parquet(entries_path)

    wide_df, long_df = parse_docx_entries_to_fields(entries_df, DocxParseConfig())

    wide_df.to_parquet(out_dir / "docx_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "docx_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'docx_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'docx_entry_fields_long.parquet'}")

    cols = ["doc_id", "entry_num", "entry_type", "cqas_id", "sender", "receiver", "date", "subject"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()