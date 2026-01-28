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
class PdfParseConfig:
    """
    Parse PDF entry_text into structured fields.
    Assumes input is pdf_entries.parquet created by segment_pdf.py.

    Key behavior:
      - Extract email-like headers if present (From/To/Date/Subject)
      - Extract CQAS id if present
      - Prefer feedback content from "Incoming ... Response" spans if present
      - Otherwise, fall back to "best-effort body text" after stripping headers
    """
    max_header_lines: int = 40
    max_subject_chars: int = 500
    max_name_chars: int = 300
    max_date_chars: int = 120

    # If Incoming/Response exist, we take Incoming-only as the feedback_text.
    prefer_incoming_section: bool = True

    # If entry looks like "Response Follow-Up" only, label it.
    classify_entry_type: bool = True


# -----------------------------
# Regex anchors (field parsing)
# -----------------------------
RE_CQAS = re.compile(r"\bCQAS[-–—]?\s*(\d{3,8})\b", re.IGNORECASE)

RE_FROM = re.compile(r"^\s*From\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_TO = re.compile(r"^\s*To\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_SUBJECT = re.compile(r"^\s*Subject\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_DATE = re.compile(r"^\s*Date\s*:\s*(.+?)\s*$", re.IGNORECASE)

# CQAS form headers (seen in your CQAS-style docs)
RE_REQUESTOR = re.compile(r"^\s*Requestor[’']?s\s+Name\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_DATE_OF_REQUEST = re.compile(r"^\s*Date\s+of\s+Request\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Section markers
RE_INCOMING = re.compile(r"^\s*Incoming\b\s*:?$", re.IGNORECASE)
RE_RESPONSE = re.compile(r"^\s*Response\b\s*:?$", re.IGNORECASE)
RE_RESPONSE_FOLLOWUP = re.compile(r"^\s*Response\s+Follow[- ]?Up\b\s*:?$", re.IGNORECASE)
RE_ORIGINAL_MESSAGE = re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE)

# Signature-ish closings (used to trim tail sometimes)
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

def _all_matches(text: str, pat: re.Pattern) -> List[str]:
    return [m.group(0).strip() for m in pat.finditer(text or "")]

def _find_section_span(lines: List[str], start_pat: re.Pattern, end_pat: re.Pattern) -> Optional[Tuple[int, int]]:
    """
    Returns (start_idx_exclusive, end_idx_inclusive) of content between markers.
    Example:
      Incoming
      <content...>
      Response
    returns indices covering only the <content...> lines.
    """
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

    # Content starts after marker line
    content_start = start_idx + 1

    # If no end marker, content goes to end
    content_end = (end_idx - 1) if end_idx is not None else (len(lines) - 1)

    if content_end < content_start:
        return None
    return (content_start, content_end)

def _strip_leading_headers(lines: List[str], cfg: PdfParseConfig) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """
    Pull common headers from the first N lines and return:
      - body_lines with header lines removed (best-effort)
      - extracted fields dict
    """
    fields: Dict[str, Optional[str]] = {
        "from": None,
        "to": None,
        "subject": None,
        "date": None,
        "requestor_name": None,
        "date_of_request": None,
    }

    # Extract values from header-like lines
    fields["from"] = _first_match(lines, RE_FROM, cfg.max_header_lines)
    fields["to"] = _first_match(lines, RE_TO, cfg.max_header_lines)
    fields["subject"] = _first_match(lines, RE_SUBJECT, cfg.max_header_lines)
    fields["date"] = _first_match(lines, RE_DATE, cfg.max_header_lines)

    fields["requestor_name"] = _first_match(lines, RE_REQUESTOR, cfg.max_header_lines)
    fields["date_of_request"] = _first_match(lines, RE_DATE_OF_REQUEST, cfg.max_header_lines)

    # Remove obvious header block: keep removing from top while line looks header-like/empty
    def looks_headerish(ln: str) -> bool:
        if not ln:
            return True
        if RE_FROM.match(ln) or RE_TO.match(ln) or RE_SUBJECT.match(ln) or RE_DATE.match(ln):
            return True
        if RE_REQUESTOR.match(ln) or RE_DATE_OF_REQUEST.match(ln):
            return True
        if RE_ORIGINAL_MESSAGE.match(ln):
            return True
        return False

    k = 0
    while k < len(lines) and k < cfg.max_header_lines and looks_headerish(lines[k]):
        k += 1

    body_lines = lines[k:]
    return body_lines, fields

def _trim_signature_tail(lines: List[str]) -> List[str]:
    """
    Trim after a likely signature closing line, if it occurs near the end.
    """
    if not lines:
        return lines
    # Find last signature marker
    last_sig = None
    for i, ln in enumerate(lines):
        if RE_SIGNATURE_LINE.match(ln):
            last_sig = i
    if last_sig is None:
        return lines

    # If signature is near the end, cut there
    if last_sig >= max(0, len(lines) - 25):
        return lines[:last_sig]
    return lines

def _classify_entry(lines: List[str]) -> str:
    """
    Very simple type guess.
    """
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
def parse_pdf_entries_to_fields(
    entries_df: pd.DataFrame,
    cfg: PdfParseConfig = PdfParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input: pdf_entries.parquet (from segmenter)
    Output:
      - fields_wide_df: one row per (doc_id, entry_num)
      - fields_long_df: one row per field per (doc_id, entry_num) (flexible truth)
    """
    wide_rows: List[Dict] = []
    long_rows: List[Dict] = []

    required_cols = {"doc_id", "entry_num", "entry_text"}
    missing = required_cols - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries_df missing required cols: {sorted(missing)}")

    for _, r in entries_df.iterrows():
        doc_id = r["doc_id"]
        entry_num = int(r["entry_num"]) if pd.notna(r["entry_num"]) else 0
        entry_text = r.get("entry_text") or ""

        lines_all = _split_lines(entry_text)

        # Extract CQAS ids (could be none or multiple)
        cqas_hits = _all_matches(entry_text, RE_CQAS)
        cqas_id = cqas_hits[0] if cqas_hits else None

        # Pull headers + remove header block from body candidate
        body_lines, header_fields = _strip_leading_headers(lines_all, cfg)

        # Prefer "Incoming ... Response" content if present
        feedback_text = ""
        if cfg.prefer_incoming_section:
            span = _find_section_span(lines_all, RE_INCOMING, RE_RESPONSE)
            if span:
                s, e = span
                feedback_lines = lines_all[s : e + 1]
                feedback_lines = _trim_signature_tail(feedback_lines)
                feedback_text = "\n".join([ln for ln in feedback_lines if ln.strip()]).strip()

        # If no incoming-span extracted, fallback to cleaned body
        if not feedback_text:
            cleaned = _trim_signature_tail(body_lines)
            feedback_text = "\n".join([ln for ln in cleaned if ln.strip()]).strip()

        # Entry type (incoming vs response-only, etc.)
        entry_type = _classify_entry(lines_all) if cfg.classify_entry_type else "unknown"

        # Pick date: prefer CQAS "Date of Request" if present, else Date:
        parsed_date = header_fields.get("date_of_request") or header_fields.get("date")

        # Wide row
        wide = {
            "doc_id": doc_id,
            "entry_num": entry_num,
            "start_page": r.get("start_page"),
            "end_page": r.get("end_page"),
            "seg_method": r.get("seg_method"),
            "seg_confidence": r.get("seg_confidence"),
            "entry_type": entry_type,

            "cqas_id": cqas_id,
            "sender": header_fields.get("from") or header_fields.get("requestor_name"),
            "receiver": header_fields.get("to"),
            "subject": (header_fields.get("subject") or "")[: cfg.max_subject_chars] if header_fields.get("subject") else None,
            "date": (parsed_date or "")[: cfg.max_date_chars] if parsed_date else None,

            "feedback_text": feedback_text,
            "raw_entry_text": entry_text,
            "error": r.get("error"),
        }
        wide_rows.append(wide)

        # Long (flexible truth): one row per field
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
        emit("sender", wide["sender"])
        emit("receiver", wide["receiver"])
        emit("subject", wide["subject"])
        emit("date", wide["date"])
        emit("entry_type", entry_type)
        emit("feedback_text", feedback_text)

    fields_wide_df = pd.DataFrame(wide_rows)
    fields_long_df = pd.DataFrame(long_rows)

    # Deterministic ordering + entry_num contiguity per doc (optional but helpful)
    if not fields_wide_df.empty:
        fields_wide_df = fields_wide_df.sort_values(["doc_id", "entry_num"]).reset_index(drop=True)

    if not fields_long_df.empty:
        fields_long_df = fields_long_df.sort_values(["doc_id", "entry_num", "field_name"]).reset_index(drop=True)

    return fields_wide_df, fields_long_df


# -----------------------------
# CLI
# -----------------------------
def main_cli() -> None:
    """
    Usage:
      conda run -n feedback python -m Caste_Project.parse.parse_pdf --in_dir data/_seg_output --out_dir data/_parse_output
    Expects:
      in_dir/pdf_entries.parquet
    Writes:
      out_dir/pdf_entry_fields.parquet       (wide)
      out_dir/pdf_entry_fields_long.parquet  (long key/value)
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries_df = pd.read_parquet(in_dir / "pdf_entries.parquet")

    wide_df, long_df = parse_pdf_entries_to_fields(entries_df, PdfParseConfig())

    wide_df.to_parquet(out_dir / "pdf_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "pdf_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'pdf_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'pdf_entry_fields_long.parquet'}")

    # quick visibility
    cols = ["doc_id", "entry_num", "entry_type", "cqas_id", "sender", "receiver", "date", "subject"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()