from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class DocxParseConfig:
    max_header_lines: int = 80
    max_tail_lines: int = 60
    prefer_filename_metadata: bool = True

    # output caps
    max_subject_chars: int = 600
    max_name_chars: int = 300
    max_receiver_chars: int = 600
    max_date_chars: int = 160


# -----------------------------
# Patterns
# -----------------------------
RE_CQAS = re.compile(r"\bCQAS[-–—]?\s*(\d{3,10})\b", re.IGNORECASE)

# Date patterns inside text (no "Date:" prompt required)
RE_DATE_TEXT = re.compile(
    r"\b("
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}"
    r"|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

# Date patterns inside filename like "10-22" or "10-22-2021" or "10222021"
RE_DATE_FILE = re.compile(
    r"\b("
    r"\d{1,2}[-_]\d{1,2}([-_]\d{2,4})?"
    r"|"
    r"\d{8}"
    r")\b"
)

# Month year patterns
RE_MONTH_YEAR = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
    re.IGNORECASE
)

# “Dear …” suggests letter format
RE_DEAR = re.compile(r"^\s*Dear\b", re.IGNORECASE)

# Closings suggest signature starts near the bottom
RE_CLOSING = re.compile(
    r"^\s*(Sincerely|Regards|Respectfully|Very Respectfully|Thank you|Thanks)\b[, ]*\s*$",
    re.IGNORECASE,
)

# crude “name-ish” line detector for signatures (not perfect, but works well in practice)
RE_NAMEISH = re.compile(r"^[A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){0,4}$")

# receiver/address block heuristics
RE_HAS_ZIP = re.compile(r"\b\d{5}(-\d{4})?\b")
RE_HAS_STATE_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(-\d{4})?\b")

# email, divider, address blocks 
RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
RE_DIVIDER = re.compile(r"^\s*[_\-]{6,}\s*$")
RE_ADDRESS_CUE = re.compile(
    r"(u\.?\s*s\.?\s*census\s+bureau|via\s+electronic\s+mail|disclosure\s+avoidance|systems?)",
    re.IGNORECASE,
)
RE_UNDERSCORE_LINE = re.compile(r"^\s*_{5,}\s*$")


# -----------------------------S
# Helpers
# -----------------------------
def _split_lines(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [ln.strip() for ln in text.splitlines()]

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

def _coerce_text(text) -> str:
    # If docx parser passes a list of paragraphs/lines, join into one string
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, (list, tuple)):
        return "\n".join(str(x) for x in text if x is not None)
    return str(text)

def _find_first_date_in_text(text, max_lines: int = 60) -> str | None:
    """
    Return the first date-like string found near the top of the text.
    Accepts either a string or list/tuple of strings.
    """
    text = _coerce_text(text)
    if not text.strip():
        return None

    lines = text.splitlines()
    scan_text = "\n".join(lines[:max_lines])

    m = RE_DATE_TEXT.search(scan_text)
    if m:
        return m.group(0)

    m = RE_MONTH_YEAR.search(scan_text)
    if m:
        return m.group(0)

    return None

# If the meta data is only in the file title
def _parse_filename_metadata(source_file: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Observed pattern:
      "Greg Robinson - ASAN CENSUS FEEDBACK Group Quarters Final 10-22.docx"
    Heuristic:
      - sender: first token before first " - "
      - date: last date-like token anywhere in filename
      - subject: middle chunk between sender and date-ish tail
    """
    meta = {"sender": None, "subject": None, "date": None}
    if not source_file:
        return meta

    stem = Path(source_file).stem

    # date candidate from filename
    date_match = None
    for m in RE_DATE_FILE.finditer(stem):
        date_match = m.group(1)
    if date_match:
        meta["date"] = date_match

    # split on " - " (your naming convention)
    parts = [p.strip() for p in stem.split(" - ") if p.strip()]
    if parts:
        meta["sender"] = parts[0][:300]

    # subject from remaining text (remove sender and remove date-ish tail)
    remainder = stem
    if meta["sender"] and remainder.startswith(meta["sender"]):
        remainder = remainder[len(meta["sender"]):].lstrip(" -_")
    if meta["date"]:
        remainder = remainder.replace(meta["date"], "").strip(" -_")

    # clean “double spaces”
    remainder = re.sub(r"\s{2,}", " ", remainder).strip()
    meta["subject"] = remainder[:600] if remainder else None

    return meta

# extract reciever from address block if it exists
def _extract_receiver_block(lines: List[str], cfg: DocxParseConfig) -> Optional[str]:
    """
     Find a receiver/address block anywhere in the header region (not just at the top).
    Strategy:
      1) Split header into blocks separated by blank lines or underscore separators.
      2) Score blocks based on strong receiver cues (email, 'via electronic mail', Census Bureau, etc.)
      3) Return the best block if it meets a minimum score.
    """
    head = lines[: cfg.max_header_lines]

    # 1) build blocks
    blocks: List[List[str]] = []
    cur: List[str] = []

    for ln in head:
        ln = (ln or "").strip()

        # treat underscore separators like a blank line boundary
        is_sep = (not ln) or bool(RE_UNDERSCORE_LINE.match(ln))
        if is_sep:
            if cur:
                blocks.append(cur)
                cur = []
            continue

        cur.append(ln)

    if cur:
        blocks.append(cur)

    if not blocks:
        return None

    # 2) score each block
    def score_block(block: List[str]) -> int:
        txt = "\n".join(block)
        t = txt.lower()

        score = 0

        # strong cues
        if "via electronic mail" in t:
            score += 5
        if "u.s. census bureau" in t or "us census bureau" in t:
            score += 4
        if RE_EMAIL.search(txt):
            score += 4

        # address-ish structure cues
        if len(block) >= 3:
            score += 2
        if RE_HAS_ZIP.search(txt) or RE_HAS_STATE_ZIP.search(txt):
            score += 2

        # penalize blocks that look like titles/subjects only (very long single line)
        if len(block) == 1 and len(block[0]) > 80:
            score -= 2

        return score

    scored = [(score_block(b), b) for b in blocks]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_block = scored[0]

    # 3) require some minimum quality
    # If you want to be strict, set this to 4; if you want looser, set to 3.
    if best_score < 4:
        return None

    # cap and return
    return "\n".join(best_block[:10]).strip()[: cfg.max_receiver_chars]


def _extract_sender_from_signature(lines: List[str], cfg: DocxParseConfig) -> Optional[str]:
    """
    Look near the end:
      find closing phrase, then search next few lines for a name-ish line.
    Fallback: scan last tail lines for a name-ish line.
    """
    tail = lines[-cfg.max_tail_lines:] if len(lines) > cfg.max_tail_lines else lines[:]

    # if closing exists, prefer name soon after
    for i, ln in enumerate(tail):
        if RE_CLOSING.match(ln):
            for j in range(i + 1, min(i + 8, len(tail))):
                cand = tail[j].strip()
                if RE_NAMEISH.match(cand):
                    return cand[: cfg.max_name_chars]
            break

    # fallback: last name-ish line in tail
    for ln in reversed(tail):
        cand = ln.strip()
        if RE_NAMEISH.match(cand):
            return cand[: cfg.max_name_chars]

    return None

def _classify_docx_entry(lines: List[str]) -> str:
    """
    DOCX letters usually:
      - have address block + date near top
      - have Dear ... and closing
    """
    head = lines[:30]
    tail = lines[-40:] if len(lines) > 40 else lines

    has_dear = any(RE_DEAR.match(ln) for ln in head)
    has_closing = any(RE_CLOSING.match(ln) for ln in tail)

    if has_dear or has_closing:
        return "letter"
    # if it looks like an email (rare in your 3)
    if any(ln.lower().startswith(("from:", "to:", "subject:", "date:")) for ln in head):
        return "email_like"
    return "memo_or_note"


# -----------------------------
# Core parse
# -----------------------------
def parse_docx_entries_to_fields(
    entries_df: pd.DataFrame,
    documents_df: Optional[pd.DataFrame] = None,
    cfg: DocxParseConfig = DocxParseConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"doc_id", "entry_num", "entry_text"}
    missing = required - set(entries_df.columns)
    if missing:
        raise ValueError(f"entries_df missing required cols: {sorted(missing)}")

    df = entries_df.copy()

    # Bring in filename metadata (critical for your observed pattern)
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
        entry_text = r.get("entry_text") or ""
        source_file = r.get("source_file")

        lines = _split_lines(entry_text)

        # CQAS (may be absent in your DOCX letters)
        cqas_ids = _all_cqas_ids(entry_text)
        cqas_id = cqas_ids[0] if cqas_ids else None

        # filename-derived
        file_meta = _parse_filename_metadata(source_file) if cfg.prefer_filename_metadata else {"sender": None, "subject": None, "date": None}

        # content-derived
        date_text = _find_first_date_in_text(lines, max_lines=25)
        receiver_block = _extract_receiver_block(lines, cfg) or "Not Applicable"
        sender_sig = _extract_sender_from_signature(lines, cfg)

        # choose best values
        sender = file_meta.get("sender") or sender_sig
        subject = file_meta.get("subject")
        date = date_text or file_meta.get("date")

        # entry type
        entry_type = _classify_docx_entry(lines)

        # feedback_text: for letters just keep whole entry (you can refine later)
        feedback_text = entry_text.strip()

        # cap strings
        if sender:
            sender = sender[: cfg.max_name_chars]
        if receiver_block:
            receiver_block = receiver_block[: cfg.max_receiver_chars]
        if subject:
            subject = subject[: cfg.max_subject_chars]
        if date:
            date = date[: cfg.max_date_chars]

        wide = {
            "doc_id": doc_id,
            "entry_num": entry_num,
            "source_file": source_file,
            "source_rel_path": r.get("source_rel_path"),

            "entry_type": entry_type,

            "cqas_id": cqas_id,
            "cqas_ids_json": json.dumps(cqas_ids, ensure_ascii=False),

            "sender": sender,
            "receiver": receiver_block,
            "date": date,
            "subject": subject,

            "feedback_text": feedback_text,
            "raw_entry_text": entry_text,
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
        emit("receiver", receiver_block)
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
      conda run -n feedback python -m Caste_Project.parse.parse_docx --in_dir data\\_seg_output --docs_dir data\\_test_output --out_dir data\\_parse_output

    Expects:
      in_dir/docx_entries.parquet
      docs_dir/docx_documents.parquet   (to get source_file)
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--entries_name", default="docx_entries.parquet")
    ap.add_argument("--docs_dir", required=False, help="Folder containing docx_documents.parquet (recommended)")
    ap.add_argument("--documents_name", default="docx_documents.parquet")
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

    wide_df, long_df = parse_docx_entries_to_fields(entries_df, documents_df, DocxParseConfig())

    wide_df.to_parquet(out_dir / "docx_entry_fields.parquet", index=False)
    long_df.to_parquet(out_dir / "docx_entry_fields_long.parquet", index=False)

    print(f"Saved: {out_dir / 'docx_entry_fields.parquet'}")
    print(f"Saved: {out_dir / 'docx_entry_fields_long.parquet'}")

    cols = ["doc_id", "entry_num", "entry_type", "sender", "date", "subject"]
    cols = [c for c in cols if c in wide_df.columns]
    print("\nParse quick checks (head):")
    print(wide_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main_cli()