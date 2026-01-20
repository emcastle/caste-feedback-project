from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class TxtReadConfig:
    """
    Configuration for reading plain text files.
    """
    encoding: Optional[str] = None  # if None, try fallbacks
    keep_empty_lines: bool = False
    normalize_newlines: bool = True
    max_chars_per_line: int = 20000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _truncate(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + " ...[TRUNCATED]"


def _read_text_with_fallbacks(path: Path, cfg: TxtReadConfig) -> str:
    if cfg.encoding:
        return path.read_text(encoding=cfg.encoding, errors="replace")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception as e:
            last_err = e

    # last resort: replace bad chars
    return path.read_text(encoding="utf-8", errors="replace")


def extract_txt_to_relational(
    txt_path: Path,
    source_rel_path: str,
    cfg: TxtReadConfig = TxtReadConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract a .txt into documents_df and lines_df.
    """
    txt_path = txt_path.resolve()
    doc_id = sha256_file(txt_path)

    source_file = txt_path.name
    source_ext = txt_path.suffix.lower() if txt_path.suffix else ".txt"
    source_type = "text"
    extractor_used = "Path.read_text"

    try:
        raw = _read_text_with_fallbacks(txt_path, cfg)

        if cfg.normalize_newlines:
            raw = raw.replace("\r\n", "\n").replace("\r", "\n")

        lines = raw.split("\n")

        if not cfg.keep_empty_lines:
            # keep line numbering stable by removing only trailing empties?
            # simplest: remove all empty lines
            lines = [ln for ln in lines if ln.strip()]

        lines_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "line_num": i,
                    "line_text": _truncate(ln, cfg.max_chars_per_line),
                    "error": None,
                }
                for i, ln in enumerate(lines)
            ]
        )

        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_lines": int(len(lines_df)),
                    "extractor_used": extractor_used,
                    "error": None,
                }
            ]
        )

        return documents_df, lines_df

    except Exception as e:
        documents_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "source_rel_path": source_rel_path,
                    "source_ext": source_ext,
                    "source_type": source_type,
                    "num_lines": None,
                    "extractor_used": extractor_used,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        lines_df = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "line_num": None,
                    "line_text": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            ]
        )

        return documents_df, lines_df
