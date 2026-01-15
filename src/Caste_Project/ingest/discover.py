from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


"""
discover.py

Purpose
-------
Discover and inventory input files under a raw data directory.

This module builds a "manifest" (an inventory table) of supported files found under
a root folder (typically `data/raw`). The manifest is used as an audit trail and as a
stable input to downstream extraction steps.

Manifest Columns
----------------
- rel_path: path relative to the raw root (normalized with forward slashes)
- abs_path: absolute file path on disk
- ext: lowercased file extension (e.g., ".pdf", ".docx")
- bytes: file size in bytes
- mtime_epoch: last-modified time (Unix epoch seconds)
- hash_sha256: optional content fingerprint (None if not computed or skipped)

Notes on SHA-256
----------------
SHA-256 hashing is optional because it can be expensive for very large files.
When enabled, it provides:
- deduplication (same content, different filenames)
- robust change detection across runs
- reproducibility ("which exact file version was processed?")

Key Functions
-------------
- iter_supported_files(root, exts): yields supported file paths under `root`
- sha256_file(path, max_mb): compute SHA-256, optionally skipping very large files
- build_manifest(raw_dir, compute_hash, max_hash_mb): returns a pandas DataFrame manifest
"""

# Expected file types
SUPPORTED_EXTS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls",
    "json",
    "json5",
    ".par"
}


def sha256_file(path: Path, max_mb: int = 50) -> Optional[str]:
    """
    Compute sha256 for dedup/debug. For very large files, optionally skip hashing.
    """
    size_bytes = path.stat().st_size
    if size_bytes > max_mb * 1024 * 1024:
        return None

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def build_manifest(raw_dir: Path, compute_hash: bool = False) -> pd.DataFrame:
    raw_dir = raw_dir.resolve()
    rows = []
    for p in iter_files(raw_dir, SUPPORTED_EXTS):
        stat = p.stat()
        rel = p.relative_to(raw_dir)
        rows.append(
            {
                "rel_path": str(rel).replace("\\", "/"),
                "abs_path": str(p),
                "ext": p.suffix.lower(),
                "bytes": stat.st_size,
                "mtime_epoch": int(stat.st_mtime),
                "hash_sha256": sha256_file(p) if compute_hash else None,
            }
        )
    df = pd.DataFrame(rows).sort_values(["ext", "rel_path"]).reset_index(drop=True)
    return df