"""
antigravity/chunker.py — Chunk classifier (spec §4).

Classifies each text chunk into one of:
  text | math | table | figure | code | references | noise

Uses deterministic heuristics, not LLM calls.
"""

from __future__ import annotations

import re

from antigravity.models import ChunkType


# ── Heuristic rules ───────────────────────────────────────────────────────────

_LATEX_PATTERN = re.compile(r"\\(?:frac|sum|int|sqrt|begin\{equation\}|mathbb|alpha|beta|gamma|delta|theta|sigma|lambda|nabla|partial)", re.IGNORECASE)
_TABLE_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)
_TABLE_HEADER = re.compile(r"Table\s+\d+", re.IGNORECASE)
_FIGURE_PATTERN = re.compile(r"(?:Figure\s+\d+|!\[.*\]\(.*\))", re.IGNORECASE)
_CODE_PATTERN = re.compile(r"```[\s\S]*?```|def\s+\w+\(|class\s+\w+[:\(]|import\s+\w+|function\s+\w+\(", re.MULTILINE)
_REFERENCE_PATTERN = re.compile(r"^\s*\[\d+\]\s+", re.MULTILINE)
_NOISE_THRESHOLD = 0.35  # ratio of non-alphanum to total chars


def classify_chunk(text: str) -> ChunkType:
    """Classify a chunk of text into one of the 7 chunk types."""
    if not text or not text.strip():
        return ChunkType.NOISE

    stripped = text.strip()
    total_chars = len(stripped)

    # ── Noise check: high ratio of non-alphanumeric chars (OCR garbage)
    alphanum = sum(1 for c in stripped if c.isalnum() or c.isspace())
    if total_chars > 20 and (alphanum / total_chars) < (1 - _NOISE_THRESHOLD):
        return ChunkType.NOISE

    # ── References: citation list format  [1] Author et al. ...
    ref_matches = _REFERENCE_PATTERN.findall(stripped)
    if len(ref_matches) >= 2:
        return ChunkType.REFERENCES

    # ── Math: LaTeX / symbolic density
    latex_hits = len(_LATEX_PATTERN.findall(stripped))
    if latex_hits >= 2:
        return ChunkType.MATH

    # ── Table: column structure or "Table X"
    table_rows = len(_TABLE_PATTERN.findall(stripped))
    if table_rows >= 2 or _TABLE_HEADER.search(stripped):
        return ChunkType.TABLE

    # ── Figure: caption or image reference
    if _FIGURE_PATTERN.search(stripped):
        return ChunkType.FIGURE

    # ── Code: code syntax patterns
    code_hits = len(_CODE_PATTERN.findall(stripped))
    if code_hits >= 2:
        return ChunkType.CODE

    # ── Default: coherent prose → text
    return ChunkType.TEXT
