"""
antigravity/extraction_cache.py — Persistent cache for S1 EXTRACT results.

Stores LLM extraction outputs keyed by chunk content SHA-256 hash.
On cache hit, the expensive LLM call is skipped entirely.

Cache file: <corpus_dir>/.extraction_cache.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExtractionCache:
    """JSON-file-backed cache for chunk extraction results."""

    def __init__(self, corpus_dir: str) -> None:
        self._path = Path(corpus_dir).resolve() / ".extraction_cache.json"
        self._data: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, chunk_hash: str) -> dict[str, Any] | None:
        """Return cached extraction dict for this chunk hash, or None."""
        entry = self._data.get(chunk_hash)
        if entry:
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def put(self, chunk_hash: str, extract_dict: dict[str, Any]) -> None:
        """Store an extraction result keyed by chunk hash."""
        extract_dict["cached_at"] = datetime.now(timezone.utc).isoformat()
        self._data[chunk_hash] = extract_dict

    def invalidate_doc(self, doc_id: str) -> int:
        """Remove all cached entries for a specific document. Returns count removed."""
        to_remove = [
            h for h, entry in self._data.items()
            if entry.get("doc_id") == doc_id
        ]
        for h in to_remove:
            del self._data[h]
        return len(to_remove)

    def save(self) -> None:
        """Persist cache to disk."""
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass  # Non-fatal: cache is a performance optimization

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "total_entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / max(self.hits + self.misses, 1), 2),
            "cache_file": str(self._path),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
