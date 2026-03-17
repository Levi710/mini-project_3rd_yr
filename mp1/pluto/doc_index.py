"""
pluto/doc_index.py — In-memory document index with disk persistence.

Stores pre-processed document data so chunks are split once,
classified once, and the LLM overview is computed once per document.
All subsequent queries reuse this cached state.

Persists to a JSON file so data survives server restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChunkMeta:
    """Metadata for a single chunk."""
    chunk_id: str          # "C0", "C1", ...
    chunk_type: str        # "text", "math", "table", "figure", "code", "references", "noise"
    mode: str              # "MODE_QUICK", "MODE_REASONING", "MODE_VISION"
    header: str = ""       # nearest heading / section title
    relevance: float = 0.0 # query-time relevance score (updated per query)


@dataclass
class DocEntry:
    """All pre-processed data for a single document."""
    doc_id: str
    filename: str = ""
    chunks: list[str] = field(default_factory=list)
    chunk_meta: list[ChunkMeta] = field(default_factory=list)
    overview: str = ""                            # LLM-generated understanding
    chunk_topics: dict[str, list[str]] = field(default_factory=dict)  # {chunk_id: [topics]}
    is_processed: bool = False                    # True after Phase A completes


class DocIndex:
    """
    In-memory index of pre-processed documents with disk persistence.

    Populated during upload (Phase A). Queried during pipeline run (Phase B).
    Persists to a JSON file so data survives server restarts.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._docs: dict[str, DocEntry] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        # Load from disk if available
        if self._persist_path:
            self._load_from_disk()

    # ── Persistence ──────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        """Persist the index to disk as JSON."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for doc_id, entry in self._docs.items():
                data[doc_id] = {
                    "doc_id": entry.doc_id,
                    "filename": entry.filename,
                    "chunks": entry.chunks,
                    "chunk_meta": [
                        {"chunk_id": m.chunk_id, "chunk_type": m.chunk_type,
                         "mode": m.mode, "header": m.header}
                        for m in entry.chunk_meta
                    ],
                    "overview": entry.overview,
                    "chunk_topics": entry.chunk_topics,
                    "is_processed": entry.is_processed,
                }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception:
            pass  # Don't crash the pipeline for a cache write failure

    def _load_from_disk(self) -> None:
        """Load the index from disk JSON."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            raw = self._persist_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            for doc_id, entry_data in data.items():
                meta_list = [
                    ChunkMeta(
                        chunk_id=m["chunk_id"],
                        chunk_type=m["chunk_type"],
                        mode=m["mode"],
                        header=m.get("header", ""),
                    )
                    for m in entry_data.get("chunk_meta", [])
                ]
                self._docs[doc_id] = DocEntry(
                    doc_id=entry_data["doc_id"],
                    filename=entry_data.get("filename", ""),
                    chunks=entry_data.get("chunks", []),
                    chunk_meta=meta_list,
                    overview=entry_data.get("overview", ""),
                    chunk_topics=entry_data.get("chunk_topics", {}),
                    is_processed=entry_data.get("is_processed", False),
                )
        except Exception:
            pass  # Corrupted file — start fresh

    # ── Write API (Phase A) ──────────────────────────────────────────

    def register_doc(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        chunk_meta: list[ChunkMeta],
    ) -> None:
        """Register a document with its pre-split chunks and metadata."""
        self._docs[doc_id] = DocEntry(
            doc_id=doc_id,
            filename=filename,
            chunks=chunks,
            chunk_meta=chunk_meta,
            overview="",
            is_processed=False,
        )
        self._save_to_disk()

    def set_overview(self, doc_id: str, overview: str) -> None:
        """Store the LLM-generated understanding for a document."""
        if doc_id in self._docs:
            self._docs[doc_id].overview = overview
            self._docs[doc_id].is_processed = True
            self._save_to_disk()

    def set_chunk_topics(self, doc_id: str, chunk_topics: dict[str, list[str]]) -> None:
        """Store per-chunk topic tags for intelligent routing."""
        if doc_id in self._docs:
            self._docs[doc_id].chunk_topics = chunk_topics
            self._save_to_disk()

    def remove_doc(self, doc_id: str) -> None:
        """Remove a document from the index."""
        self._docs.pop(doc_id, None)
        self._save_to_disk()

    # ── Read API (Phase B) ───────────────────────────────────────────

    def is_processed(self, doc_id: str) -> bool:
        """Check if a document has been fully processed (Phase A complete)."""
        entry = self._docs.get(doc_id)
        return entry.is_processed if entry else False

    def get_chunks(self, doc_id: str) -> list[str]:
        """Return all chunk texts for a document."""
        entry = self._docs.get(doc_id)
        return entry.chunks if entry else []

    def get_chunk(self, doc_id: str, chunk_index: int) -> str:
        """Return a specific chunk by index."""
        chunks = self.get_chunks(doc_id)
        if 0 <= chunk_index < len(chunks):
            return chunks[chunk_index]
        return ""

    def get_chunk_meta(self, doc_id: str) -> list[ChunkMeta]:
        """Return chunk metadata list for a document."""
        entry = self._docs.get(doc_id)
        return entry.chunk_meta if entry else []

    def get_overview(self, doc_id: str) -> str:
        """Return the LLM-generated understanding for a document."""
        entry = self._docs.get(doc_id)
        return entry.overview if entry else ""

    def get_chunk_topics(self, doc_id: str) -> dict[str, list[str]]:
        """Return chunk topic map: {chunk_id: [topic1, topic2, role]}."""
        entry = self._docs.get(doc_id)
        return entry.chunk_topics if entry else {}

    def get_chunk_count(self, doc_id: str) -> int:
        """Return number of chunks in a document."""
        return len(self.get_chunks(doc_id))

    def list_docs(self) -> list[dict[str, Any]]:
        """Return summary info for all indexed documents."""
        return [
            {
                "doc_id": e.doc_id,
                "filename": e.filename,
                "chunk_count": len(e.chunks),
                "is_processed": e.is_processed,
                "has_overview": bool(e.overview),
            }
            for e in self._docs.values()
        ]

    def has_doc(self, doc_id: str) -> bool:
        """Check if a document is in the index."""
        return doc_id in self._docs
