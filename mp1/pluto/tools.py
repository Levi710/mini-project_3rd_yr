"""
pluto/tools.py — Corpus access tools (spec §3).

Implements list_docs, search, get_chunk, get_figure, get_table, log, finish
over a local corpus/ directory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pluto.tracer import Tracer


class CorpusTools:
    """File-backed implementation of the spec's external tool interface."""

    def __init__(self, corpus_dir: str, output_dir: str = "./output", tracer: Tracer | None = None, doc_index=None) -> None:
        self.corpus_dir = Path(corpus_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracer = tracer
        self.doc_index = doc_index  # DocIndex instance (if available)
        self._doc_cache: dict[str, str] = {}
        self._chunk_cache: dict[str, list[str]] = {}  # doc_id -> list of chunks

    # ── list_docs ──────────────────────────────────────────────────────────

    def list_docs(self) -> list[dict[str, str]]:
        """Return metadata for every document in the corpus."""
        docs = []
        for f in sorted(self.corpus_dir.iterdir()):
            if f.suffix in (".md", ".txt", ".pdf"):
                docs.append({
                    "doc_id": f.stem,
                    "filename": f.name,
                    "size_bytes": str(f.stat().st_size),
                })
        if self.tracer:
            self.tracer.log("list_docs", {"count": len(docs)})
        return docs

    # ── search ─────────────────────────────────────────────────────────────

    def search(self, query: str, filters: dict | None = None) -> list[dict[str, Any]]:
        """Simple keyword search across all documents."""
        if self.tracer:
            self.tracer.record_search(query)
            self.tracer.log("search", {"query": query})

        results = []
        keywords = query.lower().split()

        for f in sorted(self.corpus_dir.iterdir()):
            if f.suffix not in (".md", ".txt"):
                continue
            content = self._read_doc(f.stem)
            score = sum(content.lower().count(kw) for kw in keywords)
            if score > 0:
                results.append({
                    "doc_id": f.stem,
                    "score": score,
                    "snippet": content[:300],
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:20]

    # ── get_chunk ──────────────────────────────────────────────────────────

    def get_chunk(self, doc_id: str, chunk_id: str) -> str:
        """Return the text of a specific chunk."""
        chunks = self.get_all_chunks(doc_id)
        if self.tracer:
            self.tracer.record_doc_opened(doc_id)
            self.tracer.log("get_chunk", {"doc_id": doc_id, "chunk_id": chunk_id})

        # chunk_id format: "C0", "C1", ...
        try:
            idx = int(chunk_id.lstrip("C"))
        except ValueError:
            return ""
        if 0 <= idx < len(chunks):
            return chunks[idx]
        return ""

    def get_all_chunks(self, doc_id: str) -> list[str]:
        """Return all chunks for a document (cached after first split)."""
        # Check DocIndex first (pre-indexed at upload)
        if self.doc_index and self.doc_index.has_doc(doc_id):
            return self.doc_index.get_chunks(doc_id)

        # Fallback: split on-the-fly + cache
        if doc_id not in self._chunk_cache:
            content = self._read_doc(doc_id)
            self._chunk_cache[doc_id] = self._split_into_chunks(content)
        return self._chunk_cache[doc_id]

    # ── get_figure ─────────────────────────────────────────────────────────

    def get_figure(self, doc_id: str, figure_id: str) -> str | None:
        """Return path to a figure image if it exists."""
        for ext in (".png", ".jpg", ".jpeg", ".svg"):
            p = self.corpus_dir / f"{doc_id}_{figure_id}{ext}"
            if p.exists():
                return str(p)
        return None

    # ── get_table ──────────────────────────────────────────────────────────

    def get_table(self, doc_id: str, table_id: str) -> str:
        """Return table text extracted from the document."""
        content = self._read_doc(doc_id)
        tables = re.findall(
            r"(\|.+\|(?:\n\|.+\|)+)",
            content,
            re.MULTILINE,
        )
        idx = int(table_id.replace("T", "")) if table_id.startswith("T") else 0
        if 0 <= idx < len(tables):
            return tables[idx]
        return ""

    # ── log ────────────────────────────────────────────────────────────────

    def log(self, event: str, payload: dict[str, Any]) -> None:
        """Append event to the trace log."""
        if self.tracer:
            self.tracer.log(event, payload)

    # ── finish ─────────────────────────────────────────────────────────────

    def finish(self, final_json: dict) -> Path:
        """Write final JSON output to disk."""
        out_path = self.output_dir / "final_output.json"
        out_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.tracer:
            self.tracer.log("finish", {"output_path": str(out_path)})
        return out_path

    # ── Internal helpers ───────────────────────────────────────────────────

    def _read_doc(self, doc_id: str) -> str:
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]

        for ext in (".md", ".txt"):
            p = self.corpus_dir / f"{doc_id}{ext}"
            if p.exists():
                text = p.read_text(encoding="utf-8")
                self._doc_cache[doc_id] = text
                return text
        return ""

    def _split_into_chunks(self, content: str, max_chunk: int = 1500) -> list[str]:
        """Split document into chunks by headings or paragraph groups."""
        # Split on markdown headings first
        sections = re.split(r"\n(?=#+\s)", content)
        chunks: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= max_chunk:
                chunks.append(section)
            else:
                # Further split on double newlines
                paras = section.split("\n\n")
                current = ""
                for para in paras:
                    if len(current) + len(para) + 2 > max_chunk and current:
                        chunks.append(current.strip())
                        current = para
                    else:
                        current += "\n\n" + para if current else para
                if current.strip():
                    chunks.append(current.strip())
        return chunks if chunks else [content]
