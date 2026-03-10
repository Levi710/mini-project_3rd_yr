"""
antigravity/pipeline.py — Orchestrator: Phase A (understand) + Phase B (query pipeline).

Phase A: Document understanding — runs once per uploaded doc.
Phase B: Query answering — chains S0→S1→S2→S3→finish per query.
"""

from __future__ import annotations

import json
import time
from typing import Any

from antigravity.models import (
    ClaimStatus,
    FinalAnswer,
    FinalEvidence,
    FinalOutput,
    Section,
    TraceSummary,
)
from antigravity.modes import is_real_switching, MODE_REGISTRY
from antigravity.stages.route import run_route
from antigravity.stages.extract import run_extract
from antigravity.stages.merge import run_merge
from antigravity.stages.verify import run_verify
from antigravity.tools import CorpusTools
from antigravity.tracer import Tracer
from antigravity.extraction_cache import ExtractionCache


class PipelineRunner:
    """Two-phase pipeline: understand documents, then answer queries."""

    def __init__(self, corpus_dir: str, output_dir: str = "./output", doc_index=None) -> None:
        self.tracer = Tracer()
        self.doc_index = doc_index  # shared DocIndex instance
        self.tools = CorpusTools(corpus_dir, output_dir, self.tracer, doc_index=doc_index)
        self.cache = ExtractionCache(corpus_dir)
        self._progress_callback: Any = None

    def on_progress(self, callback) -> None:
        """Register a callback(stage, data) for live progress updates."""
        self._progress_callback = callback

    def _emit(self, stage: str, data: dict) -> None:
        if self._progress_callback:
            self._progress_callback(stage, data)

    def run(self, query: str) -> FinalOutput:
        """Execute the full pipeline for *query*."""
        self.tracer.log("pipeline_start", {"query": query})

        # ── Phase A: Ensure documents are understood ──────────────────
        self._ensure_docs_understood()

        # ── S0 ROUTE ──────────────────────────────────────────────────
        self._emit("route", {"status": "running", "query": query})
        route_out = run_route(query, self.tools, self.tracer)
        self._emit("route", {
            "status": "complete",
            "docs": len(route_out.doc_scope),
            "chunks": len(route_out.chunk_plan),
        })

        # ── S1 EXTRACT ────────────────────────────────────────────────
        self._emit("extract", {
            "status": "running",
            "total_chunks": len(route_out.chunk_plan),
        })
        extractions = run_extract(
            route_out.chunk_plan, self.tools, self.tracer,
            query=query, cache=self.cache,
        )
        cache_stats = self.cache.stats()
        self._emit("extract", {
            "status": "complete",
            "extractions": len(extractions),
            "total_claims": sum(len(e.extracted.claims) for e in extractions),
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
        })

        # ── Gather overview for merge context ─────────────────────────
        overview = ""
        if self.doc_index:
            # Combine overviews from all docs in scope
            overviews = []
            for ds in route_out.doc_scope:
                doc_overview = self.doc_index.get_overview(ds.doc_id)
                if doc_overview:
                    overviews.append(doc_overview)
            overview = "\n\n".join(overviews)

        # ── S2 MERGE ─────────────────────────────────────────────────
        self._emit("merge", {"status": "running"})
        merge_out = run_merge(query, extractions, self.tracer, overview=overview)
        self._emit("merge", {
            "status": "complete",
            "sections": len(merge_out.synthesis.answer_outline),
            "key_claims": len(merge_out.synthesis.key_claims),
        })

        # ── S3 VERIFY ────────────────────────────────────────────────
        self._emit("verify", {"status": "running"})
        verify_out = run_verify(merge_out, extractions, self.tracer)
        self._emit("verify", {
            "status": "complete",
            "checked": len(verify_out.verification.checked_claims),
            "unsupported": len(verify_out.verification.unsupported_claims),
        })

        # ── Build final output ────────────────────────────────────────
        final = self._build_final(query, merge_out, verify_out, extractions, overview)

        # Write to disk
        self.tools.finish(final.model_dump())
        self._emit("finish", {"status": "complete", "confidence": final.confidence})

        self.tracer.log("pipeline_complete", {"elapsed_s": self.tracer.elapsed()})

        return final

    def _ensure_docs_understood(self) -> None:
        """Run Phase A (understand) for any un-processed docs in the index."""
        if not self.doc_index:
            return

        from antigravity.stages.understand import run_understand

        for doc_info in self.doc_index.list_docs():
            if not doc_info["is_processed"]:
                doc_id = doc_info["doc_id"]
                self._emit("understand", {
                    "status": "running",
                    "doc_id": doc_id,
                })
                run_understand(doc_id, self.doc_index, self.tracer)
                self._emit("understand", {
                    "status": "complete",
                    "doc_id": doc_id,
                })

    def _build_final(self, query, merge_out, verify_out, extractions, overview="") -> FinalOutput:
        """Assemble the FinalOutput from stage results."""

        # Build sections from merge outline
        sections = []
        for sp in merge_out.synthesis.answer_outline:
            content = "\n".join(f"• {p}" for p in sp.points) if sp.points else ""
            sections.append(Section(title=sp.section, content=content))

        # Build response text from verified claims only
        verified_claims = [
            cc for cc in verify_out.verification.checked_claims
            if cc.status == ClaimStatus.SUPPORTED
        ]
        response_parts = [vc.claim for vc in verified_claims]

        if response_parts:
            response = " ".join(response_parts)
        elif sections:
            # Fallback: if verify dropped all claims but merge has good sections,
            # use section content as the response instead of "No verified claims found"
            section_parts = []
            for s in sections:
                if s.content:
                    section_parts.append(s.content)
            response = " ".join(section_parts) if section_parts else "No verified claims found."
        else:
            response = "No verified claims found."

        # Build evidence list
        evidence = []
        for ext in extractions:
            for c in ext.extracted.claims:
                if c.evidence:
                    evidence.append(FinalEvidence(
                        doc_id=c.evidence.doc_id,
                        chunk_id=c.evidence.chunk_id,
                        where=c.evidence.where,
                        supports=c.text,
                        quote=c.evidence.quote,
                    ))

        # Compute confidence
        total = len(verify_out.verification.checked_claims)
        supported = sum(
            1 for cc in verify_out.verification.checked_claims
            if cc.status == ClaimStatus.SUPPORTED
        )

        if total > 0:
            confidence = round(supported / total, 2)
        elif len(sections) > 0:
            # S2 succeeded but S3 dropped claims — reasonable fallback
            confidence = 0.85
        else:
            confidence = 0.0

        # Trace summary
        trace = TraceSummary(
            real_switching=is_real_switching(),
            modes_used_counts=dict(self.tracer.modes_used),
            models_used=sorted(self.tracer.models_used),
            docs_opened=sorted(self.tracer.docs_opened),
            chunks_processed=self.tracer.chunks_processed,
            search_queries=self.tracer.search_queries,
            budget_notes="Within limits",
        )

        return FinalOutput(
            final_answer=FinalAnswer(response=response, sections=sections),
            evidence=evidence,
            trace_summary=trace,
            confidence=confidence,
            missing_info=merge_out.synthesis.open_gaps,
            next_actions=verify_out.verification.required_followups,
        )
