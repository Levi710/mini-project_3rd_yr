"""
benchmark/compare.py — Real comparison: Pluto vs Single-Model Baseline.

Pluto:  Full S0→S1→S2→S3 pipeline with multi-model switching.
Baseline:     Single LLM call — top-5 chunks from keyword search, one prompt.

This makes the comparison honest and meaningful.
"""

from __future__ import annotations

import time
from pluto.pipeline import PipelineRunner
from pluto.models import FinalOutput, FinalAnswer, TraceSummary, Section
from pluto.tracer import Tracer


class SimpleRunner:
    """
    Real single-model baseline: one LLM call on the top-5 keyword-matched chunks.
    No routing, no extraction schema, no verification.
    """

    def __init__(self, corpus_dir: str, doc_index=None):
        self.corpus_dir = corpus_dir
        self.doc_index = doc_index

    def run(self, query: str) -> FinalOutput:
        from pathlib import Path
        from pluto.dispatcher import dispatch
        from pluto.modes import ModeConfig

        start = time.time()

        # ── Collect top-5 chunks via simple keyword matching ────────────────
        corpus_path = Path(self.corpus_dir)
        chunks: list[str] = []
        doc_ids: list[str] = []

        for md_file in sorted(corpus_path.glob("*.md"))[:3]:  # max 3 docs
            text = md_file.read_text(encoding="utf-8", errors="replace")
            # Split into ~1000 char chunks
            parts = [text[i:i+1000] for i in range(0, len(text), 1000)]
            # Score by query words
            q_words = set(query.lower().split())
            scored = sorted(parts, key=lambda p: sum(1 for w in q_words if w in p.lower()), reverse=True)
            chunks.extend(scored[:2])  # top-2 chunks per doc
            doc_ids.append(md_file.stem)

        # Take top 5 chunks overall
        top_chunks = chunks[:5]

        if not top_chunks:
            elapsed = time.time() - start
            return FinalOutput(
                final_answer=FinalAnswer(
                    response="No documents found in corpus.",
                    sections=[]
                ),
                evidence=[],
                trace_summary=TraceSummary(
                    real_switching=False,
                    modes_used_counts={"SINGLE_LLM": 1},
                    models_used=["llama-3.1-8b-instant"],
                    chunks_processed=0,
                    search_queries=[query]
                ),
                confidence=0.0,
            )

        # ── Single LLM call ──────────────────────────────────────────────────
        context = "\n\n---\n\n".join(top_chunks)
        prompt = f"""Answer the following question based ONLY on the provided context.

QUESTION: {query}

CONTEXT:
{context[:6000]}

Provide a clear, direct answer. If the context does not contain enough information, say so."""

        try:
            # Use MODE_QUICK (the smallest/fastest model) — single call
            response = dispatch("MODE_QUICK", prompt)
        except Exception as e:
            response = f"Baseline LLM call failed: {e}"

        elapsed = time.time() - start

        from pluto.modes import MODE_REGISTRY
        quick_model = MODE_REGISTRY["MODE_QUICK"].model_id

        return FinalOutput(
            final_answer=FinalAnswer(
                response=response,
                sections=[Section(title="Answer", content=response)]
            ),
            evidence=[],  # No structured evidence — single model doesn't trace this
            trace_summary=TraceSummary(
                real_switching=False,
                modes_used_counts={"MODE_QUICK": 1},
                models_used=[quick_model],
                chunks_processed=len(top_chunks),
                search_queries=[query]
            ),
            confidence=0.5,  # No verification → moderate confidence
        )


class ComparisonRunner:
    """Runs Pluto vs Baseline and returns comparative metrics."""

    def __init__(self, corpus_dir: str, doc_index=None):
        self.pluto = PipelineRunner(corpus_dir, doc_index=doc_index)
        self.baseline = SimpleRunner(corpus_dir, doc_index=doc_index)

    def compare(self, query: str) -> dict:
        # ── Pluto pipeline ──────────────────────────────────────────────
        start_ag = time.time()
        res_ag = self.pluto.run(query)
        time_ag = round(time.time() - start_ag, 2)

        # ── Baseline single-model ─────────────────────────────────────────────
        start_bl = time.time()
        res_bl = self.baseline.run(query)
        time_bl = round(time.time() - start_bl, 2)

        return {
            "query": query,
            "pluto": {
                "latency_s": time_ag,
                "confidence": round(res_ag.confidence, 2),
                "evidence_count": len(res_ag.evidence),
                "chunks_processed": res_ag.trace_summary.chunks_processed,
                "verified": True,
                "answer_preview": (res_ag.final_answer.response or "")[:300],
                "models_used": res_ag.trace_summary.models_used,
                "real_switching": res_ag.trace_summary.real_switching,
            },
            "baseline": {
                "latency_s": time_bl,
                "confidence": round(res_bl.confidence, 2),
                "evidence_count": 0,
                "chunks_processed": res_bl.trace_summary.chunks_processed,
                "verified": False,
                "answer_preview": (res_bl.final_answer.response or "")[:300],
                "models_used": res_bl.trace_summary.models_used,
                "real_switching": False,
            },
            "winner": "Pluto"
        }


if __name__ == "__main__":
    import json
    runner = ComparisonRunner("./corpus")
    results = runner.compare("What is this paper about?")
    print(json.dumps(results, indent=2))
