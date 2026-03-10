"""
antigravity/stages/route.py — S0 ROUTE stage (Phase B).

Uses pre-indexed chunks from DocIndex. Scores each chunk by query relevance,
selects the top N most relevant chunks, and builds a prioritized chunk plan.
"""

from __future__ import annotations

import re

from antigravity.chunker import classify_chunk
from antigravity.models import (
    Budgets,
    ChunkPlan,
    ChunkType,
    DocScope,
    ModeName,
    Priority,
    RouteOutput,
    CHUNK_TYPE_TO_MODE,
)
from antigravity.tools import CorpusTools
from antigravity.tracer import Tracer


def run_route(query: str, tools: CorpusTools, tracer: Tracer) -> RouteOutput:
    """S0 — Route: score chunks by query relevance, select top N, build plan."""
    tracer.log("stage_start", {"stage": "route", "query": query})

    # 1. List all available docs
    docs = tools.list_docs()

    # 2. Search for relevant docs
    search_results = tools.search(query)

    # 3. Build doc scope from search hits (or all docs if no hits)
    doc_scope: list[DocScope] = []
    target_doc_ids: list[str] = []

    if search_results:
        for sr in search_results[:5]:
            doc_scope.append(DocScope(
                doc_id=sr["doc_id"],
                reason=f"Matched query with score {sr['score']}",
            ))
            target_doc_ids.append(sr["doc_id"])
    else:
        for d in docs[:5]:
            doc_scope.append(DocScope(
                doc_id=d["doc_id"],
                reason="No search hits — including for broad coverage",
            ))
            target_doc_ids.append(d["doc_id"])

    # 4. Two-pass chunk selection with DYNAMIC budget
    budgets = Budgets()
    # Adjust extraction budget based on query complexity
    budgets.max_extractions = _dynamic_budget(query)
    all_scored: list[dict] = []

    for doc_id in target_doc_ids:
        # Get all chunks (from DocIndex cache or on-the-fly with cache)
        chunks = tools.get_all_chunks(doc_id)

        # Get the chunk topic map from Phase A understanding (if available)
        chunk_topic_map: dict[str, list[str]] = {}
        if tools.doc_index:
            chunk_topic_map = tools.doc_index.get_chunk_topics(doc_id)

        for ci, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            chunk_id = f"C{ci}"
            # Get topic tags from Phase A understanding
            chunk_topics = chunk_topic_map.get(chunk_id, [])

            # Score by query relevance (fast, no LLM call)
            score = _score_relevance(query, chunk_text, ci, chunk_topics)

            all_scored.append({
                "doc_id": doc_id,
                "chunk_index": ci,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "relevance_score": score,
            })

    # Sort by relevance score (highest first)
    all_scored.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Take top N most relevant chunks
    top_chunks = all_scored[:budgets.max_extractions]

    tracer.log("route_scoring", {
        "total_chunks_scanned": len(all_scored),
        "top_chunks_selected": len(top_chunks),
        "top_scores": [round(c["relevance_score"], 3) for c in top_chunks[:5]],
    })

    # 5. Classify and build chunk plan from selected chunks
    chunk_plan: list[ChunkPlan] = []
    for item in top_chunks:
        chunk_type = classify_chunk(item["chunk_text"])

        # Skip noise
        if chunk_type == ChunkType.NOISE:
            continue

        mode = CHUNK_TYPE_TO_MODE[chunk_type]

        # Priority based on relevance score
        if item["relevance_score"] >= 0.5:
            priority = Priority.HIGH
        elif item["relevance_score"] >= 0.2:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        chunk_plan.append(ChunkPlan(
            doc_id=item["doc_id"],
            chunk_id=item["chunk_id"],
            where=f"chunk {item['chunk_index']}",
            chunk_type=chunk_type,
            mode=mode,
            priority=priority,
            task=f"Extract from {chunk_type.value} chunk (relevance: {item['relevance_score']:.2f})",
        ))

    # Sort by priority (high first)
    priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    chunk_plan.sort(key=lambda cp: priority_order[cp.priority])

    result = RouteOutput(
        user_query=query,
        doc_scope=doc_scope,
        chunk_plan=chunk_plan,
        budgets=budgets,
    )

    tracer.log("stage_complete", {
        "stage": "route",
        "docs_found": len(doc_scope),
        "chunks_planned": len(chunk_plan),
        "total_scanned": len(all_scored),
    })

    return result


# ── Synonym map for query expansion ──────────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "architecture": ["model", "network", "structure", "framework", "design"],
    "model": ["architecture", "network", "method"],
    "accuracy": ["precision", "performance", "miou", "result"],
    "performance": ["accuracy", "speed", "fps", "result", "efficiency"],
    "dataset": ["data", "benchmark", "corpus", "training"],
    "method": ["approach", "technique", "algorithm", "procedure"],
    "result": ["finding", "outcome", "performance", "accuracy"],
    "limitation": ["weakness", "drawback", "constraint", "challenge"],
    "conclusion": ["summary", "finding", "result"],
    "training": ["learning", "optimization", "fitting"],
    "encoder": ["backbone", "feature extractor"],
    "decoder": ["segmentation head", "output"],
    "loss": ["objective", "cost function", "criterion"],
    "speed": ["fps", "latency", "inference time", "real-time"],
    "comparison": ["benchmark", "versus", "baseline"],
}


def _score_relevance(
    query: str,
    chunk_text: str,
    chunk_index: int,
    chunk_topics: list[str] | None = None,
) -> float:
    """
    Smart relevance scoring using both keywords AND the model's understanding.

    Uses:
      1. Keyword overlap (fast, always available)
      2. Topic overlap from Phase A understanding (when available)
    """
    query_lower = query.lower()
    chunk_lower = chunk_text.lower()

    # Tokenize query into meaningful words (skip short stopwords)
    query_words = [w for w in re.split(r'\W+', query_lower) if len(w) > 2]
    if not query_words:
        return 0.1

    # Expand with common synonyms for better matching
    expanded = set(query_words)
    for w in query_words:
        expanded.update(_SYNONYMS.get(w, []))
    expanded_words = list(expanded)

    # ── Layer 1: Keyword overlap ─────────────────────────────────────
    word_hits = sum(1 for w in expanded_words if w in chunk_lower)
    score = word_hits / len(query_words)  # normalize by original query length

    # Bonus for exact phrase match
    if query_lower in chunk_lower:
        score += 0.5

    # Bonus for multiple keyword hits
    if word_hits >= 2:
        score += 0.1 * (word_hits - 1)

    # Small bonus for early chunks (abstract/intro)
    if chunk_index < 3:
        score += 0.05

    # ── Layer 2: Topic overlap from understanding ────────────────────
    # This is the "student knows which chapter to flip to" part
    if chunk_topics:
        topic_hits = sum(
            1 for w in query_words
            if any(w in topic for topic in chunk_topics)
        )
        if topic_hits > 0:
            score += 0.3 * (topic_hits / len(query_words))

        # Boost results/method chunks for specific question types
        query_str = " ".join(query_words)
        if any(role in chunk_topics for role in ["results", "discussion"]):
            if any(kw in query_str for kw in ["result", "accuracy", "performance", "dataset", "test", "experiment", "achieve"]):
                score += 0.25
        if any(role in chunk_topics for role in ["method", "methodology"]):
            if any(kw in query_str for kw in ["method", "architecture", "model", "approach", "design", "how", "technique"]):
                score += 0.25

    return score


def _dynamic_budget(query: str) -> int:
    """
    Determine extraction budget based on query complexity.

    Simple queries ("What year?") → 8 chunks
    Medium queries ("What is the architecture?") → 15 chunks
    Complex queries ("Compare methods and results across...") → 25 chunks
    """
    words = query.split()
    word_count = len(words)

    # Multi-part or comparative questions
    complex_signals = ["compare", "versus", "difference", "relationship",
                       "how does", "explain in detail", "all", "every",
                       "comprehensive", "summarize the entire"]
    if any(sig in query.lower() for sig in complex_signals):
        return 25

    # Simple factual lookups
    simple_signals = ["what year", "who wrote", "when was", "what is the title",
                      "how many pages", "what journal", "published in"]
    if any(sig in query.lower() for sig in simple_signals):
        return 8

    # Default: scale with query length
    if word_count <= 5:
        return 10
    elif word_count <= 12:
        return 15
    else:
        return 25
