"""
antigravity/stages/extract.py — S1 EXTRACT stage.

Iterates the chunk plan, dispatches each chunk to the correct mode/model,
and produces structured ExtractOutput per chunk.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from antigravity.dispatcher import dispatch
from antigravity.models import (
    Claim,
    ChunkPlan,
    Evidence,
    ExtractedContent,
    ExtractOutput,
    Importance,
    SupportType,
    compute_chunk_hash,
)
from antigravity.tools import CorpusTools
from antigravity.tracer import Tracer
from antigravity.modes import get_mode

if TYPE_CHECKING:
    from antigravity.extraction_cache import ExtractionCache


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are a structured fact extractor. Given the following text chunk, extract factual claims with PRIORITY given to facts relevant to the user's question.

USER QUESTION: {query}

RESPOND ONLY WITH VALID JSON matching this schema:
{{
  "claims": [
    {{
      "claim_id": "{claim_prefix}-CL<N>",
      "text": "Atomic factual claim (one sentence)",
      "importance": "high|medium|low",
      "support_type": "explicit|implicit|inferred",
      "numbers": ["any numeric values mentioned"],
      "entities": ["key terms / named entities"],
      "dependencies": [],
      "quote": "exact quote from the text (<=200 chars)"
    }}
  ],
  "definitions": [
    {{"term": "...", "definition": "..."}}
  ],
  "math": [
    {{"expression": "...", "interpretation": "..."}}
  ],
  "chunk_summary": "2-3 sentence summary of this chunk"
}}

RULES:
- Prioritize extracting claims that directly answer or relate to the user question
- Mark query-relevant claims as "high" importance
- Still extract other notable facts but mark them as "medium" or "low" importance
- Never invent numbers — only extract what is explicitly stated
- Each claim must be atomic (one fact per claim)
- Quotes must be exact substrings from the source text
- If uncertain, state uncertainty explicitly in the claim text
- claim_id format: {claim_prefix}-CL1, {claim_prefix}-CL2, etc.

CHUNK TYPE: {chunk_type}
CHUNK TEXT:
---
{chunk_text}
---
"""


def run_extract(
    chunk_plan: list[ChunkPlan],
    tools: CorpusTools,
    tracer: Tracer,
    query: str = "",
    cache: ExtractionCache | None = None,
) -> list[ExtractOutput]:
    """S1 — Extract: process each planned chunk through the assigned mode."""
    tracer.log("stage_start", {"stage": "extract", "chunk_count": len(chunk_plan)})

    outputs: list[ExtractOutput] = []

    for cp in chunk_plan:
        tracer.log("extract_chunk_start", {
            "doc_id": cp.doc_id,
            "chunk_id": cp.chunk_id,
            "mode": cp.mode.value,
        })

        # Get chunk content
        chunk_text = tools.get_chunk(cp.doc_id, cp.chunk_id)
        if not chunk_text:
            continue

        chunk_hash = compute_chunk_hash(chunk_text)
        # Two-layer cache: query-specific key + base key
        query_cache_key = compute_chunk_hash(chunk_text + "||" + query) if query else chunk_hash
        base_cache_key = chunk_hash

        # ── Cache check (Layer 1: query-specific, Layer 2: base) ─────
        if cache:
            # Layer 1: exact query+chunk match
            cached = cache.get(query_cache_key)
            if cached:
                try:
                    output = ExtractOutput(**{k: v for k, v in cached.items() if k != "cached_at"})
                    outputs.append(output)
                    tracer.record_chunk_processed()
                    tracer.log("extract_cache_hit", {
                        "doc_id": cp.doc_id, "chunk_id": cp.chunk_id, "layer": "query",
                    })
                    continue
                except Exception:
                    pass

            # Layer 2: base extraction (same chunk, any query)
            # Use if available — still useful as partial context
            cached_base = cache.get(base_cache_key)
            if cached_base:
                try:
                    output = ExtractOutput(**{k: v for k, v in cached_base.items() if k != "cached_at"})
                    outputs.append(output)
                    tracer.record_chunk_processed()
                    tracer.log("extract_cache_hit", {
                        "doc_id": cp.doc_id, "chunk_id": cp.chunk_id, "layer": "base",
                    })
                    continue
                except Exception:
                    pass

        # ── Fresh LLM extraction ──────────────────────────────────────
        claim_prefix = f"{cp.doc_id}-{cp.chunk_id}"
        mode_cfg = get_mode(cp.mode.value)

        prompt = _EXTRACT_PROMPT.format(
            query=query or "Extract all important facts from this chunk.",
            claim_prefix=claim_prefix,
            chunk_type=cp.chunk_type.value,
            chunk_text=chunk_text,
        )

        raw_response = dispatch(cp.mode.value, prompt, tracer=tracer)
        extracted = _parse_extraction(raw_response, cp.doc_id, cp.chunk_id)

        output = ExtractOutput(
            doc_id=cp.doc_id,
            chunk_id=cp.chunk_id,
            chunk_hash=chunk_hash,
            chunk_type=cp.chunk_type,
            mode_used=cp.mode,
            model_id=mode_cfg.model_id,
            extracted=extracted,
        )

        # Store in both cache layers
        if cache:
            dumped = output.model_dump()
            cache.put(query_cache_key, dumped)      # Layer 1: query-specific
            cache.put(base_cache_key, dumped)        # Layer 2: base (reusable across queries)

        outputs.append(output)
        tracer.record_chunk_processed()

        tracer.log("extract_chunk_complete", {
            "doc_id": cp.doc_id,
            "chunk_id": cp.chunk_id,
            "claims_extracted": len(extracted.claims),
        })

    # Persist cache to disk
    if cache:
        cache.save()

    tracer.log("stage_complete", {
        "stage": "extract",
        "total_extractions": len(outputs),
        "total_claims": sum(len(o.extracted.claims) for o in outputs),
        "cache_hits": cache.hits if cache else 0,
        "cache_misses": cache.misses if cache else 0,
    })

    return outputs


def _parse_extraction(raw: str, doc_id: str, chunk_id: str) -> ExtractedContent:
    """Parse the LLM JSON response into ExtractedContent."""
    from antigravity.utils import extract_json_from_response
    json_str = extract_json_from_response(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: treat entire response as a summary
        return ExtractedContent(chunk_summary=raw[:500])

    claims = []
    for i, c in enumerate(data.get("claims", []), 1):
        claim_id = c.get("claim_id", f"{doc_id}-{chunk_id}-CL{i}")
        evidence = Evidence(
            doc_id=doc_id,
            chunk_id=chunk_id,
            where=f"chunk {chunk_id}",
            quote=str(c.get("quote", ""))[:200],
        )
        claims.append(Claim(
            claim_id=claim_id,
            text=str(c.get("text", "")),
            importance=_safe_enum(Importance, c.get("importance", "medium")),
            support_type=_safe_enum(SupportType, c.get("support_type", "explicit")),
            numbers=[str(n) for n in c.get("numbers", [])],
            entities=[str(e) for e in c.get("entities", [])],
            dependencies=[str(d) for d in c.get("dependencies", [])],
            evidence=evidence,
        ))

    math_items = []
    for m in data.get("math", []):
        from antigravity.models import MathItem
        math_items.append(MathItem(
            expression=m.get("expression", ""),
            interpretation=m.get("interpretation", ""),
        ))

    return ExtractedContent(
        claims=claims,
        definitions=data.get("definitions", []),
        math=math_items,
        chunk_summary=data.get("chunk_summary", ""),
    )


def _safe_enum(enum_cls, value: str):
    """Safely parse an enum value, defaulting to the first member."""
    try:
        return enum_cls(value.lower())
    except (ValueError, AttributeError):
        return list(enum_cls)[0]
