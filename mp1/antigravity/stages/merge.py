"""
antigravity/stages/merge.py — S2 MERGE stage.

Combines all extraction outputs into a unified synthesis with
answer outline, key claims, and open gaps.
"""

from __future__ import annotations

import json
import re

from antigravity.dispatcher import dispatch
from antigravity.models import (
    Evidence,
    ExtractOutput,
    KeyClaim,
    ClaimStatus,
    MergeOutput,
    SectionPoint,
    Synthesis,
)
from antigravity.tracer import Tracer


_MERGE_PROMPT = """You are a structured synthesis engine. Given the following extracted facts from multiple document chunks, merge them into a coherent answer that DIRECTLY addresses the user's question.

RESPOND ONLY WITH VALID JSON matching this schema:
{{
  "answer_outline": [
    {{"section": "Overview", "points": ["point1", "point2"]}},
    {{"section": "Method", "points": [...]}},
    {{"section": "Results", "points": [...]}},
    {{"section": "Limitations", "points": [...]}}
  ],
  "key_claims": [
    {{
      "claim": "Merged atomic claim",
      "support": "supported|unsupported|uncertain",
      "evidence_doc_ids": ["D1"],
      "evidence_chunk_ids": ["C0"]
    }}
  ],
  "open_gaps": ["Information that is missing or unclear"]
}}

RULES:
- Focus the answer on the user's specific question — do NOT produce a generic document overview
- Only combine facts from the provided extractions — NO new information
- Each key_claim must be traceable to at least one extraction
- Flag any contradictions in open_gaps
- Only include sections relevant to answering the user's question

USER QUERY: {query}

{overview_context}EXTRACTED FACTS:
{extractions_json}
"""


def run_merge(
    query: str,
    extractions: list[ExtractOutput],
    tracer: Tracer,
    overview: str = "",
) -> MergeOutput:
    """S2 — Merge: combine all extraction outputs into a query-focused synthesis."""
    tracer.log("stage_start", {"stage": "merge", "extraction_count": len(extractions)})

    # Build a compact representation of extractions for the prompt
    compact = []
    for ext in extractions:
        claims_data = []
        for c in ext.extracted.claims:
            claims_data.append({
                "id": c.claim_id,
                "text": c.text[:200],
                "importance": c.importance.value,
                "doc": ext.doc_id,
                "chunk": ext.chunk_id,
                "quote": (c.evidence.quote if c.evidence else "")[:100],
            })

        compact.append({
            "doc_id": ext.doc_id,
            "chunk_id": ext.chunk_id,
            "summary": ext.extracted.chunk_summary[:200] if ext.extracted.chunk_summary else "",
            "claims": claims_data,
        })

    # ── Truncate to fit context window ──────────────────────────────
    # Sort extractions by total claim importance (critical/high first)
    IMPORTANCE_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    def _score(entry):
        return sum(IMPORTANCE_RANK.get(c.get("importance", "low"), 1) for c in entry["claims"])

    compact.sort(key=_score, reverse=True)

    # Cap at 20 extractions to stay within context budget
    MAX_MERGE_EXTRACTIONS = 20
    if len(compact) > MAX_MERGE_EXTRACTIONS:
        tracer.log("merge_truncated", {
            "original": len(compact),
            "kept": MAX_MERGE_EXTRACTIONS,
        })
        compact = compact[:MAX_MERGE_EXTRACTIONS]

    extractions_json = json.dumps(compact, indent=1, ensure_ascii=False)

    # Extra safety: hard-cap JSON payload at 16K chars
    if len(extractions_json) > 16000:
        extractions_json = extractions_json[:16000] + "\n... (truncated)"

    # Build overview context — this is the model's "understanding" of the doc,
    # not content to copy. It helps the model navigate the extractions.
    overview_context = ""
    if overview:
        overview_context = (
            "YOUR UNDERSTANDING OF THE DOCUMENT (use this as background knowledge "
            "to guide your answer — do NOT copy from this, only use facts from the "
            "EXTRACTED FACTS below):\n"
            f"{overview[:2000]}\n\n"
        )

    prompt = _MERGE_PROMPT.format(
        query=query,
        overview_context=overview_context,
        extractions_json=extractions_json,
    )

    raw = dispatch("MODE_REASONING", prompt, tracer=tracer)

    result = _parse_merge(raw)

    tracer.log("stage_complete", {
        "stage": "merge",
        "sections": len(result.synthesis.answer_outline),
        "key_claims": len(result.synthesis.key_claims),
        "open_gaps": len(result.synthesis.open_gaps),
    })

    return result


def _parse_merge(raw: str) -> MergeOutput:
    """Parse the LLM merge response into MergeOutput."""
    from antigravity.utils import extract_json_from_response
    json_str = extract_json_from_response(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return MergeOutput(synthesis=Synthesis(
            answer_outline=[SectionPoint(section="Summary", points=[raw[:500]])],
        ))

    # Parse answer outline
    outline = []
    for sec in data.get("answer_outline", []):
        outline.append(SectionPoint(
            section=sec.get("section", ""),
            points=sec.get("points", []),
        ))

    # Parse key claims with evidence references
    key_claims = []
    for kc in data.get("key_claims", []):
        evidence_refs = []
        doc_ids = kc.get("evidence_doc_ids") or []
        chunk_ids = kc.get("evidence_chunk_ids") or []
        for d, c in zip(doc_ids, chunk_ids):
            evidence_refs.append(Evidence(doc_id=d or "", chunk_id=c or ""))

        support_str = kc.get("support", "supported")
        try:
            support = ClaimStatus(support_str.lower())
        except ValueError:
            support = ClaimStatus.SUPPORTED

        key_claims.append(KeyClaim(
            claim=kc.get("claim", ""),
            support=support,
            evidence_refs=evidence_refs,
        ))

    return MergeOutput(
        synthesis=Synthesis(
            answer_outline=outline,
            key_claims=key_claims,
            open_gaps=data.get("open_gaps", []),
        )
    )
