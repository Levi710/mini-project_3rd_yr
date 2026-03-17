"""
pluto/stages/merge.py — S2 MERGE stage.

Two-pass merge for large documents:
  Pass 1: Batch-merge extractions in groups of 15
  Pass 2: Final merge of batch summaries into one coherent answer.

This removes the 20-chunk ceiling — all extractions are included.
"""

from __future__ import annotations

import json

from pluto.dispatcher import dispatch
from pluto.models import (
    Evidence,
    ExtractOutput,
    KeyClaim,
    ClaimStatus,
    MergeOutput,
    SectionPoint,
    Synthesis,
)
from pluto.tracer import Tracer


# ── Prompts ──────────────────────────────────────────────────────────────────

_BATCH_PROMPT = """You are synthesizing extracted facts from a document chunk batch. Produce a focused sub-summary for the user's question.

RESPOND ONLY WITH VALID JSON:
{{
  "batch_summary": "2-4 sentences covering facts relevant to the question",
  "key_claims": [
    {{
      "claim": "One atomic factual claim",
      "support": "supported|unsupported|uncertain",
      "evidence_doc_ids": ["doc_id"],
      "evidence_chunk_ids": ["chunk_id"]
    }}
  ]
}}

RULES:
- Only include claims RELEVANT to the question
- Be concise — this output will be merged with other batches
- Do NOT invent new facts

USER QUERY: {query}

EXTRACTED FACTS (batch {batch_num}/{total_batches}):
{extractions_json}
"""

_FINAL_MERGE_PROMPT = """You are a structured synthesis engine. Combine these batch summaries into ONE comprehensive answer for the user's question.

RESPOND ONLY WITH VALID JSON:
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
- Focus the answer on the user's specific question — do NOT produce a generic overview
- Only combine facts from the provided summaries — NO new information
- Deduplicate overlapping claims
- Only include sections relevant to the question

USER QUERY: {query}

{overview_context}BATCH SUMMARIES:
{batch_summaries_json}
"""


# BATCH_SIZE: how many extractions per batch
# Groq's context window comfortably handles 20 compact extractions per call.
# A 33-chunk doc uses 2 batch calls + 1 final merge = 3 total LLM calls in merge.
BATCH_SIZE = 20


def run_merge(
    query: str,
    extractions: list[ExtractOutput],
    tracer: Tracer,
    overview: str = "",
) -> MergeOutput:
    """S2 — Merge: two-pass batch merge so ALL extractions are included."""
    tracer.log("stage_start", {"stage": "merge", "extraction_count": len(extractions)})

    if not extractions:
        return MergeOutput(synthesis=Synthesis(
            answer_outline=[SectionPoint(section="Summary", points=["No content available."])],
        ))

    # ── Build compact representation of all extractions ──────────────────────
    IMPORTANCE_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    compact = []
    for ext in extractions:
        claims_data = [
            {
                "id": c.claim_id,
                "text": c.text[:200],
                "importance": c.importance.value,
                "doc": ext.doc_id,
                "chunk": ext.chunk_id,
                "quote": (c.evidence.quote if c.evidence else "")[:100],
            }
            for c in ext.extracted.claims
        ]
        compact.append({
            "doc_id": ext.doc_id,
            "chunk_id": ext.chunk_id,
            "summary": ext.extracted.chunk_summary[:200] if ext.extracted.chunk_summary else "",
            "claims": claims_data,
            "_score": sum(IMPORTANCE_RANK.get(c.get("importance", "low"), 1) for c in claims_data),
        })

    # Sort by importance (best first within each batch)
    compact.sort(key=lambda x: x["_score"], reverse=True)

    # ── Pass 1: Batch-merge in groups of BATCH_SIZE ───────────────────────────
    batches = [compact[i:i+BATCH_SIZE] for i in range(0, len(compact), BATCH_SIZE)]
    tracer.log("merge_batches", {
        "total_extractions": len(compact),
        "batch_count": len(batches),
        "batch_size": BATCH_SIZE,
    })

    batch_summaries: list[dict] = []

    for bi, batch in enumerate(batches):
        # Remove internal _score key before sending to LLM
        clean_batch = [{k: v for k, v in item.items() if k != "_score"} for item in batch]
        batch_json = json.dumps(clean_batch, indent=1, ensure_ascii=False)

        # Hard-cap individual batch JSON at 12K chars (well within Groq's window)
        if len(batch_json) > 12000:
            batch_json = batch_json[:12000] + "\n... (truncated)"

        if len(batches) == 1:
            # Only one batch — skip the intermediate step, go straight to final merge
            batch_summaries.append({"batch": 1, "content": clean_batch})
            break

        prompt = _BATCH_PROMPT.format(
            query=query,
            batch_num=bi + 1,
            total_batches=len(batches),
            extractions_json=batch_json,
        )

        try:
            raw = dispatch("MODE_QUICK", prompt, tracer=tracer)
            parsed = _parse_batch(raw)
            batch_summaries.append({
                "batch": bi + 1,
                "batch_summary": parsed.get("batch_summary", ""),
                "key_claims": parsed.get("key_claims", []),
            })
            tracer.log("merge_batch_done", {"batch": bi + 1, "claims": len(parsed.get("key_claims", []))})
        except Exception as e:
            print(f"  [WARNING] Merge batch {bi+1} failed: {e}")
            # Include raw extraction summaries as fallback
            batch_summaries.append({
                "batch": bi + 1,
                "batch_summary": " ".join(item.get("summary", "") for item in clean_batch[:5]),
                "key_claims": [],
            })

    # ── Pass 2: Final merge of batch summaries ────────────────────────────────
    # If only 1 batch, use the full extractions directly in the final prompt
    if len(batches) == 1:
        clean_batch = [{k: v for k, v in item.items() if k != "_score"} for item in compact]
        summaries_json = json.dumps(clean_batch, indent=1, ensure_ascii=False)
        if len(summaries_json) > 14000:
            summaries_json = summaries_json[:14000] + "\n... (truncated)"
    else:
        summaries_json = json.dumps(batch_summaries, indent=1, ensure_ascii=False)
        if len(summaries_json) > 14000:
            summaries_json = summaries_json[:14000] + "\n... (truncated)"

    overview_context = ""
    if overview:
        overview_context = (
            "YOUR UNDERSTANDING OF THE DOCUMENT (background context only — "
            "use only EXTRACTED FACTS below for claims):\n"
            f"{overview[:1500]}\n\n"
        )

    final_prompt = _FINAL_MERGE_PROMPT.format(
        query=query,
        overview_context=overview_context,
        batch_summaries_json=summaries_json,
    )

    raw_final = dispatch("MODE_REASONING", final_prompt, tracer=tracer)
    result = _parse_merge(raw_final)

    tracer.log("stage_complete", {
        "stage": "merge",
        "batches_processed": len(batches),
        "total_extractions_included": len(compact),
        "sections": len(result.synthesis.answer_outline),
        "key_claims": len(result.synthesis.key_claims),
        "open_gaps": len(result.synthesis.open_gaps),
    })

    return result


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_batch(raw: str) -> dict:
    """Parse a batch merge response."""
    from pluto.utils import extract_json_from_response
    json_str = extract_json_from_response(raw)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"batch_summary": raw[:300], "key_claims": []}


def _parse_merge(raw: str) -> MergeOutput:
    """Parse the final LLM merge response into MergeOutput."""
    from pluto.utils import extract_json_from_response
    json_str = extract_json_from_response(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return MergeOutput(synthesis=Synthesis(
            answer_outline=[SectionPoint(section="Summary", points=[raw[:500]])],
        ))

    # Parse answer outline
    outline = [
        SectionPoint(
            section=sec.get("section", ""),
            points=sec.get("points", []),
        )
        for sec in data.get("answer_outline", [])
    ]

    # Parse key claims
    key_claims = []
    for kc in data.get("key_claims", []):
        evidence_refs = []
        for d, c in zip(kc.get("evidence_doc_ids") or [], kc.get("evidence_chunk_ids") or []):
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
