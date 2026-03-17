"""
pluto/stages/verify.py — S3 VERIFY stage.

Cross-checks merged claims against source evidence.
Unsupported claims are removed or downgraded.
"""

from __future__ import annotations

import json
import re

from pluto.dispatcher import dispatch
from pluto.models import (
    CheckedClaim,
    ClaimStatus,
    Evidence,
    MergeOutput,
    ExtractOutput,
    Verification,
    VerifyOutput,
)
from pluto.tracer import Tracer


_VERIFY_PROMPT = """You are an evidence verification engine. Check each claim below against the source evidence provided.

For EACH claim, determine if it is:
- "supported": directly backed by evidence quotes
- "unsupported": no evidence found
- "uncertain": partial or weak evidence

RESPOND ONLY WITH VALID JSON:
{{
  "checked_claims": [
    {{
      "claim": "The claim text",
      "status": "supported|unsupported|uncertain",
      "evidence_doc_id": "D1",
      "evidence_chunk_id": "C0",
      "reason": "Brief explanation"
    }}
  ],
  "unsupported_claims": ["claims that should be removed"],
  "required_followups": ["questions that need further investigation"]
}}

RULES:
- Unsupported claims MUST be flagged for removal
- Do not accept claims without direct evidence
- Flag contradictions between claims

CLAIMS TO VERIFY:
{claims_json}

SOURCE EVIDENCE:
{evidence_json}
"""


def run_verify(
    merge_output: MergeOutput,
    extractions: list[ExtractOutput],
    tracer: Tracer,
) -> VerifyOutput:
    """S3 — Verify: cross-check merged claims against extraction evidence."""
    tracer.log("stage_start", {"stage": "verify"})

    # Collect all claims from merge
    claims_data = []
    for kc in merge_output.synthesis.key_claims:
        claims_data.append({
            "claim": kc.claim,
            "current_status": kc.support.value,
            "evidence_refs": [
                {"doc_id": e.doc_id, "chunk_id": e.chunk_id}
                for e in kc.evidence_refs
            ],
        })

    # Collect all source evidence from extractions (capped for context)
    evidence_data = []
    for ext in extractions:
        for c in ext.extracted.claims:
            if c.evidence:
                evidence_data.append({
                    "claim_id": c.claim_id,
                    "text": c.text[:150],
                    "doc_id": c.evidence.doc_id,
                    "chunk_id": c.evidence.chunk_id,
                    "quote": c.evidence.quote[:100] if c.evidence.quote else "",
                })

    # Cap evidence to prevent context overflow
    if len(evidence_data) > 30:
        evidence_data = evidence_data[:30]

    claims_str = json.dumps(claims_data, indent=1, ensure_ascii=False)
    evidence_str = json.dumps(evidence_data, indent=1, ensure_ascii=False)

    # Hard-cap payloads
    if len(evidence_str) > 10000:
        evidence_str = evidence_str[:10000] + "\n... (truncated)"

    prompt = _VERIFY_PROMPT.format(
        claims_json=claims_str,
        evidence_json=evidence_str,
    )

    raw = dispatch("MODE_QUICK", prompt, tracer=tracer)
    
    # DEBUG: Print exact LLM output to see why checked_claims is empty
    print(f"\n[DEBUG VERIFY RAW]\n{raw}\n[/DEBUG VERIFY RAW]\n")

    result = _parse_verify(raw)

    tracer.log("stage_complete", {
        "stage": "verify",
        "checked": len(result.verification.checked_claims),
        "unsupported": len(result.verification.unsupported_claims),
        "followups": len(result.verification.required_followups),
    })

    return result


def _parse_verify(raw: str) -> VerifyOutput:
    """Parse the LLM verification response."""
    from pluto.utils import extract_json_from_response
    json_str = extract_json_from_response(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return VerifyOutput(verification=Verification())

    checked = []
    for cc in data.get("checked_claims", []):
        status_str = cc.get("status", "uncertain")
        try:
            status = ClaimStatus(status_str.lower())
        except ValueError:
            status = ClaimStatus.UNCERTAIN

        evidence = []
        if cc.get("evidence_doc_id"):
            evidence.append(Evidence(
                doc_id=cc["evidence_doc_id"],
                chunk_id=cc.get("evidence_chunk_id") or "",
            ))

        checked.append(CheckedClaim(
            claim=cc.get("claim", ""),
            status=status,
            evidence=evidence,
        ))

    return VerifyOutput(
        verification=Verification(
            checked_claims=checked,
            unsupported_claims=data.get("unsupported_claims", []),
            required_followups=data.get("required_followups", []),
        )
    )
