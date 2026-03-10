"""
antigravity/models.py — Pydantic schemas for all 4 pipeline stages + final output.

Every model matches the spec JSON schemas exactly (§6-§7).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TEXT = "text"
    MATH = "math"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    REFERENCES = "references"
    NOISE = "noise"


class ModeName(str, Enum):
    MODE_QUICK = "MODE_QUICK"
    MODE_REASONING = "MODE_REASONING"
    MODE_VISION = "MODE_VISION"


class SupportType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    INFERRED = "inferred"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class Importance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Evidence ───────────────────────────────────────────────────────────────────

class Evidence(BaseModel):
    doc_id: str
    chunk_id: str
    where: str = ""
    quote: str = Field(default="", max_length=200)


# ── S0 ROUTE ───────────────────────────────────────────────────────────────────

class DocScope(BaseModel):
    doc_id: str
    reason: str


class ChunkPlan(BaseModel):
    doc_id: str
    chunk_id: str
    where: str = ""
    chunk_type: ChunkType
    mode: ModeName
    priority: Priority = Priority.MEDIUM
    task: str = ""


class Budgets(BaseModel):
    max_chunks_to_read: int = 200
    max_extractions: int = 25


class RouteOutput(BaseModel):
    stage: str = "route"
    user_query: str
    doc_scope: list[DocScope] = Field(default_factory=list)
    chunk_plan: list[ChunkPlan] = Field(default_factory=list)
    budgets: Budgets = Field(default_factory=Budgets)


# ── S1 EXTRACT ─────────────────────────────────────────────────────────────────

class Claim(BaseModel):
    claim_id: str
    text: str
    importance: Importance = Importance.MEDIUM
    support_type: SupportType = SupportType.EXPLICIT
    numbers: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    evidence: Evidence | None = None


class MathItem(BaseModel):
    expression: str
    interpretation: str = ""
    evidence: Evidence | None = None


class TableItem(BaseModel):
    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    evidence: Evidence | None = None


class FigureItem(BaseModel):
    caption: str = ""
    description: str = ""
    evidence: Evidence | None = None


class CodeItem(BaseModel):
    language: str = ""
    snippet: str = ""
    description: str = ""
    evidence: Evidence | None = None


class ExtractedContent(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    definitions: list[dict] = Field(default_factory=list)
    math: list[MathItem] = Field(default_factory=list)
    table: list[TableItem] = Field(default_factory=list)
    figure: list[FigureItem] = Field(default_factory=list)
    code: list[CodeItem] = Field(default_factory=list)
    chunk_summary: str = ""


class ExtractOutput(BaseModel):
    stage: str = "extract"
    doc_id: str
    chunk_id: str
    chunk_hash: str = ""
    chunk_type: ChunkType
    mode_used: ModeName
    model_id: str = ""
    extracted: ExtractedContent = Field(default_factory=ExtractedContent)


# ── S2 MERGE ───────────────────────────────────────────────────────────────────

class SectionPoint(BaseModel):
    section: str
    points: list[str] = Field(default_factory=list)


class KeyClaim(BaseModel):
    claim: str
    support: ClaimStatus = ClaimStatus.SUPPORTED
    evidence_refs: list[Evidence] = Field(default_factory=list)


class Synthesis(BaseModel):
    answer_outline: list[SectionPoint] = Field(default_factory=list)
    key_claims: list[KeyClaim] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)


class MergeOutput(BaseModel):
    stage: str = "merge"
    synthesis: Synthesis = Field(default_factory=Synthesis)


# ── S3 VERIFY ──────────────────────────────────────────────────────────────────

class CheckedClaim(BaseModel):
    claim: str
    status: ClaimStatus
    evidence: list[Evidence] = Field(default_factory=list)


class Verification(BaseModel):
    checked_claims: list[CheckedClaim] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)


class VerifyOutput(BaseModel):
    stage: str = "verify"
    verification: Verification = Field(default_factory=Verification)


# ── FINAL OUTPUT ───────────────────────────────────────────────────────────────

class Section(BaseModel):
    title: str
    content: str


class FinalAnswer(BaseModel):
    response: str
    sections: list[Section] = Field(default_factory=list)


class FinalEvidence(BaseModel):
    doc_id: str
    chunk_id: str
    where: str = ""
    supports: str = ""
    quote: str = Field(default="", max_length=200)


class TraceSummary(BaseModel):
    real_switching: bool = False
    modes_used_counts: dict[str, int] = Field(default_factory=dict)
    models_used: list[str] = Field(default_factory=list)
    docs_opened: list[str] = Field(default_factory=list)
    chunks_processed: int = 0
    search_queries: list[str] = Field(default_factory=list)
    budget_notes: str = ""


class FinalOutput(BaseModel):
    final_answer: FinalAnswer = Field(default_factory=FinalAnswer)
    evidence: list[FinalEvidence] = Field(default_factory=list)
    trace_summary: TraceSummary = Field(default_factory=TraceSummary)
    confidence: float = 0.0
    missing_info: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────

CHUNK_TYPE_TO_MODE: dict[ChunkType, ModeName] = {
    ChunkType.TEXT: ModeName.MODE_REASONING,
    ChunkType.MATH: ModeName.MODE_REASONING,
    ChunkType.TABLE: ModeName.MODE_REASONING,
    ChunkType.FIGURE: ModeName.MODE_VISION,
    ChunkType.CODE: ModeName.MODE_REASONING,
    ChunkType.REFERENCES: ModeName.MODE_QUICK,
    ChunkType.NOISE: ModeName.MODE_QUICK,
}


def compute_chunk_hash(content: str) -> str:
    """SHA-256 hash of chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
