"""
pluto/stages/understand.py — Phase A: Document Understanding.

Runs once per uploaded document (before any query).
Like a student reading a book — builds a mental map of what's where,
so future questions can be answered by navigating to the right sections.

Stores:
  - overview: structured understanding (title, topics, section map)
  - chunk_topics: per-chunk topic tags for intelligent chunk selection

MULTI-PASS: For large docs, sends chunks in batches so ALL chunks
get topic tags, not just the first 20K chars.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pluto.dispatcher import dispatch
from pluto.tracer import Tracer

if TYPE_CHECKING:
    from pluto.doc_index import DocIndex


# ── Full-doc overview prompt (first pass) ────────────────────────────────

_OVERVIEW_PROMPT = """You are a student reading a research document for the first time. Your goal is to BUILD YOUR UNDERSTANDING — create a mental map of what this document covers and WHERE each topic is discussed.

RESPOND ONLY WITH VALID JSON matching this schema:
{{
  "title": "Document title or inferred title",
  "summary": "3-5 sentence overall summary — what is this document about?",
  "topics": ["topic1", "topic2", "..."],
  "section_map": [
    {{"heading": "Section name", "covers": "What this section discusses", "chunk_ids": ["C0", "C1"]}}
  ],
  "chunk_topics": [
    {{"chunk_id": "C0", "topics": ["topic1", "topic2"], "role": "introduction|method|results|discussion|references|other"}}
  ],
  "key_concepts": ["concept1", "concept2", "..."],
  "methodology": "Brief description of methods/approach (if applicable)",
  "datasets": "Datasets mentioned (if applicable)",
  "conclusions": "Key conclusions of the document"
}}

RULES:
- For chunk_topics, tag EVERY chunk shown with its topics and role
- The chunk_ids in section_map should reference which chunks belong to each section
- This is your study notes — be thorough about WHAT is WHERE
- Do NOT fabricate — only summarize what is actually present
- The topics should be specific enough to match future questions

DOCUMENT CONTENT:
---
{doc_content}
---
"""

# ── Continuation prompt (subsequent passes for remaining chunks) ──────

_CONTINUE_PROMPT = """You already read the first part of a document. Here is your understanding so far:

{existing_understanding}

Now read the REMAINING chunks and update your understanding.

RESPOND ONLY WITH VALID JSON matching this schema:
{{
  "chunk_topics": [
    {{"chunk_id": "C15", "topics": ["topic1", "topic2"], "role": "introduction|method|results|discussion|references|other"}}
  ],
  "additional_topics": ["any new topics found"],
  "additional_findings": "Any new key findings from these chunks"
}}

RULES:
- Tag EVERY chunk shown with its topics and role
- Only add to your understanding — don't repeat what you already know
- Focus on what's NEW in these chunks

REMAINING CHUNKS:
---
{doc_content}
---
"""


def run_understand(
    doc_id: str,
    doc_index: DocIndex,
    tracer: Tracer,
) -> str:
    """
    Phase A — Understand: read the document like a student building comprehension.

    Uses PARALLEL MULTI-PASS for large documents:
      Pass 1: Read first batch, build initial overview + chunk tags
      Pass 2+: Read remaining batches in parallel, tag remaining chunks
    """
    import concurrent.futures
    
    tracer.log("stage_start", {"stage": "understand", "doc_id": doc_id})

    chunks = doc_index.get_chunks(doc_id)
    if not chunks:
        tracer.log("understand_skip", {"reason": "no chunks", "doc_id": doc_id})
        return ""

    # Split chunks into batches (max 4000 chars for Mistral stability)
    batches = _split_into_batches(chunks, max_chars=4000)

    tracer.log("understand_batches", {
        "total_chunks": len(chunks),
        "batch_count": len(batches),
        "batch_sizes": [len(b) for b in batches],
    })

    # ── Pass 1: Full overview from first batch (Sequential) ──────────────────
    # We do this first to establish context for subsequent batches
    first_batch_content = _format_batch(batches[0])
    prompt = _OVERVIEW_PROMPT.format(doc_content=first_batch_content)

    print(f"  [PHASE A] Reading first batch of {doc_id} to build overview...")
    raw = dispatch("MODE_VISION", prompt, tracer=tracer)
    overview_text, chunk_topic_map = _parse_overview(raw)

    # ── Pass 2+: Tag remaining chunks (Parallel) ──────────────────────
    if len(batches) > 1:
        print(f"  [PHASE A] Processing remaining {len(batches)-1} batches in parallel...")
        
        # Compact version of existing understanding for context
        existing = overview_text[:1500] if len(overview_text) > 1500 else overview_text

        def process_batch(idx):
            batch_content = _format_batch(batches[idx])
            cont_prompt = _CONTINUE_PROMPT.format(
                existing_understanding=existing,
                doc_content=batch_content,
            )
            try:
                cont_raw = dispatch("MODE_VISION", cont_prompt, tracer=tracer)
                return _parse_continuation(cont_raw)
            except Exception as e:
                print(f"  [WARNING] Batch {idx} failed: {e}")
                return {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_batch = {executor.submit(process_batch, i): i for i in range(1, len(batches))}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    extra_topics = future.result()
                    chunk_topic_map.update(extra_topics)
                    tracer.log("understand_continue", {
                        "batch": batch_idx + 1,
                        "new_chunks_tagged": len(extra_topics),
                    })
                except Exception as e:
                    print(f"  [ERROR] Processing batch {batch_idx+1} result: {e}")

    # Store in doc index
    doc_index.set_overview(doc_id, overview_text)
    doc_index.set_chunk_topics(doc_id, chunk_topic_map)

    tracer.log("stage_complete", {
        "stage": "understand",
        "doc_id": doc_id,
        "overview_length": len(overview_text),
        "chunks_mapped": len(chunk_topic_map),
        "total_chunks": len(chunks),
        "passes": len(batches),
    })

    return overview_text


def _split_into_batches(chunks: list[str], max_chars: int) -> list[list[tuple[int, str]]]:
    """Split chunks into batches, each fitting within max_chars."""
    batches: list[list[tuple[int, str]]] = []
    current_batch: list[tuple[int, str]] = []
    current_size = 0

    for i, chunk in enumerate(chunks):
        entry_size = len(f"[Chunk C{i}]\n{chunk}\n")
        if current_size + entry_size > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append((i, chunk))
        current_size += entry_size

    if current_batch:
        batches.append(current_batch)

    return batches if batches else [[(0, "")]]


def _format_batch(batch: list[tuple[int, str]]) -> str:
    """Format a batch of (index, chunk_text) pairs for the LLM prompt."""
    parts = []
    for i, chunk_text in batch:
        parts.append(f"[Chunk C{i}]\n{chunk_text}")
    return "\n\n".join(parts)


def _parse_overview(raw: str) -> tuple[str, dict[str, list[str]]]:
    """
    Parse the LLM overview response.

    Returns:
        overview_text: readable understanding string
        chunk_topic_map: {chunk_id: [topic1, topic2, ...]} for smart routing
    """
    from pluto.utils import extract_json_from_response

    json_str = extract_json_from_response(raw)
    chunk_topic_map: dict[str, list[str]] = {}

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return raw[:2000], {}

    # Extract chunk-level topic map
    for ct in data.get("chunk_topics", []):
        cid = ct.get("chunk_id", "")
        topics = ct.get("topics", [])
        role = ct.get("role", "other")
        if cid:
            chunk_topic_map[cid] = [str(t).lower() for t in topics] + [role.lower()]

    # Build a structured overview string
    parts = []

    title = data.get("title", "")
    if title:
        parts.append(f"TITLE: {title}")

    summary = data.get("summary", "")
    if summary:
        parts.append(f"UNDERSTANDING: {summary}")

    topics = data.get("topics", [])
    if topics:
        parts.append(f"TOPICS COVERED: {', '.join(str(t) for t in topics)}")

    section_map = data.get("section_map", [])
    if section_map:
        sec_lines = []
        for s in section_map:
            heading = s.get("heading", "")
            covers = s.get("covers", "")
            cids = s.get("chunk_ids", [])
            sec_lines.append(f"  - {heading} [{', '.join(cids)}]: {covers}")
        parts.append("SECTION MAP:\n" + "\n".join(sec_lines))

    concepts = data.get("key_concepts", [])
    if concepts:
        parts.append(f"KEY CONCEPTS: {', '.join(str(c) for c in concepts)}")

    methodology = data.get("methodology", "")
    if methodology:
        parts.append(f"METHODOLOGY: {methodology}")

    datasets = data.get("datasets", "")
    if datasets:
        parts.append(f"DATASETS: {datasets}")

    conclusions = data.get("conclusions", "")
    if conclusions:
        parts.append(f"CONCLUSIONS: {conclusions}")

    overview_text = "\n\n".join(parts) if parts else raw[:2000]
    return overview_text, chunk_topic_map


def _parse_continuation(raw: str) -> dict[str, list[str]]:
    """Parse the continuation response — extract chunk_topics only."""
    from pluto.utils import extract_json_from_response

    json_str = extract_json_from_response(raw)
    chunk_topic_map: dict[str, list[str]] = {}

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    for ct in data.get("chunk_topics", []):
        cid = ct.get("chunk_id", "")
        topics = ct.get("topics", [])
        role = ct.get("role", "other")
        if cid:
            chunk_topic_map[cid] = [str(t).lower() for t in topics] + [role.lower()]

    return chunk_topic_map
