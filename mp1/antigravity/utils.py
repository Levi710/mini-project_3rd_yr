"""
antigravity/utils.py — Shared utilities for response parsing.
"""

from __future__ import annotations

import re


def strip_think_block(text: str) -> str:
    """Remove DeepSeek-style <think>...</think> blocks from LLM responses."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_from_response(raw: str) -> str:
    """Extract JSON string from an LLM response (handles markdown blocks, think blocks)."""
    # Strip think blocks first
    cleaned = strip_think_block(raw)
    
    # Try markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", cleaned)
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find raw JSON object
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        return brace_match.group(0).strip()
    
    return cleaned.strip()
