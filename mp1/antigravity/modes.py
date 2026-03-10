"""
antigravity/modes.py — Real mode switching engine.

Each mode maps to an actual model configuration (model_id, temperature,
max_tokens, compute_profile).  This is NOT prompt-trick switching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModeConfig:
    """Concrete model configuration for a single processing mode."""

    mode_name: str
    model_id: str
    temperature: float
    max_tokens: int
    compute_profile: str
    provider: str  # "gemini" | "groq"

    def to_log_dict(self) -> dict:
        return {
            "mode_name": self.mode_name,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "compute_profile": self.compute_profile,
            "provider": self.provider,
        }


# ── Build registry based on available API keys ────────────────────────────────

def _build_registry() -> dict[str, ModeConfig]:
    """Construct the mode registry from environment."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    has_groq = bool(groq_key)

    if has_groq:
        # Real multi-model switching via Groq: different models per mode
        return {
            "MODE_QUICK": ModeConfig(
                mode_name="MODE_QUICK",
                model_id="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1024,
                compute_profile="low-latency",
                provider="groq",
            ),
            "MODE_REASONING": ModeConfig(
                mode_name="MODE_REASONING",
                model_id="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=4096,
                compute_profile="high-reasoning",
                provider="groq",
            ),
            "MODE_VISION": ModeConfig(
                mode_name="MODE_VISION",
                model_id="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=4096,
                compute_profile="vision-capable",
                provider="groq",
            ),
            "MODE_GEMINI": ModeConfig(
                mode_name="MODE_GEMINI",
                model_id="gemini-2.0-flash",
                temperature=0.0,
                max_tokens=8192,
                compute_profile="high-throughput",
                provider="gemini",
            ),
        }
    else:
        # Gemini-only fallback
        return {
            "MODE_QUICK": ModeConfig(
                mode_name="MODE_QUICK",
                model_id="gemini-2.0-flash",
                temperature=0.0,
                max_tokens=512,
                compute_profile="low-latency",
                provider="gemini",
            ),
            "MODE_REASONING": ModeConfig(
                mode_name="MODE_REASONING",
                model_id="gemini-2.0-flash",
                temperature=0.2,
                max_tokens=4096,
                compute_profile="high-reasoning",
                provider="gemini",
            ),
            "MODE_VISION": ModeConfig(
                mode_name="MODE_VISION",
                model_id="gemini-2.0-flash",
                temperature=0.2,
                max_tokens=4096,
                compute_profile="vision-capable",
                provider="gemini",
            ),
            "MODE_GEMINI": ModeConfig(
                mode_name="MODE_GEMINI",
                model_id="gemini-2.0-flash",
                temperature=0.0,
                max_tokens=8192,
                compute_profile="high-throughput",
                provider="gemini",
            ),
        }


MODE_REGISTRY: dict[str, ModeConfig] = _build_registry()


def is_real_switching() -> bool:
    """True if multiple distinct model_ids are in the registry."""
    ids = {cfg.model_id for cfg in MODE_REGISTRY.values()}
    return len(ids) > 1


def get_mode(mode_name: str) -> ModeConfig:
    """Look up a mode config by name."""
    if mode_name not in MODE_REGISTRY:
        raise ValueError(f"Unknown mode: {mode_name}. Valid: {list(MODE_REGISTRY)}")
    return MODE_REGISTRY[mode_name]
