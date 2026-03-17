"""
pluto/modes.py — Real mode switching engine.

Groq primary:
  - MODE_QUICK:     llama-3.1-8b-instant   (fast, lightweight)
  - MODE_REASONING: llama-3.3-70b-versatile (deep, accurate)
  - MODE_VISION:    llama-3.1-8b-instant   (text/doc understanding)

Mistral fallback (if Groq fails or no key):
  - All modes: mistral-small-latest

Real switching = True because MODE_QUICK uses 8b and MODE_REASONING uses 70b.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
    provider: str  # "groq" | "mistral"

    def to_log_dict(self) -> dict:
        return {
            "mode_name": self.mode_name,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "compute_profile": self.compute_profile,
            "provider": self.provider,
        }


def _build_registry() -> dict[str, ModeConfig]:
    """
    Real mode switching:
      - MODE_QUICK     → llama-3.1-8b-instant   (Groq)  — fast, low-cost
      - MODE_REASONING → llama-3.3-70b-versatile (Groq)  — deep, accurate (DIFFERENT model)
      - MODE_VISION    → llama-3.1-8b-instant   (Groq)  — doc reading
    Mistral fallback if Groq not available.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if groq_key:
        return {
            "MODE_QUICK": ModeConfig(
                mode_name="MODE_QUICK",
                model_id="llama-3.1-8b-instant",       # 8B — fast
                temperature=0.1,
                max_tokens=1024,
                compute_profile="low-latency",
                provider="groq",
            ),
            "MODE_REASONING": ModeConfig(
                mode_name="MODE_REASONING",
                model_id="llama-3.3-70b-versatile",    # 70B — deep reasoning (REAL SWITCH)
                temperature=0.3,
                max_tokens=4096,
                compute_profile="high-reasoning",
                provider="groq",
            ),
            "MODE_VISION": ModeConfig(
                mode_name="MODE_VISION",
                model_id="llama-3.1-8b-instant",       # 8B — doc understanding
                temperature=0.1,
                max_tokens=4096,
                compute_profile="vision-capable",
                provider="groq",
            ),
            "MODE_GEMINI": ModeConfig(
                mode_name="MODE_GEMINI",
                model_id="llama-3.3-70b-versatile",    # 70B for synthesis
                temperature=0.0,
                max_tokens=4096,
                compute_profile="high-throughput",
                provider="groq",
            ),
        }
    else:
        # Mistral fallback — still uses small for quick, large for reasoning
        return {
            "MODE_QUICK": ModeConfig(
                mode_name="MODE_QUICK",
                model_id="mistral-small-latest",
                temperature=0.1,
                max_tokens=1024,
                compute_profile="low-latency",
                provider="mistral",
            ),
            "MODE_REASONING": ModeConfig(
                mode_name="MODE_REASONING",
                model_id="mistral-small-latest",
                temperature=0.3,
                max_tokens=4096,
                compute_profile="high-reasoning",
                provider="mistral",
            ),
            "MODE_VISION": ModeConfig(
                mode_name="MODE_VISION",
                model_id="mistral-small-latest",
                temperature=0.1,
                max_tokens=4096,
                compute_profile="vision-capable",
                provider="mistral",
            ),
            "MODE_GEMINI": ModeConfig(
                mode_name="MODE_GEMINI",
                model_id="mistral-small-latest",
                temperature=0.0,
                max_tokens=4096,
                compute_profile="high-throughput",
                provider="mistral",
            ),
        }


MODE_REGISTRY: dict[str, ModeConfig] = _build_registry()


def is_real_switching() -> bool:
    """True if MODE_QUICK and MODE_REASONING use DIFFERENT model_ids."""
    quick = MODE_REGISTRY["MODE_QUICK"].model_id
    reasoning = MODE_REGISTRY["MODE_REASONING"].model_id
    return quick != reasoning


def get_mode(mode_name: str) -> ModeConfig:
    """Look up a mode config by name."""
    if mode_name not in MODE_REGISTRY:
        raise ValueError(f"Unknown mode: {mode_name}. Valid: {list(MODE_REGISTRY)}")
    return MODE_REGISTRY[mode_name]
