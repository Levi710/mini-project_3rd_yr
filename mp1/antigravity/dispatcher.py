"""
antigravity/dispatcher.py — Model dispatcher with per-call logging.

Calls the real model API based on ModeConfig. Every call is logged with
model_id, temperature, max_tokens, and compute_profile.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

from antigravity.modes import ModeConfig, get_mode
from antigravity.tracer import Tracer

load_dotenv()

# ── Lazy-initialised clients ──────────────────────────────────────────────────

_gemini_client: Any = None
_groq_client: Any = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY", "")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    return _groq_client


# ── Dispatch ───────────────────────────────────────────────────────────────────

def dispatch(
    mode_name: str,
    prompt: str,
    tracer: Tracer | None = None,
    images: list[bytes] | None = None,
) -> str:
    """
    Send *prompt* to the real model API configured for *mode_name*.

    Returns the raw text response.  Every call is logged via *tracer*.
    """
    cfg = get_mode(mode_name)

    # Log the call
    if tracer:
        tracer.log("dispatch", {
            "mode": cfg.mode_name,
            "model_id": cfg.model_id,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "compute_profile": cfg.compute_profile,
            "provider": cfg.provider,
            "prompt_length": len(prompt),
        })

    t0 = time.perf_counter()

    if cfg.provider == "gemini":
        text = _call_gemini(cfg, prompt, images)
    elif cfg.provider == "groq":
        text = _call_groq(cfg, prompt)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")

    elapsed = time.perf_counter() - t0
    if tracer:
        tracer.log("dispatch_complete", {
            "mode": cfg.mode_name,
            "model_id": cfg.model_id,
            "elapsed_s": round(elapsed, 3),
            "response_length": len(text),
        })

    return text


# ── Provider implementations ──────────────────────────────────────────────────

MAX_RETRIES = 5
BASE_DELAY = 30  # seconds — Gemini free tier needs longer waits
INTER_CALL_DELAY = 4  # seconds between consecutive Gemini calls

_last_gemini_call: float = 0


def _call_gemini(cfg: ModeConfig, prompt: str, images: list[bytes] | None) -> str:
    """Call Gemini via google-genai SDK with retry on rate limit."""
    global _last_gemini_call
    # Throttle: wait between consecutive Gemini calls
    since_last = time.perf_counter() - _last_gemini_call
    if _last_gemini_call > 0 and since_last < INTER_CALL_DELAY:
        time.sleep(INTER_CALL_DELAY - since_last)
    client = _get_gemini_client()

    contents: list[Any] = []
    if images:
        from google.genai import types

        for img_bytes in images:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
    contents.append(prompt)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=cfg.model_id,
                contents=contents,
                config={
                    "temperature": cfg.temperature,
                    "max_output_tokens": cfg.max_tokens,
                },
            )
            _last_gemini_call = time.perf_counter()
            return response.text or ""
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "resource" in err_str or "quota" in err_str or "rate" in err_str) and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  [RETRY] Gemini rate-limited, waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise


def _call_groq(cfg: ModeConfig, prompt: str) -> str:
    """Call Groq via SDK with smart retry on rate limit."""
    import re
    client = _get_groq_client()

    # Allow more retries for large batch processing (e.g. 71 chunks)
    GROQ_MAX_RETRIES = 10

    for attempt in range(GROQ_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate limit" in err_str or "quota" in err_str) and attempt < GROQ_MAX_RETRIES:
                # Try to parse exact wait time from Groq's error message (e.g., "Please try again in 5.3s")
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                if match:
                    delay = float(match.group(1)) + 0.5  # add 500ms buffer
                    print(f"  [RETRY] Groq requested wait: {delay}s (attempt {attempt + 1}/{GROQ_MAX_RETRIES})...")
                else:
                    # Generic backoff if no time specified, capped at 60s
                    delay = min(60.0, 5.0 * (1.5 ** attempt))
                    print(f"  [RETRY] Groq generic backoff: {delay}s (attempt {attempt + 1}/{GROQ_MAX_RETRIES})...")
                
                time.sleep(delay)
            else:
                raise

