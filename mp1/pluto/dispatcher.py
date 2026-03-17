"""
pluto/dispatcher.py — Model dispatcher (Groq + Mistral only).

Routes calls to Groq (primary) or Mistral (fallback).
All other providers have been removed.
"""

from __future__ import annotations

import os
import re
import time
import threading
from typing import Any

from dotenv import load_dotenv

from pluto.modes import ModeConfig, get_mode
from pluto.tracer import Tracer

load_dotenv()

# ── Concurrency control ─────────────────────────────────────────────────────
# Groq has strict RPM limits — semaphore prevents parallel-call overflow.
_groq_semaphore = threading.BoundedSemaphore(2)
_groq_client: Any = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    return _groq_client


# ── Dispatch ────────────────────────────────────────────────────────────────

def dispatch(
    mode_name: str,
    prompt: str,
    tracer: Tracer | None = None,
    images: list[bytes] | None = None,
) -> str:
    """Route prompt to Groq, fall back to Mistral if Groq fails."""
    cfg = get_mode(mode_name)

    if tracer:
        tracer.log("dispatch", {
            "mode": cfg.mode_name,
            "model_id": cfg.model_id,
            "provider": cfg.provider,
            "prompt_length": len(prompt),
        })

    print(f"  [DISPATCH] {cfg.provider} / {cfg.model_id} ({len(prompt)} chars)")
    t0 = time.perf_counter()

    try:
        if cfg.provider == "groq":
            text = _call_groq(cfg, prompt)
        elif cfg.provider == "mistral":
            text = _call_mistral(cfg, prompt)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

    except Exception as e:
        print(f"  [WARNING] {cfg.provider} failed: {e}")

        # ── Fallback to the other provider ──────────────────────────────────
        if cfg.provider == "groq":
            print("  [FALLBACK] Trying Mistral...")
            fb = ModeConfig(
                mode_name=cfg.mode_name,
                model_id="mistral-small-latest",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                compute_profile="fallback",
                provider="mistral",
            )
            text = _call_mistral(fb, prompt)
        else:
            # Mistral failed — try Groq if key exists
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key:
                print("  [FALLBACK] Trying Groq...")
                fb = ModeConfig(
                    mode_name=cfg.mode_name,
                    model_id="llama-3.1-8b-instant",
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    compute_profile="fallback",
                    provider="groq",
                )
                text = _call_groq(fb, prompt)
            else:
                raise

    elapsed = time.perf_counter() - t0
    if tracer:
        tracer.log("dispatch_complete", {
            "mode": cfg.mode_name,
            "elapsed_s": round(elapsed, 3),
            "response_length": len(text),
        })
    return text


# ── Groq ─────────────────────────────────────────────────────────────────────

def _call_groq(cfg: ModeConfig, prompt: str) -> str:
    """Call Groq with smart retry — respects the 'try again in Xs' header."""
    GROQ_MAX_RETRIES = 8

    with _groq_semaphore:
        client = _get_groq_client()

        for attempt in range(GROQ_MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=cfg.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                return resp.choices[0].message.content or ""

            except Exception as e:
                err = str(e).lower()
                is_rate = "429" in err or "rate limit" in err or "quota" in err
                if is_rate and attempt < GROQ_MAX_RETRIES:
                    m = re.search(r"try again in ([\d\.]+)s", err)
                    delay = float(m.group(1)) + 0.5 if m else min(60.0, 5.0 * (1.5 ** attempt))
                    print(f"  [RETRY] Groq rate-limit — waiting {delay:.1f}s (attempt {attempt+1})")
                    time.sleep(delay)
                else:
                    raise


# ── Mistral ───────────────────────────────────────────────────────────────────

def _call_mistral(cfg: ModeConfig, prompt: str) -> str:
    """Call Mistral AI REST API with retry on rate-limit."""
    import requests

    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    for attempt in range(6):
        try:
            r = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                delay = 10 * (attempt + 1)
                print(f"  [RETRY] Mistral rate-limit — waiting {delay}s (attempt {attempt+1})")
                time.sleep(delay)
            else:
                raise Exception(f"Mistral {r.status_code}: {r.text[:200]}")
        except Exception:
            if attempt == 5:
                raise
            time.sleep(5)

    return ""
