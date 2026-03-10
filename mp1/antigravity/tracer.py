"""
antigravity/tracer.py — Logging & trace system for pipeline execution.

Maintains an in-memory log of all events and can export as JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    timestamp: float
    event: str
    payload: dict[str, Any]


class Tracer:
    """Collects structured trace events throughout a pipeline run."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.search_queries: list[str] = []
        self.docs_opened: set[str] = set()
        self.modes_used: dict[str, int] = {}
        self.models_used: set[str] = set()
        self.chunks_processed: int = 0
        self._start = time.perf_counter()

    def log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self.events.append(TraceEvent(
            timestamp=time.perf_counter() - self._start,
            event=event,
            payload=payload,
        ))

        # Auto-track mode / model usage from dispatch events
        if event == "dispatch":
            mode = payload.get("mode", "")
            model = payload.get("model_id", "")
            self.modes_used[mode] = self.modes_used.get(mode, 0) + 1
            self.models_used.add(model)

    def record_search(self, query: str) -> None:
        self.search_queries.append(query)

    def record_doc_opened(self, doc_id: str) -> None:
        self.docs_opened.add(doc_id)

    def record_chunk_processed(self) -> None:
        self.chunks_processed += 1

    def to_json(self) -> list[dict]:
        return [
            {
                "t": round(e.timestamp, 4),
                "event": e.event,
                "payload": e.payload,
            }
            for e in self.events
        ]

    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 3)
