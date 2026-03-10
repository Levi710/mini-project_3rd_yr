"""
antigravity/server.py — FastAPI server bridging pipeline <-> web UI.

Endpoints:
  POST /api/run      — start pipeline, return final JSON
  POST /api/upload   — upload files to the corpus
  GET  /api/corpus   — list corpus documents
  GET  /api/stream   — SSE stream of pipeline progress
  GET  /            — serve the frontend dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from antigravity.pipeline import PipelineRunner
from antigravity.extraction_cache import ExtractionCache
from antigravity.doc_index import DocIndex

app = FastAPI(title="Pluto Pipeline", version="1.0.0")

# ── State ─────────────────────────────────────────────────────────────────────

_progress_queue: asyncio.Queue = asyncio.Queue()  # Always exists — reset per run
_latest_result: dict | None = None

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CORPUS_DIR = Path(__file__).parent.parent / "corpus"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Shared instances
_extraction_cache = ExtractionCache(str(CORPUS_DIR))
_doc_index = DocIndex(persist_path=CORPUS_DIR / ".doc_index.json")


# ── Startup: re-index existing corpus files ─────────────────────────────────

@app.on_event("startup")
async def startup_reindex():
    """On server start, index any corpus files not already in DocIndex."""
    import logging
    from antigravity.ingest import ingest_file, _split_into_chunks, _classify_and_tag_chunks
    from antigravity.doc_index import ChunkMeta

    logger = logging.getLogger("antigravity")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for md_file in sorted(CORPUS_DIR.glob("*.md")):
        doc_id = md_file.stem
        if _doc_index.has_doc(doc_id):
            continue  # Already indexed (loaded from disk)

        logger.info(f"Re-indexing existing corpus file: {doc_id}")
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            chunks = _split_into_chunks(content)
            chunk_meta_list = _classify_and_tag_chunks(chunks)
            meta_objects = [
                ChunkMeta(
                    chunk_id=m["chunk_id"],
                    chunk_type=m["chunk_type"],
                    mode=m["mode"],
                    header=m["header"],
                )
                for m in chunk_meta_list
            ]
            _doc_index.register_doc(
                doc_id=doc_id,
                filename=md_file.name,
                chunks=chunks,
                chunk_meta=meta_objects,
            )
        except Exception as e:
            logger.warning(f"Failed to re-index {doc_id}: {e}")

    logger.info(f"DocIndex ready: {len(_doc_index.list_docs())} documents indexed")


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = FRONTEND_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")



# ── API routes ────────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run_pipeline(request: Request):
    """Run the full pipeline for a user query."""
    global _latest_result, _progress_queue

    body = await request.json()
    query = body.get("query", "")
    corpus_dir = body.get("corpus_dir", str(CORPUS_DIR))

    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    # Reset queue for this run (drain any leftover events without replacing the object)
    while not _progress_queue.empty():
        try:
            _progress_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    def progress_callback(stage: str, data: dict):
        _progress_queue.put_nowait({"stage": stage, **data})

    # Run pipeline in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    runner = PipelineRunner(
        corpus_dir=corpus_dir, output_dir=str(OUTPUT_DIR),
        doc_index=_doc_index,
    )
    runner.on_progress(progress_callback)

    try:
        result = await loop.run_in_executor(None, runner.run, query)
        _latest_result = result.model_dump()

        # Include cache stats in the response
        cache_stats = runner.cache.stats()
        _latest_result["cache_hits"] = cache_stats["hits"]
        _latest_result["cache_misses"] = cache_stats["misses"]

        # Signal completion
        await _progress_queue.put({"stage": "done", "status": "complete"})

        return JSONResponse(_latest_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Log error to frontend stream
        await _progress_queue.put({"stage": "error", "status": "failed", "detail": str(e)})
            
        return JSONResponse({"error": f"Pipeline crashed: {str(e)}"}, status_code=500)


@app.get("/api/stream")
async def stream_progress():
    """SSE stream of pipeline progress events."""

    async def event_generator():
        # Wait for events from the pipeline — keep connection open
        while True:
            try:
                event = await asyncio.wait_for(_progress_queue.get(), timeout=120.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("stage") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'stage': 'heartbeat'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/result")
async def get_result():
    """Return the latest pipeline result."""
    if _latest_result:
        return JSONResponse(_latest_result)
    return JSONResponse({"error": "No result yet"}, status_code=404)


# ── File upload ───────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more files to the corpus."""
    from antigravity.ingest import ingest_file

    results = []
    errors = []

    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append({"filename": file.filename, "error": f"Unsupported type: {ext}"})
            continue

        # Save to temp, then ingest
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir) / (file.filename or "upload")
            with open(tmp_path, "wb") as f:
                content = await file.read()
                f.write(content)

            info = ingest_file(tmp_path, str(CORPUS_DIR), doc_index=_doc_index)

            # ── Phase A: understand in BACKGROUND (don't block upload) ──
            doc_id = info["doc_id"]
            if not _doc_index.is_processed(doc_id):
                import threading
                def _bg_understand(did):
                    try:
                        from antigravity.stages.understand import run_understand
                        from antigravity.tracer import Tracer
                        tracer = Tracer()
                        run_understand(did, _doc_index, tracer)
                    except Exception as e:
                        import traceback
                        print(f"Background Phase A failed for {did}: {e}")
                        traceback.print_exc()
                        pass  # Non-fatal: _ensure_docs_understood is the safety net
                threading.Thread(target=_bg_understand, args=(doc_id,), daemon=True).start()
                info["understanding"] = "in_progress"
            else:
                info["understanding"] = "complete"

            results.append(info)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return JSONResponse({
        "uploaded": results,
        "errors": errors,
        "corpus_size": len(list(CORPUS_DIR.glob("*.md"))),
    })


@app.get("/api/doc-status/{doc_id}")
async def doc_status(doc_id: str):
    """Check if a document has been fully understood (Phase A complete)."""
    if not _doc_index.has_doc(doc_id):
        return JSONResponse({"doc_id": doc_id, "status": "not_found"}, status_code=404)
    is_done = _doc_index.is_processed(doc_id)
    return JSONResponse({
        "doc_id": doc_id,
        "status": "ready" if is_done else "understanding",
        "has_overview": bool(_doc_index.get_overview(doc_id)),
        "chunk_count": _doc_index.get_chunk_count(doc_id),
    })


@app.get("/api/cache/stats")
async def cache_stats():
    """Return extraction cache statistics."""
    return JSONResponse(_extraction_cache.stats())


@app.get("/api/corpus")
async def list_corpus():
    """List all documents in the corpus."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in sorted(CORPUS_DIR.glob("*.md")):
        docs.append({
            "doc_id": f.stem,
            "filename": f.name,
            "size": f.stat().st_size,
        })
    return JSONResponse({"documents": docs, "total": len(docs)})


@app.delete("/api/corpus/{doc_id}")
async def delete_corpus_doc(doc_id: str):
    """Delete a document from the corpus."""
    target = CORPUS_DIR / f"{doc_id}.md"
    if target.exists():
        target.unlink()
        # Remove from doc index
        _doc_index.remove_doc(doc_id)
        # Invalidate extraction cache for this doc
        removed = _extraction_cache.invalidate_doc(doc_id)
        _extraction_cache.save()
        return JSONResponse({"deleted": doc_id, "cache_entries_cleared": removed})
    return JSONResponse({"error": f"Document {doc_id} not found"}, status_code=404)


# ── Static file mount (AFTER all API routes to prevent shadowing) ─────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

