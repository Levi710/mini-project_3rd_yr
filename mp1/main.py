"""
main.py — CLI entry point for the AntiGravity pipeline.

Usage:
  python main.py --query "Your question" --corpus ./corpus
  python main.py --serve                              # start web UI
  python main.py --serve --query "..." --corpus ...   # both
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="AntiGravity — Real Mode-Switching Extraction Pipeline",
    )
    parser.add_argument("--query", "-q", type=str, help="User query to process")
    parser.add_argument("--corpus", "-c", type=str, default="./corpus", help="Path to corpus directory")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Path to output directory")
    parser.add_argument("--serve", action="store_true", help="Start the web dashboard server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    # Validate corpus dir exists
    corpus = Path(args.corpus)
    if not corpus.exists():
        corpus.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created empty corpus directory: {corpus}")

    if args.serve:
        _start_server(args.port)
    elif args.query:
        _run_cli(args.query, str(corpus), args.output)
    else:
        parser.print_help()
        sys.exit(1)


def _run_cli(query: str, corpus_dir: str, output_dir: str):
    """Run the pipeline from CLI and print structured output."""
    from antigravity.pipeline import PipelineRunner

    print("=" * 60)
    print("  AntiGravity Pipeline v1.0")
    print("=" * 60)
    print(f"  Query:  {query}")
    print(f"  Corpus: {corpus_dir}")
    print("=" * 60)
    print()

    def progress(stage, data):
        status = data.get("status", "")
        if status == "running":
            print(f"  > S{_stage_num(stage)} {stage.upper()} ... running")
        elif status == "complete":
            extras = {k: v for k, v in data.items() if k != "status"}
            print(f"  [OK] S{_stage_num(stage)} {stage.upper()} -- done  {extras}")

    runner = PipelineRunner(corpus_dir=corpus_dir, output_dir=output_dir)
    runner.on_progress(progress)

    result = runner.run(query)

    print()
    print("-" * 60)
    print("  FINAL OUTPUT")
    print("-" * 60)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=True))
    print()
    print(f"  Confidence: {result.confidence}")
    print(f"  Real switching: {result.trace_summary.real_switching}")
    print(f"  Modes used: {result.trace_summary.modes_used_counts}")
    print(f"  Models used: {result.trace_summary.models_used}")
    print(f"  Chunks processed: {result.trace_summary.chunks_processed}")
    print("=" * 60)


def _start_server(port: int):
    """Start the FastAPI web dashboard."""
    import uvicorn
    print(f"  Starting AntiGravity Dashboard on http://localhost:{port}")
    uvicorn.run("antigravity.server:app", host="0.0.0.0", port=port, reload=False)


def _stage_num(stage: str) -> int:
    return {"route": 0, "extract": 1, "merge": 2, "verify": 3, "finish": 4}.get(stage, -1)


if __name__ == "__main__":
    main()
