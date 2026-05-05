#!/usr/bin/env python3
"""Run offline RAG evaluation against the knowledge base.

Usage:
    python scripts/run_rag_eval.py [--eval-file PATH] [--top-k N]

This script does NOT require OpenAI API keys — it uses a mock retrieval
function by default to validate the evaluation framework itself.

To run against a real vector store, ensure the KB has been ingested
and set OPENAI_API_KEY in .env for embedding-based retrieval.

Examples:
    # Framework-only evaluation (no API key needed):
    python scripts/run_rag_eval.py

    # With custom eval file:
    python scripts/run_rag_eval.py --eval-file data/eval/retrieval_eval_cases.md

    # With custom top-k:
    python scripts/run_rag_eval.py --top-k 10
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.eval.rag_evaluation import (
    DEFAULT_EVAL_FILE,
    run_rag_evaluation,
    format_rag_eval_report,
)


def _try_real_retrieval():
    """Try to use the real retrieval pipeline if available."""
    try:
        from src.kb.retrieval import retrieve_documents

        class _DocAdapter:
            """Adapt retrieval results to have metadata['filename']."""

            def __init__(self, doc):
                self.metadata = getattr(doc, "metadata", {})
                if not isinstance(self.metadata, dict):
                    self.metadata = {}

        def retrieval_fn(query: str, top_k: int = 5):
            results = retrieve_documents(query, top_k=top_k)
            return [_DocAdapter(doc) for doc in results]

        return retrieval_fn
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run offline RAG evaluation",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=str(DEFAULT_EVAL_FILE),
        help="Path to eval cases Markdown file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results per query",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Attempt to use real retrieval pipeline (requires ingested KB)",
    )
    args = parser.parse_args()

    retrieval_fn = None
    if args.real:
        retrieval_fn = _try_real_retrieval()
        if retrieval_fn is None:
            print("Warning: Could not initialize real retrieval. Using dry-run mode.")

    report = run_rag_evaluation(
        eval_file=args.eval_file,
        retrieval_fn=retrieval_fn,
        top_k=args.top_k,
    )

    print(format_rag_eval_report(report))

    # Exit with non-zero if there are failures
    if report.retrieval.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
