#!/usr/bin/env python3
"""Run RAGAs content-quality evaluation for Learn/RAG pipeline.

Usage:
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --model gpt-4o-mini

Requires:
    pip install ragas
    OPENAI_API_KEY set in .env or environment
    Chroma vector store populated (python scripts/run_rag_eval.py --real)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(override=True)

from src.eval.ragas_evaluation import (
    run_ragas_evaluation,
    format_ragas_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run RAGAs evaluation on Learn content")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model for RAGAs judge (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RAGAs Content Quality Evaluation")
    print("=" * 60)
    print(f"  Judge model: {args.model}")
    print(f"  Cases: 3 (LLM Basics, RAG, AI Agents)")
    print()

    report = run_ragas_evaluation(model=args.model)
    print(format_ragas_report(report))


if __name__ == "__main__":
    main()
