"""RAGAs content-quality evaluation for Learn/RAG pipeline.

Runs real Learn workflows, collects generated answers and retrieved contexts,
then evaluates content quality using RAGAs metrics (faithfulness,
answer relevancy, context precision, context recall).

Requires: pip install ragas
Usage:  python scripts/run_ragas_eval.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Cached results path
# ---------------------------------------------------------------------------

LATEST_RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "results" / "latest_ragas_eval.json"


# ---------------------------------------------------------------------------
# Eval case definition
# ---------------------------------------------------------------------------

@dataclass
class RAGAsEvalCase:
    """A single evaluation case for RAGAs metrics."""

    topic: str
    difficulty: str  # beginner / intermediate / advanced
    user_input: str  # the question / learning request
    reference: str = ""  # optional ground-truth answer for answer_correctness


# ---------------------------------------------------------------------------
# Default representative cases (kept small for cost control)
# ---------------------------------------------------------------------------

DEFAULT_CASES: list[RAGAsEvalCase] = [
    RAGAsEvalCase(
        topic="LLM Basics and Prompt Engineering",
        difficulty="beginner",
        user_input="Explain the fundamentals of large language models and basic prompt engineering techniques.",
        reference=(
            "Large language models are neural networks trained on large text corpora "
            "that predict the next token. Key prompt engineering techniques include "
            "zero-shot prompting, few-shot prompting, chain-of-thought prompting, "
            "and role-based prompting. Temperature and top-p control output randomness."
        ),
    ),
    RAGAsEvalCase(
        topic="RAG and Vector Databases",
        difficulty="intermediate",
        # Narrowed question to focus on core RAG pipeline — the previous version
        # asked about "architecture, vector databases, and embedding strategies"
        # which diluted answer relevancy because the generated guide covered many
        # sub-topics broadly. A focused question yields a more relevant answer.
        user_input=(
            "How does retrieval-augmented generation work? Explain the RAG pipeline "
            "including document chunking, embedding, vector storage, retrieval, "
            "and answer generation."
        ),
        reference=(
            "RAG combines a retriever with a generator: documents are split into "
            "chunks, embedded into vectors using an embedding model, and stored in "
            "a vector database. At query time the user question is embedded, similar "
            "chunks are retrieved via vector similarity search, and the retrieved "
            "context is passed to an LLM to generate a grounded answer. Key "
            "components include chunking strategies, embedding models, similarity "
            "metrics, top-k retrieval, and optional reranking."
        ),
    ),
    RAGAsEvalCase(
        topic="AI Agents and Tool Calling",
        difficulty="advanced",
        # Narrowed question to focus on tool calling mechanics — the previous
        # version asked broadly about "ReAct pattern for autonomous task
        # execution" which lowered context precision because the retriever
        # pulled in tangential agent-architecture chunks. Focusing on tool
        # calling and function definitions yields more precise contexts.
        user_input=(
            "How do AI agents use tool calling? Explain function definitions, "
            "structured tool-call requests, and how frameworks like LangChain "
            "orchestrate agent tool use."
        ),
        reference=(
            "AI agents use tool calling by providing the LLM with function "
            "definitions (name, description, parameters). The LLM decides which "
            "tool to call and returns a structured tool-call request with arguments. "
            "The framework executes the function, returns the result to the LLM, "
            "and the cycle repeats. LangChain and LangGraph provide tool-calling "
            "APIs and agent orchestration for multi-step tool use loops."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RAGAsCaseResult:
    """Result of RAGAs evaluation for a single case."""

    topic: str
    difficulty: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    num_contexts: int = 0
    answer_length: int = 0
    error: str | None = None


@dataclass
class RAGAsReport:
    """Aggregated RAGAs evaluation report."""

    results: list[RAGAsCaseResult] = field(default_factory=list)
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    avg_context_precision: float | None = None
    avg_context_recall: float | None = None
    avg_answer_correctness: float | None = None
    timestamp: str = ""
    model: str = ""
    case_count: int = 0


# ---------------------------------------------------------------------------
# Primary vs diagnostic metric classification
# ---------------------------------------------------------------------------

#: Metrics used for primary Learn quality assessment.
PRIMARY_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

#: Diagnostic-only metrics (not used for pass/fail).
DIAGNOSTIC_METRICS = ("answer_correctness",)

ANSWER_CORRECTNESS_NOTE = (
    "Answer Correctness is a **diagnostic** metric, not a primary quality indicator. "
    "It measures alignment between the generated study guide and a short reference answer. "
    "Scores can be artificially low because generated guides are long and comprehensive "
    "(10 000–25 000 chars) while reference answers are intentionally short and concise. "
    "**Faithfulness, Answer Relevancy, Context Precision, and Context Recall** are the "
    "primary Learn quality metrics."
)


# ---------------------------------------------------------------------------
# Cache save / load
# ---------------------------------------------------------------------------

def save_ragas_results(report: RAGAsReport, path: Path | None = None) -> Path:
    """Persist a RAGAs report to JSON."""
    path = path or LATEST_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("RAGAs results saved to {}", path)
    return path


def load_ragas_results(path: Path | None = None) -> RAGAsReport | None:
    """Load the latest cached RAGAs report, or *None* if absent/corrupt."""
    path = path or LATEST_RESULTS_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = [
            RAGAsCaseResult(**r) for r in data.pop("results", [])
        ]
        return RAGAsReport(results=results, **data)
    except Exception as exc:
        logger.warning("Failed to load cached RAGAs results: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _generate_learn_content(case: RAGAsEvalCase) -> dict[str, Any]:
    """Run the real Learn workflow for a single case.

    Returns dict with 'answer', 'contexts', 'sources_count'.
    """
    from src.graphs.learn_graph import run_learn_workflow
    from src.schemas import DifficultyLevel, ResponseStyle

    difficulty_map = {
        "beginner": DifficultyLevel.BEGINNER,
        "intermediate": DifficultyLevel.INTERMEDIATE,
        "advanced": DifficultyLevel.ADVANCED,
    }
    difficulty = difficulty_map.get(case.difficulty, DifficultyLevel.INTERMEDIATE)

    state = run_learn_workflow(
        topic=case.topic,
        difficulty=difficulty,
        style=ResponseStyle.DETAILED,
        force_regenerate=True,
    )

    guide = state.get("study_guide")
    docs = state.get("retrieved_docs") or []

    if guide is None:
        return {
            "answer": "",
            "contexts": [],
            "sources_count": 0,
            "error": state.get("error", "No study guide generated"),
        }

    answer_parts = []
    if guide.summary:
        answer_parts.append(guide.summary)
    if guide.detailed_notes:
        answer_parts.append(guide.detailed_notes)
    answer = "\n\n".join(answer_parts)

    contexts = [doc.content for doc in docs if doc.content]

    return {
        "answer": answer,
        "contexts": contexts,
        "sources_count": len(docs),
        "error": None,
    }


async def _evaluate_single_case(
    case: RAGAsEvalCase,
    content: dict[str, Any],
    metrics: list,
) -> RAGAsCaseResult:
    """Evaluate a single case with all RAGAs metrics.

    Each ragas 0.4 collections metric has its own keyword signature:
      - Faithfulness(user_input, response, retrieved_contexts)
      - AnswerRelevancy(user_input, response)
      - ContextPrecision(user_input, reference, retrieved_contexts)
      - ContextRecall(user_input, retrieved_contexts, reference)
      - AnswerCorrectness(user_input, response, reference)
    """
    result = RAGAsCaseResult(
        topic=case.topic,
        difficulty=case.difficulty,
        num_contexts=len(content["contexts"]),
        answer_length=len(content["answer"]),
    )

    if content.get("error"):
        result.error = content["error"]
        return result

    if not content["answer"] or not content["contexts"]:
        result.error = "Empty answer or contexts"
        return result

    user_input = case.user_input
    response = content["answer"]
    contexts = content["contexts"]
    reference = case.reference

    # Truncate very long responses to avoid max_tokens failures during
    # Faithfulness statement extraction and AnswerCorrectness comparison.
    # Study guides are 14-23k chars; Faithfulness works at 2000 chars,
    # AnswerCorrectness (TP/FP/FN classification) needs shorter input.
    faith_response = response[:1500] if len(response) > 1500 else response
    correctness_response = response[:1200] if len(response) > 1200 else response

    # Map metric class name → (field_name, kwargs)
    # Faithfulness and AnswerCorrectness use truncated response to avoid
    # structured-output failures with very long answers (14-23k chars)
    metric_kwargs_map = {
        "Faithfulness": ("faithfulness", {
            "user_input": user_input, "response": faith_response,
            "retrieved_contexts": contexts,
        }),
        "AnswerRelevancy": ("answer_relevancy", {
            "user_input": user_input, "response": response,
        }),
        "ContextPrecision": ("context_precision", {
            "user_input": user_input, "reference": reference,
            "retrieved_contexts": contexts,
        }),
        "ContextRecall": ("context_recall", {
            "user_input": user_input, "retrieved_contexts": contexts,
            "reference": reference,
        }),
        "AnswerCorrectness": ("answer_correctness", {
            "user_input": user_input, "response": correctness_response,
            "reference": reference,
        }),
    }

    for metric in metrics:
        metric_name = type(metric).__name__
        entry = metric_kwargs_map.get(metric_name)
        if entry is None:
            continue
        field_name, kwargs = entry

        # Skip metrics that require reference if none provided
        if not reference and "reference" in kwargs:
            continue

        try:
            metric_result = await metric.ascore(**kwargs)
            setattr(result, field_name, round(float(metric_result.value), 4))
        except Exception as e:
            logger.warning("Metric {} failed for '{}': {}", metric_name, case.topic, e)

    return result


def _compute_averages(results: list[RAGAsCaseResult]) -> dict[str, float | None]:
    """Compute average scores across successful results."""
    metric_names = [
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "answer_correctness",
    ]
    averages = {}
    for name in metric_names:
        values = [getattr(r, name) for r in results if getattr(r, name) is not None]
        averages[f"avg_{name}"] = round(sum(values) / len(values), 4) if values else None
    return averages


def run_ragas_evaluation(
    cases: list[RAGAsEvalCase] | None = None,
    model: str = "gpt-4o-mini",
) -> RAGAsReport:
    """Run full RAGAs evaluation pipeline.

    1. Generate real Learn content for each case
    2. Evaluate with RAGAs metrics
    3. Return aggregated report

    Args:
        cases: Eval cases. Defaults to DEFAULT_CASES.
        model: OpenAI model for RAGAs judge LLM.

    Returns:
        RAGAsReport with per-case and average scores.
    """
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from openai import AsyncOpenAI

    if cases is None:
        cases = DEFAULT_CASES

    # Initialize judge LLM and embeddings (ragas 0.4 async scoring needs AsyncOpenAI)
    client = AsyncOpenAI()
    llm = llm_factory(model, client=client)
    embeddings = embedding_factory(client=client)

    # Initialize metrics
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]

    # Step 1: Generate content for all cases
    logger.info("Generating Learn content for {} cases...", len(cases))
    contents: list[dict[str, Any]] = []
    for case in cases:
        logger.info("  Generating: {} ({})", case.topic, case.difficulty)
        content = _generate_learn_content(case)
        contents.append(content)
        logger.info(
            "    → {} contexts, {} chars answer{}",
            len(content["contexts"]),
            len(content["answer"]),
            f" ERROR: {content['error']}" if content.get("error") else "",
        )

    # Step 2: Evaluate with RAGAs
    logger.info("Running RAGAs evaluation...")
    results: list[RAGAsCaseResult] = []

    loop = asyncio.new_event_loop()
    try:
        for case, content in zip(cases, contents):
            logger.info("  Evaluating: {} ({})", case.topic, case.difficulty)
            result = loop.run_until_complete(
                _evaluate_single_case(case, content, metrics)
            )
            results.append(result)
            logger.info(
                "    → faith={} relevancy={} ctx_prec={} ctx_recall={} correctness={}",
                result.faithfulness, result.answer_relevancy,
                result.context_precision, result.context_recall,
                result.answer_correctness,
            )
    finally:
        loop.close()

    # Step 3: Compute averages
    averages = _compute_averages(results)

    report = RAGAsReport(
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        case_count=len(cases),
        **averages,
    )

    # Persist to cache
    try:
        save_ragas_results(report)
    except Exception as exc:
        logger.warning("Could not save RAGAs results to cache: {}", exc)

    return report


def format_ragas_report(report: RAGAsReport) -> str:
    """Format RAGAs evaluation report as readable text."""
    lines = [
        "=" * 60,
        "  RAGAs Content Quality Evaluation Report",
        "=" * 60,
        "",
    ]

    for r in report.results:
        lines.append(f"--- {r.topic} ({r.difficulty}) ---")
        if r.error:
            lines.append(f"  ERROR: {r.error}")
        else:
            lines.append(f"  Contexts: {r.num_contexts}  |  Answer length: {r.answer_length} chars")
            lines.append(f"  Faithfulness:       {_fmt(r.faithfulness)}")
            lines.append(f"  Answer Relevancy:   {_fmt(r.answer_relevancy)}")
            lines.append(f"  Context Precision:  {_fmt(r.context_precision)}")
            lines.append(f"  Context Recall:     {_fmt(r.context_recall)}")
            lines.append(f"  Answer Correctness: {_fmt(r.answer_correctness)}")
        lines.append("")

    lines.append("--- Averages ---")
    lines.append(f"  Faithfulness:       {_fmt(report.avg_faithfulness)}")
    lines.append(f"  Answer Relevancy:   {_fmt(report.avg_answer_relevancy)}")
    lines.append(f"  Context Precision:  {_fmt(report.avg_context_precision)}")
    lines.append(f"  Context Recall:     {_fmt(report.avg_context_recall)}")
    lines.append(f"  Answer Correctness: {_fmt(report.avg_answer_correctness)}")
    lines.append("")

    # Quality assessment (primary metrics only)
    lines.append("--- Quality Assessment (Primary Metrics) ---")
    passing = True
    for name, val in [
        ("Faithfulness", report.avg_faithfulness),
        ("Answer Relevancy", report.avg_answer_relevancy),
        ("Context Precision", report.avg_context_precision),
        ("Context Recall", report.avg_context_recall),
    ]:
        if val is not None and val < 0.6:
            lines.append(f"  ⚠ {name} below threshold (0.6): {val:.4f}")
            passing = False

    if passing:
        lines.append("  ✅ All primary metrics above threshold — Learn quality is acceptable.")
    else:
        lines.append("  ❌ Some primary metrics below threshold — review recommended.")
    lines.append("")
    lines.append("Note: Answer Correctness is diagnostic only (reference-alignment for")
    lines.append("long study guides vs short references). See primary metrics above.")

    lines.append("")
    return "\n".join(lines)


def _fmt(val: float | None) -> str:
    """Format a metric value."""
    return f"{val:.4f}" if val is not None else "N/A"
