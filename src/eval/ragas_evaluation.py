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
    surface: str = "topic_mode"  # topic_mode / help_assistant
    label_suffix: str = ""
    reference: str = ""  # optional ground-truth answer for answer_correctness


# ---------------------------------------------------------------------------
# Default representative cases (kept small for cost control)
# ---------------------------------------------------------------------------

DEFAULT_CASES: list[RAGAsEvalCase] = [
    RAGAsEvalCase(
        topic="LLM Basics and Prompt Engineering",
        difficulty="beginner",
        surface="learn_path",
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
        surface="learn_path",
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
        surface="learn_path",
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
    RAGAsEvalCase(
        topic="LangGraph",
        difficulty="",
        surface="topic_mode",
        label_suffix="Topic Mode",
        user_input=(
            "How does LangGraph manage shared state across nodes? Explain reducers, "
            "state updates, and why explicit state helps multi-step workflows."
        ),
        reference=(
            "LangGraph keeps workflow data in an explicit shared state object. Nodes "
            "read from that state and return updates, while reducers merge values when "
            "multiple branches write to the same field. Explicit state makes multi-step "
            "workflows easier to inspect, debug, and resume."
        ),
    ),
    RAGAsEvalCase(
        topic="How does this app work?",
        difficulty="",
        surface="help_assistant",
        label_suffix="App Workflow Context, Help Assistant",
        user_input=(
            "How does this app work?"
        ),
        reference=(
            "The app has Learn, Quiz, Progress, Dashboard, and Help Assistant surfaces. "
            "Learn generates grounded study guides and Learn Paths. Quiz generates and "
            "evaluates RAG-grounded quizzes. Progress shows saved attempts and weak areas. "
            "Dashboard exposes observability, KB health, runtime diagnostics, and evaluation readiness."
        ),
    ),
    RAGAsEvalCase(
        topic="Difference between RAG and Agentic RAG",
        difficulty="",
        surface="help_assistant",
        label_suffix="Live official docs enrichment, Help Assistant",
        user_input="Difference between RAG and Agentic RAG",
        reference=(
            "RAG retrieves relevant context and then generates an answer from that retrieved evidence. "
            "Agentic RAG adds planning or tool-using agent behavior around retrieval, such as iterative "
            "query reformulation, branching, validation, or multi-step tool use before answering."
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
    surface: str = "topic_mode"
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
    "Answer Correctness is shown as a diagnostic metric; primary RAG quality is tracked "
    "through faithfulness, answer relevancy, context precision, and context recall."
)

SURFACE_LABELS = {
    "learn_path": "Learn Path",
    "topic_mode": "Topic Mode",
    "help_assistant": "Help Assistant",
}


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


def _generate_help_assistant_content(case: RAGAsEvalCase) -> dict[str, Any]:
    """Run the Help Assistant workflow for a single evaluation case."""
    from src.services.help_assistant import (
        answer_help_question,
        get_help_assistant_app_workflow_context,
        get_help_assistant_runtime_defaults,
    )

    result = answer_help_question(
        case.user_input,
        personality_mode="Technical",
        runtime_config=get_help_assistant_runtime_defaults("Technical"),
    )
    if result.get("status") != "answered":
        return {
            "answer": "",
            "contexts": [],
            "sources_count": 0,
            "error": result.get("message") or f"Help Assistant status={result.get('status')}",
        }

    source_rows = result.get("sources", []) or []
    contexts = [str(row.get("Snippet", "")).strip() for row in source_rows if str(row.get("Snippet", "")).strip()]
    if not contexts:
        contexts = [get_help_assistant_app_workflow_context()]

    return {
        "answer": str(result.get("answer_markdown", "")).strip(),
        "contexts": contexts,
        "sources_count": len(source_rows) if source_rows else 1,
        "error": None,
    }


def _generate_eval_case_content(case: RAGAsEvalCase) -> dict[str, Any]:
    """Dispatch one evaluation case to the correct app surface."""
    if case.surface == "help_assistant":
        return _generate_help_assistant_content(case)
    return _generate_learn_content(case)


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
        surface=case.surface,
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


def _get_configured_case_map(
    configured_cases: list[RAGAsEvalCase] | None = None,
) -> dict[str, RAGAsEvalCase]:
    """Return configured evaluation cases keyed by topic."""
    cases = configured_cases or DEFAULT_CASES
    return {case.topic: case for case in cases}


def _resolve_case_surface_label(
    topic: str,
    *,
    surface: str | None = None,
    configured_cases: list[RAGAsEvalCase] | None = None,
) -> str:
    """Resolve the reviewer-facing surface label for one case."""
    configured = _get_configured_case_map(configured_cases).get(topic)
    if configured and configured.label_suffix:
        return configured.label_suffix
    resolved_surface = configured.surface if configured is not None else (surface or "")
    return SURFACE_LABELS.get(resolved_surface, "")


def _format_case_display_label(
    topic: str,
    *,
    difficulty: str = "",
    surface: str | None = None,
    configured_cases: list[RAGAsEvalCase] | None = None,
) -> str:
    """Return the reviewer-facing label suffix for one configured or evaluated case."""
    configured = _get_configured_case_map(configured_cases).get(topic)
    if configured and configured.label_suffix:
        return configured.label_suffix

    parts: list[str] = []
    if difficulty:
        parts.append(difficulty.capitalize())
    surface_label = _resolve_case_surface_label(
        topic,
        surface=surface,
        configured_cases=configured_cases,
    )
    if surface_label:
        parts.append(surface_label)
    return ", ".join(parts)


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
        content = _generate_eval_case_content(case)
        contents.append(content)
        logger.info(
            "    → {} [{}] contexts, {} chars answer{}",
            case.surface,
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


def format_ragas_report(
    report: RAGAsReport,
    *,
    configured_cases: list[RAGAsEvalCase] | None = None,
) -> str:
    """Format RAGAs evaluation report as readable text."""
    lines = [
        "=" * 60,
        "  RAGAs Content Quality Evaluation Report",
        "=" * 60,
        "",
    ]

    result_by_topic = {result.topic: result for result in report.results}
    configured = configured_cases or []

    def _append_case_block(topic: str, difficulty: str, result: RAGAsCaseResult | None) -> None:
        label_detail = _format_case_display_label(
            topic,
            difficulty=difficulty,
            surface=getattr(result, "surface", None) if result is not None else None,
            configured_cases=configured_cases,
        )
        if label_detail:
            lines.append(f"--- {topic} ({label_detail}) ---")
        else:
            lines.append(f"--- {topic} ---")

        if result is None:
            lines.append("  Status: Pending evaluation")
            lines.append("  Run a fresh benchmark to generate metrics.")
        elif result.error:
            lines.append(f"  ERROR: {result.error}")
        else:
            lines.append(f"  Contexts: {result.num_contexts}  |  Answer length: {result.answer_length} chars")
            lines.append(f"  Faithfulness:       {_fmt(result.faithfulness)}")
            lines.append(f"  Answer Relevancy:   {_fmt(result.answer_relevancy)}")
            lines.append(f"  Context Precision:  {_fmt(result.context_precision)}")
            lines.append(f"  Context Recall:     {_fmt(result.context_recall)}")
        lines.append("")

    if configured:
        for case in configured:
            _append_case_block(case.topic, case.difficulty, result_by_topic.get(case.topic))
        for result in report.results:
            if result.topic not in {case.topic for case in configured}:
                _append_case_block(result.topic, result.difficulty, result)
    else:
        for result in report.results:
            _append_case_block(result.topic, result.difficulty, result)

    lines.append("--- Averages ---")
    lines.append(f"  Faithfulness:       {_fmt(report.avg_faithfulness)}")
    lines.append(f"  Answer Relevancy:   {_fmt(report.avg_answer_relevancy)}")
    lines.append(f"  Context Precision:  {_fmt(report.avg_context_precision)}")
    lines.append(f"  Context Recall:     {_fmt(report.avg_context_recall)}")
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
    return "\n".join(lines)


def _fmt(val: float | None) -> str:
    """Format a metric value."""
    return f"{val:.4f}" if val is not None else "N/A"
