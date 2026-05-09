"""Lightweight offline RAG evaluation.

Extends retrieval validation with source coverage metrics and
structured reporting. Designed to work without external API keys.

Note: RAGAs integration can be added later by installing the `ragas`
package and using the eval cases as RAGAs-compatible datasets.
See: https://docs.ragas.io/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.eval.retrieval_validation import (
    EvalCase,
    CaseResult,
    ValidationSummary,
    parse_eval_cases,
    validate_retrieval,
)

DEFAULT_EVAL_FILE = Path("data/eval/retrieval_eval_cases.md")


@dataclass
class SourceCoverage:
    """Coverage of expected source files across all eval cases."""

    total_expected_files: int
    unique_expected_files: list[str]
    files_hit: list[str]
    files_missed: list[str]
    coverage_rate: float


@dataclass
class RAGEvalReport:
    """Full RAG evaluation report combining retrieval and coverage."""

    retrieval: ValidationSummary
    coverage: SourceCoverage
    has_ragas: bool = False
    ragas_note: str = ""


def compute_source_coverage(
    cases: list[EvalCase],
    results: list[CaseResult],
) -> SourceCoverage:
    """Compute source coverage across all eval results.

    Args:
        cases: Original eval cases.
        results: Per-case retrieval results.

    Returns:
        SourceCoverage with hit/miss analysis.
    """
    all_expected: set[str] = set()
    for case in cases:
        all_expected.update(case.expected_files)

    unique_expected = sorted(all_expected)

    all_retrieved: set[str] = set()
    for result in results:
        all_retrieved.update(result.retrieved_files)

    files_hit = sorted(f for f in unique_expected if f in all_retrieved)
    files_missed = sorted(f for f in unique_expected if f not in all_retrieved)
    total = len(unique_expected)
    coverage_rate = (len(files_hit) / total * 100.0) if total > 0 else 0.0

    return SourceCoverage(
        total_expected_files=total,
        unique_expected_files=unique_expected,
        files_hit=files_hit,
        files_missed=files_missed,
        coverage_rate=coverage_rate,
    )


def run_rag_evaluation(
    eval_file: str | Path = DEFAULT_EVAL_FILE,
    retrieval_fn: callable | None = None,
    top_k: int = 5,
) -> RAGEvalReport:
    """Run a full offline RAG evaluation.

    Args:
        eval_file: Path to eval cases Markdown file.
        retrieval_fn: Function(query, top_k) -> list[Document].
            If None, uses a dummy retrieval that returns no results.
        top_k: Number of results to retrieve per query.

    Returns:
        RAGEvalReport with retrieval validation and source coverage.
    """
    cases = parse_eval_cases(eval_file)
    if not cases:
        logger.warning("No eval cases found in {}", eval_file)
        empty_summary = ValidationSummary(
            total=0, passed=0, failed=0, pass_rate=0.0,
        )
        empty_coverage = SourceCoverage(
            total_expected_files=0,
            unique_expected_files=[],
            files_hit=[],
            files_missed=[],
            coverage_rate=0.0,
        )
        return RAGEvalReport(
            retrieval=empty_summary,
            coverage=empty_coverage,
            has_ragas=_check_ragas_available(),
            ragas_note=_ragas_availability_note(),
        )

    if retrieval_fn is None:
        def retrieval_fn(query: str, top_k: int = 5) -> list:
            return []

    summary = validate_retrieval(cases, retrieval_fn, top_k=top_k)
    coverage = compute_source_coverage(cases, summary.results)

    report = RAGEvalReport(
        retrieval=summary,
        coverage=coverage,
        has_ragas=_check_ragas_available(),
        ragas_note=_ragas_availability_note(),
    )

    logger.info(
        "RAG evaluation complete: {}/{} passed, {:.1f}% source coverage",
        summary.passed, summary.total, coverage.coverage_rate,
    )

    return report


def format_rag_eval_report(report: RAGEvalReport) -> str:
    """Format a full RAG evaluation report as readable text.

    Args:
        report: RAG evaluation report.

    Returns:
        Human-readable report string.
    """
    lines = [
        "=" * 50,
        "  RAG Evaluation Report",
        "=" * 50,
        "",
        "--- Retrieval Hit Rate ---",
        f"Total queries:  {report.retrieval.total}",
        f"Passed:         {report.retrieval.passed}",
        f"Failed:         {report.retrieval.failed}",
        f"Hit rate:       {report.retrieval.pass_rate:.1f}%",
        "",
        "--- Source Coverage ---",
        f"Expected files: {report.coverage.total_expected_files}",
        f"Files hit:      {len(report.coverage.files_hit)}",
        f"Files missed:   {len(report.coverage.files_missed)}",
        f"Coverage rate:  {report.coverage.coverage_rate:.1f}%",
    ]

    if report.coverage.files_missed:
        lines.append("")
        lines.append("Missed files:")
        for f in report.coverage.files_missed:
            lines.append(f"  - {f}")

    if report.retrieval.failed > 0:
        lines.append("")
        lines.append("--- Failed Cases ---")
        for r in report.retrieval.results:
            if not r.passed:
                lines.append(f'  Query: "{r.query}"')
                lines.append(f"    Expected:  {r.expected_files}")
                lines.append(f"    Retrieved: {r.retrieved_files}")
                lines.append(f"    Missing:   {r.missing_files}")
                lines.append("")

    if report.ragas_note:
        lines.append("")
        lines.append(f"Note: {report.ragas_note}")

    lines.append("")
    return "\n".join(lines)


def _check_ragas_available() -> bool:
    """Check if the ragas package is installed."""
    try:
        import ragas  # noqa: F401
        return True
    except ImportError:
        return False


def _ragas_availability_note() -> str:
    """Return a note about RAGAs availability."""
    if _check_ragas_available():
        return "RAGAs is installed. Advanced metrics (faithfulness, relevancy) can be enabled."
    return (
        "RAGAs is not installed. Install with `pip install ragas` "
        "to enable advanced evaluation metrics (faithfulness, answer relevancy, context precision)."
    )
