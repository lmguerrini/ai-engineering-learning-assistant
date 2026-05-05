"""Retrieval validation against expected eval cases.

Parses retrieval evaluation cases from a Markdown file and validates
that the retrieval pipeline returns expected source documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class EvalCase:
    """A single retrieval evaluation case."""

    query: str
    expected_files: list[str]
    category: str = ""


@dataclass
class CaseResult:
    """Result of a single eval case validation."""

    query: str
    expected_files: list[str]
    retrieved_files: list[str]
    passed: bool
    missing_files: list[str] = field(default_factory=list)


@dataclass
class ValidationSummary:
    """Summary of all eval case results."""

    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[CaseResult] = field(default_factory=list)


def parse_eval_cases(filepath: str | Path) -> list[EvalCase]:
    """Parse retrieval eval cases from a Markdown file.

    Expected format per case:
        Query:
        "some query"

        Expected:
        - filename.md
        - other_file.md

    Args:
        filepath: Path to the eval cases Markdown file.

    Returns:
        List of parsed EvalCase objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("Eval cases file not found: {}", filepath)
        return []

    text = filepath.read_text(encoding="utf-8")
    cases: list[EvalCase] = []
    current_category = ""

    # Split into sections by ---
    sections = re.split(r"\n---\n", text)

    query: str | None = None
    expected: list[str] = []

    for section in sections:
        lines = section.strip().splitlines()

        # Track category from headers
        for line in lines:
            header_match = re.match(r"^#{1,3}\s+\d*\.?\s*(.*)", line)
            if header_match:
                current_category = header_match.group(1).strip()

        # Look for Query: / Expected: pattern
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Query:"):
                # Next non-empty line is the query
                query = None
                expected = []
                j = i + 1
                while j < len(lines):
                    candidate = lines[j].strip()
                    if candidate:
                        # Remove surrounding quotes
                        query = candidate.strip('"').strip("'")
                        break
                    j += 1
                i = j + 1
                continue

            if line.startswith("Expected:"):
                expected = []
                j = i + 1
                while j < len(lines):
                    candidate = lines[j].strip()
                    if not candidate:
                        break
                    if candidate.startswith("- "):
                        item = candidate[2:].strip()
                        # Skip non-file entries like "No relevant documents"
                        if item.endswith(".md") or item.endswith(".txt"):
                            expected.append(item)
                    j += 1

                if query is not None:
                    cases.append(EvalCase(
                        query=query,
                        expected_files=expected,
                        category=current_category,
                    ))
                    query = None
                    expected = []
                i = j + 1
                continue

            i += 1

    logger.info("Parsed {} eval cases from {}", len(cases), filepath.name)
    return cases


def validate_retrieval(
    cases: list[EvalCase],
    retrieval_fn: callable,
    top_k: int = 5,
) -> ValidationSummary:
    """Run retrieval validation against eval cases.

    Args:
        cases: List of eval cases to validate.
        retrieval_fn: Function(query, top_k) -> list[Document].
            Each Document must have metadata["filename"].
        top_k: Number of results to retrieve per query.

    Returns:
        ValidationSummary with per-case results and aggregate metrics.
    """
    results: list[CaseResult] = []

    for case in cases:
        try:
            docs = retrieval_fn(case.query, top_k=top_k)
            retrieved_files = list(dict.fromkeys(
                doc.metadata.get("filename", "") for doc in docs
            ))
        except Exception as e:
            logger.error("Retrieval failed for '{}': {}", case.query[:60], e)
            retrieved_files = []

        if not case.expected_files:
            # Out-of-domain case: pass if retrieval returns nothing
            # or we accept any result (no strict enforcement)
            passed = True
            missing = []
        else:
            missing = [
                f for f in case.expected_files
                if f not in retrieved_files
            ]
            passed = len(missing) == 0

        results.append(CaseResult(
            query=case.query,
            expected_files=case.expected_files,
            retrieved_files=retrieved_files,
            passed=passed,
            missing_files=missing,
        ))

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    pass_rate = (passed_count / total * 100.0) if total > 0 else 0.0

    summary = ValidationSummary(
        total=total,
        passed=passed_count,
        failed=failed_count,
        pass_rate=pass_rate,
        results=results,
    )

    logger.info(
        "Retrieval validation: {}/{} passed ({:.1f}%)",
        passed_count, total, pass_rate,
    )

    return summary


def format_validation_report(summary: ValidationSummary) -> str:
    """Format validation summary as a readable text report.

    Args:
        summary: Validation summary to format.

    Returns:
        Human-readable report string.
    """
    lines = [
        "=== Retrieval Validation Report ===",
        f"Total cases: {summary.total}",
        f"Passed:      {summary.passed}",
        f"Failed:      {summary.failed}",
        f"Pass rate:   {summary.pass_rate:.1f}%",
        "",
    ]

    if summary.failed > 0:
        lines.append("--- Failed Cases ---")
        for r in summary.results:
            if not r.passed:
                lines.append(f"  Query: \"{r.query}\"")
                lines.append(f"    Missing: {r.missing_files}")
                lines.append(f"    Retrieved: {r.retrieved_files}")
                lines.append("")

    return "\n".join(lines)
