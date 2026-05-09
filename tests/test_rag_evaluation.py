"""Tests for the offline RAG evaluation module."""

from dataclasses import dataclass

import pytest

from src.eval.retrieval_validation import EvalCase, CaseResult, ValidationSummary
from src.eval.rag_evaluation import (
    SourceCoverage,
    RAGEvalReport,
    compute_source_coverage,
    run_rag_evaluation,
    format_rag_eval_report,
    _check_ragas_available,
    _ragas_availability_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeDoc:
    metadata: dict


def _make_retrieval_fn(mapping: dict[str, list[str]]):
    """Build a mock retrieval function from query -> filename mapping."""
    def fn(query: str, top_k: int = 5):
        filenames = mapping.get(query, [])
        return [_FakeDoc(metadata={"filename": f}) for f in filenames[:top_k]]
    return fn


# ---------------------------------------------------------------------------
# Source Coverage
# ---------------------------------------------------------------------------

class TestSourceCoverage:
    def test_full_coverage(self):
        cases = [
            EvalCase(query="q1", expected_files=["a.md", "b.md"]),
            EvalCase(query="q2", expected_files=["c.md"]),
        ]
        results = [
            CaseResult(query="q1", expected_files=["a.md", "b.md"],
                       retrieved_files=["a.md", "b.md"], passed=True),
            CaseResult(query="q2", expected_files=["c.md"],
                       retrieved_files=["c.md", "d.md"], passed=True),
        ]
        cov = compute_source_coverage(cases, results)
        assert cov.total_expected_files == 3
        assert cov.coverage_rate == 100.0
        assert cov.files_missed == []
        assert sorted(cov.files_hit) == ["a.md", "b.md", "c.md"]

    def test_partial_coverage(self):
        cases = [
            EvalCase(query="q1", expected_files=["a.md", "b.md"]),
        ]
        results = [
            CaseResult(query="q1", expected_files=["a.md", "b.md"],
                       retrieved_files=["a.md"], passed=False,
                       missing_files=["b.md"]),
        ]
        cov = compute_source_coverage(cases, results)
        assert cov.total_expected_files == 2
        assert cov.files_hit == ["a.md"]
        assert cov.files_missed == ["b.md"]
        assert cov.coverage_rate == 50.0

    def test_no_coverage(self):
        cases = [EvalCase(query="q1", expected_files=["a.md"])]
        results = [
            CaseResult(query="q1", expected_files=["a.md"],
                       retrieved_files=[], passed=False,
                       missing_files=["a.md"]),
        ]
        cov = compute_source_coverage(cases, results)
        assert cov.coverage_rate == 0.0
        assert cov.files_missed == ["a.md"]

    def test_empty_cases(self):
        cov = compute_source_coverage([], [])
        assert cov.total_expected_files == 0
        assert cov.coverage_rate == 0.0

    def test_out_of_domain_cases_ignored(self):
        cases = [
            EvalCase(query="q1", expected_files=[]),  # out-of-domain
            EvalCase(query="q2", expected_files=["a.md"]),
        ]
        results = [
            CaseResult(query="q1", expected_files=[], retrieved_files=[],
                       passed=True),
            CaseResult(query="q2", expected_files=["a.md"],
                       retrieved_files=["a.md"], passed=True),
        ]
        cov = compute_source_coverage(cases, results)
        assert cov.total_expected_files == 1
        assert cov.coverage_rate == 100.0

    def test_deduplicates_expected_files(self):
        cases = [
            EvalCase(query="q1", expected_files=["a.md"]),
            EvalCase(query="q2", expected_files=["a.md", "b.md"]),
        ]
        results = [
            CaseResult(query="q1", expected_files=["a.md"],
                       retrieved_files=["a.md"], passed=True),
            CaseResult(query="q2", expected_files=["a.md", "b.md"],
                       retrieved_files=["a.md", "b.md"], passed=True),
        ]
        cov = compute_source_coverage(cases, results)
        assert cov.total_expected_files == 2
        assert cov.unique_expected_files == ["a.md", "b.md"]


# ---------------------------------------------------------------------------
# Run RAG Evaluation
# ---------------------------------------------------------------------------

class TestRunRagEvaluation:
    def test_with_mock_retrieval(self, tmp_path):
        eval_file = tmp_path / "cases.md"
        eval_file.write_text(
            '## Test\n\nQuery:\n"What is RAG?"\n\n'
            'Expected:\n- rag_basics.md\n\n---\n\n'
            'Query:\n"What is an agent?"\n\n'
            'Expected:\n- ai_agents_intro.md\n'
        )
        retrieval_fn = _make_retrieval_fn({
            "What is RAG?": ["rag_basics.md"],
            "What is an agent?": ["ai_agents_intro.md"],
        })
        report = run_rag_evaluation(eval_file, retrieval_fn)
        assert report.retrieval.total == 2
        assert report.retrieval.passed == 2
        assert report.retrieval.pass_rate == 100.0
        assert report.coverage.coverage_rate == 100.0

    def test_with_no_retrieval_fn(self, tmp_path):
        eval_file = tmp_path / "cases.md"
        eval_file.write_text(
            '## Test\n\nQuery:\n"What is RAG?"\n\n'
            'Expected:\n- rag_basics.md\n'
        )
        report = run_rag_evaluation(eval_file, retrieval_fn=None)
        assert report.retrieval.total == 1
        assert report.retrieval.failed == 1
        assert report.coverage.coverage_rate == 0.0

    def test_empty_eval_file(self, tmp_path):
        eval_file = tmp_path / "empty.md"
        eval_file.write_text("# Empty\n")
        report = run_rag_evaluation(eval_file)
        assert report.retrieval.total == 0
        assert report.coverage.total_expected_files == 0

    def test_missing_eval_file(self, tmp_path):
        report = run_rag_evaluation(tmp_path / "nonexistent.md")
        assert report.retrieval.total == 0

    def test_ragas_note_present(self, tmp_path):
        eval_file = tmp_path / "cases.md"
        eval_file.write_text("# Empty\n")
        report = run_rag_evaluation(eval_file)
        assert report.ragas_note  # should be non-empty


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------

class TestFormatRagEvalReport:
    def test_passing_report(self):
        summary = ValidationSummary(total=2, passed=2, failed=0, pass_rate=100.0)
        coverage = SourceCoverage(
            total_expected_files=2,
            unique_expected_files=["a.md", "b.md"],
            files_hit=["a.md", "b.md"],
            files_missed=[],
            coverage_rate=100.0,
        )
        report = RAGEvalReport(retrieval=summary, coverage=coverage)
        text = format_rag_eval_report(report)
        assert "Hit rate:       100.0%" in text
        assert "Coverage rate:  100.0%" in text
        assert "Failed Cases" not in text
        assert "Missed files" not in text

    def test_failing_report_shows_details(self):
        results = [
            CaseResult(query="q1", expected_files=["a.md"],
                       retrieved_files=["b.md"], passed=False,
                       missing_files=["a.md"]),
        ]
        summary = ValidationSummary(
            total=1, passed=0, failed=1, pass_rate=0.0, results=results,
        )
        coverage = SourceCoverage(
            total_expected_files=1,
            unique_expected_files=["a.md"],
            files_hit=[],
            files_missed=["a.md"],
            coverage_rate=0.0,
        )
        report = RAGEvalReport(retrieval=summary, coverage=coverage)
        text = format_rag_eval_report(report)
        assert "Failed Cases" in text
        assert "Missed files" in text
        assert "a.md" in text

    def test_ragas_note_in_report(self):
        summary = ValidationSummary(total=0, passed=0, failed=0, pass_rate=0.0)
        coverage = SourceCoverage(
            total_expected_files=0, unique_expected_files=[],
            files_hit=[], files_missed=[], coverage_rate=0.0,
        )
        report = RAGEvalReport(
            retrieval=summary, coverage=coverage,
            ragas_note="Test note about RAGAs",
        )
        text = format_rag_eval_report(report)
        assert "Test note about RAGAs" in text


# ---------------------------------------------------------------------------
# RAGAs Availability
# ---------------------------------------------------------------------------

class TestRagasAvailability:
    def test_ragas_installed(self):
        assert _check_ragas_available() is True

    def test_ragas_note_confirms_installed(self):
        note = _ragas_availability_note()
        assert "installed" in note.lower()

    def test_report_has_ragas_flag(self, tmp_path):
        eval_file = tmp_path / "cases.md"
        eval_file.write_text("# Empty\n")
        report = run_rag_evaluation(eval_file)
        assert report.has_ragas is True
