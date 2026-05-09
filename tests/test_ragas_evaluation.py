"""Tests for RAGAs content-quality evaluation module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from src.eval.ragas_evaluation import (
    ANSWER_CORRECTNESS_NOTE,
    DIAGNOSTIC_METRICS,
    LATEST_RESULTS_PATH,
    PRIMARY_METRICS,
    RAGAsEvalCase,
    RAGAsCaseResult,
    RAGAsReport,
    DEFAULT_CASES,
    _compute_averages,
    _evaluate_single_case,
    format_ragas_report,
    load_ragas_results,
    save_ragas_results,
    _fmt,
)


# ---------------------------------------------------------------------------
# Test data structures
# ---------------------------------------------------------------------------

class TestRAGAsEvalCase:
    def test_default_cases_count(self):
        assert len(DEFAULT_CASES) == 3

    def test_default_cases_topics(self):
        topics = [c.topic for c in DEFAULT_CASES]
        assert "LLM Basics and Prompt Engineering" in topics
        assert "RAG and Vector Databases" in topics
        assert "AI Agents and Tool Calling" in topics

    def test_default_cases_have_references(self):
        for case in DEFAULT_CASES:
            assert case.reference, f"Case '{case.topic}' missing reference"

    def test_default_cases_have_user_input(self):
        for case in DEFAULT_CASES:
            assert case.user_input, f"Case '{case.topic}' missing user_input"

    def test_rag_case_question_focused(self):
        rag_case = [c for c in DEFAULT_CASES if "RAG" in c.topic][0]
        assert "RAG pipeline" in rag_case.user_input

    def test_agents_case_question_focused(self):
        agent_case = [c for c in DEFAULT_CASES if "Agent" in c.topic][0]
        assert "tool calling" in agent_case.user_input.lower()

    def test_case_fields(self):
        case = RAGAsEvalCase(
            topic="Test", difficulty="beginner",
            user_input="question", reference="answer",
        )
        assert case.topic == "Test"
        assert case.difficulty == "beginner"
        assert case.user_input == "question"
        assert case.reference == "answer"

    def test_case_optional_reference(self):
        case = RAGAsEvalCase(
            topic="Test", difficulty="beginner",
            user_input="question",
        )
        assert case.reference == ""


class TestRAGAsCaseResult:
    def test_default_values(self):
        r = RAGAsCaseResult(topic="t", difficulty="d")
        assert r.faithfulness is None
        assert r.answer_relevancy is None
        assert r.context_precision is None
        assert r.context_recall is None
        assert r.answer_correctness is None
        assert r.num_contexts == 0
        assert r.answer_length == 0
        assert r.error is None

    def test_with_scores(self):
        r = RAGAsCaseResult(
            topic="t", difficulty="d",
            faithfulness=0.9, answer_relevancy=0.8,
            context_precision=0.7, context_recall=1.0,
            answer_correctness=0.6,
            num_contexts=10, answer_length=5000,
        )
        assert r.faithfulness == 0.9
        assert r.context_recall == 1.0


class TestRAGAsReport:
    def test_default_report(self):
        report = RAGAsReport()
        assert report.results == []
        assert report.avg_faithfulness is None

    def test_report_metadata_fields(self):
        report = RAGAsReport(timestamp="2025-01-01T00:00:00Z", model="gpt-4o-mini", case_count=3)
        assert report.timestamp == "2025-01-01T00:00:00Z"
        assert report.model == "gpt-4o-mini"
        assert report.case_count == 3


# ---------------------------------------------------------------------------
# Test compute averages
# ---------------------------------------------------------------------------

class TestComputeAverages:
    def test_all_scores_present(self):
        results = [
            RAGAsCaseResult(topic="a", difficulty="b", faithfulness=0.8, answer_relevancy=0.7),
            RAGAsCaseResult(topic="c", difficulty="d", faithfulness=1.0, answer_relevancy=0.9),
        ]
        avgs = _compute_averages(results)
        assert avgs["avg_faithfulness"] == 0.9
        assert avgs["avg_answer_relevancy"] == 0.8

    def test_partial_scores(self):
        results = [
            RAGAsCaseResult(topic="a", difficulty="b", faithfulness=0.8),
            RAGAsCaseResult(topic="c", difficulty="d", faithfulness=None),
        ]
        avgs = _compute_averages(results)
        assert avgs["avg_faithfulness"] == 0.8

    def test_no_scores(self):
        results = [
            RAGAsCaseResult(topic="a", difficulty="b"),
        ]
        avgs = _compute_averages(results)
        assert avgs["avg_faithfulness"] is None

    def test_empty_results(self):
        avgs = _compute_averages([])
        assert avgs["avg_faithfulness"] is None
        assert avgs["avg_answer_relevancy"] is None


# ---------------------------------------------------------------------------
# Test evaluate single case
# ---------------------------------------------------------------------------

class TestEvaluateSingleCase:
    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_error_content(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "", "contexts": [], "error": "generation failed"}
        result = self._run(_evaluate_single_case(case, content, []))
        assert result.error == "generation failed"
        assert result.faithfulness is None

    def test_empty_answer(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "", "contexts": ["ctx"], "error": None}
        result = self._run(_evaluate_single_case(case, content, []))
        assert result.error == "Empty answer or contexts"

    def test_empty_contexts(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "some answer", "contexts": [], "error": None}
        result = self._run(_evaluate_single_case(case, content, []))
        assert result.error == "Empty answer or contexts"

    def test_skips_reference_metrics_without_reference(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q", reference="")
        content = {"answer": "answer", "contexts": ["ctx"], "error": None}

        mock_metric = MagicMock()
        mock_metric.__class__.__name__ = "ContextRecall"
        type(mock_metric).__name__ = "ContextRecall"

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))
        # ContextRecall requires reference, so should be skipped
        assert result.context_recall is None

    def test_successful_metric_scoring(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "answer text", "contexts": ["ctx1"], "error": None}

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "AnswerRelevancy"
        mock_result = MagicMock()
        mock_result.value = 0.85
        mock_metric.ascore = AsyncMock(return_value=mock_result)

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))
        assert result.answer_relevancy == 0.85

    def test_metric_exception_handled(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "answer", "contexts": ["ctx"], "error": None}

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "Faithfulness"
        mock_metric.ascore = AsyncMock(side_effect=Exception("LLM error"))

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))
        assert result.faithfulness is None
        assert result.error is None  # metric failure != case error


# ---------------------------------------------------------------------------
# Test formatting
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_format_with_scores(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Test Topic", difficulty="beginner",
                    faithfulness=0.95, answer_relevancy=0.80,
                    context_precision=0.70, context_recall=1.0,
                    num_contexts=10, answer_length=5000,
                ),
            ],
            avg_faithfulness=0.95,
            avg_answer_relevancy=0.80,
            avg_context_precision=0.70,
            avg_context_recall=1.0,
        )
        text = format_ragas_report(report)
        assert "Test Topic" in text
        assert "0.9500" in text
        assert "✅" in text

    def test_format_with_error(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Failed", difficulty="beginner",
                    error="Something went wrong",
                ),
            ],
        )
        text = format_ragas_report(report)
        assert "ERROR" in text
        assert "Something went wrong" in text

    def test_format_below_threshold(self):
        report = RAGAsReport(
            results=[],
            avg_faithfulness=0.5,
            avg_answer_relevancy=0.4,
        )
        text = format_ragas_report(report)
        assert "⚠" in text
        assert "❌" in text

    def test_format_na_values(self):
        report = RAGAsReport(results=[])
        text = format_ragas_report(report)
        assert "N/A" in text


class TestFmt:
    def test_with_value(self):
        assert _fmt(0.9) == "0.9000"

    def test_with_none(self):
        assert _fmt(None) == "N/A"


# ---------------------------------------------------------------------------
# Test metric classification constants
# ---------------------------------------------------------------------------

class TestMetricClassification:
    def test_primary_metrics_contains_four(self):
        assert len(PRIMARY_METRICS) == 4
        assert "faithfulness" in PRIMARY_METRICS
        assert "answer_relevancy" in PRIMARY_METRICS
        assert "context_precision" in PRIMARY_METRICS
        assert "context_recall" in PRIMARY_METRICS

    def test_diagnostic_metrics_contains_correctness(self):
        assert "answer_correctness" in DIAGNOSTIC_METRICS

    def test_correctness_not_in_primary(self):
        assert "answer_correctness" not in PRIMARY_METRICS

    def test_answer_correctness_note_mentions_diagnostic(self):
        assert "diagnostic" in ANSWER_CORRECTNESS_NOTE.lower()


# ---------------------------------------------------------------------------
# Test cache save / load
# ---------------------------------------------------------------------------

class TestCacheSaveLoad:
    def _make_report(self):
        return RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Test", difficulty="beginner",
                    faithfulness=0.95, answer_relevancy=0.80,
                    num_contexts=5, answer_length=3000,
                ),
            ],
            avg_faithfulness=0.95,
            avg_answer_relevancy=0.80,
            timestamp="2025-05-09T00:00:00Z",
            model="gpt-4o-mini",
            case_count=1,
        )

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "test_results.json"
        report = self._make_report()
        save_ragas_results(report, path=path)
        assert path.exists()

    def test_round_trip(self, tmp_path):
        path = tmp_path / "test_results.json"
        original = self._make_report()
        save_ragas_results(original, path=path)
        loaded = load_ragas_results(path=path)
        assert loaded is not None
        assert loaded.avg_faithfulness == original.avg_faithfulness
        assert loaded.timestamp == original.timestamp
        assert loaded.model == original.model
        assert loaded.case_count == original.case_count
        assert len(loaded.results) == 1
        assert loaded.results[0].topic == "Test"
        assert loaded.results[0].faithfulness == 0.95

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_ragas_results(path=path) is None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all")
        assert load_ragas_results(path=path) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "results.json"
        save_ragas_results(self._make_report(), path=path)
        assert path.exists()

    def test_latest_results_path_is_in_data_eval(self):
        assert "data" in LATEST_RESULTS_PATH.parts
        assert "eval" in LATEST_RESULTS_PATH.parts
        assert LATEST_RESULTS_PATH.name == "latest_ragas_eval.json"


# ---------------------------------------------------------------------------
# Test format_ragas_report quality assessment section
# ---------------------------------------------------------------------------

class TestFormatReportQualityAssessment:
    def test_includes_primary_assessment_label(self):
        report = RAGAsReport(
            results=[],
            avg_faithfulness=0.9, avg_answer_relevancy=0.8,
            avg_context_precision=0.7, avg_context_recall=1.0,
        )
        text = format_ragas_report(report)
        assert "Primary Metrics" in text

    def test_includes_context_recall_in_assessment(self):
        report = RAGAsReport(
            results=[],
            avg_faithfulness=0.9, avg_answer_relevancy=0.8,
            avg_context_precision=0.7, avg_context_recall=0.3,
        )
        text = format_ragas_report(report)
        assert "Context Recall" in text
        assert "⚠" in text

    def test_diagnostic_note_in_report(self):
        report = RAGAsReport(results=[])
        text = format_ragas_report(report)
        assert "diagnostic" in text.lower()
