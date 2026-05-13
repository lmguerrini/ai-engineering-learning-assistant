"""Tests for RAGAs content-quality evaluation module."""

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.ragas_evaluation import (
    ANSWER_CORRECTNESS_NOTE,
    DIAGNOSTIC_METRICS,
    LEARN_PATH_EVAL_TOPIC_MAP,
    LATEST_RESULTS_PATH,
    MAX_EVAL_ANSWER_CHARS,
    MAX_EVAL_CONTEXT_CHARS,
    PRIMARY_METRICS,
    DEFAULT_CASES,
    RAGAsCaseResult,
    RAGAsEvalCase,
    RAGAsReport,
    _compute_averages,
    _compute_category_averages,
    _evaluate_single_case,
    _format_metric_error_reason,
    format_ragas_report,
    load_ragas_results,
    run_ragas_evaluation,
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

    def test_agents_case_is_advanced(self):
        agent_case = [c for c in DEFAULT_CASES if "Agent" in c.topic][0]
        assert agent_case.difficulty == "advanced"

    def test_default_cases_are_all_learn_path(self):
        surfaces = {c.surface for c in DEFAULT_CASES}
        assert "learn_path" in surfaces
        assert surfaces == {"learn_path"}

    def test_default_cases_have_no_pending_label_suffixes(self):
        assert all(case.label_suffix == "" for case in DEFAULT_CASES)

    def test_case_fields(self):
        case = RAGAsEvalCase(
            topic="Test", difficulty="beginner",
            user_input="question", reference="answer",
        )
        assert case.topic == "Test"
        assert case.difficulty == "beginner"
        assert case.user_input == "question"
        assert case.surface == "topic_mode"
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
        assert r.surface == "topic_mode"
        assert r.faithfulness is None
        assert r.answer_relevancy is None
        assert r.context_precision is None
        assert r.context_recall is None
        assert r.answer_correctness is None
        assert r.num_contexts == 0
        assert r.answer_length == 0
        assert r.contexts_count == 0
        assert r.answer_length_original == 0
        assert r.answer_length_evaluated == 0
        assert r.was_truncated is False
        assert r.metric_errors == {}
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
        assert report.category_averages == {}
        assert report.global_averages == {}

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


class TestComputeCategoryAverages:
    def test_groups_by_surface_and_excludes_na_metrics(self):
        results = [
            RAGAsCaseResult(
                topic="lp1",
                difficulty="beginner",
                surface="learn_path",
                faithfulness=0.8,
                answer_relevancy=0.7,
            ),
            RAGAsCaseResult(
                topic="lp2",
                difficulty="intermediate",
                surface="learn_path",
                faithfulness=1.0,
                answer_relevancy=None,
            ),
            RAGAsCaseResult(
                topic="ha1",
                difficulty="",
                surface="help_assistant",
                faithfulness=0.2,
                answer_relevancy=0.1,
            ),
        ]

        avgs = _compute_category_averages(results)

        assert avgs["learn_path"]["avg_faithfulness"] == 0.9
        assert avgs["learn_path"]["avg_answer_relevancy"] == 0.7
        assert avgs["help_assistant"]["avg_faithfulness"] == 0.2


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
        assert result.error == "Generated content invalid: empty answer."

    def test_empty_contexts(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "A" * 4000, "contexts": [], "error": None}
        result = self._run(_evaluate_single_case(case, content, []))
        assert result.error == "Generated content invalid: no evaluation contexts were produced."

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
        content = {"answer": "A" * 4000, "contexts": ["ctx1"], "error": None}

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "AnswerRelevancy"
        mock_result = MagicMock()
        mock_result.value = 0.85
        mock_metric.ascore = AsyncMock(return_value=mock_result)

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))
        assert result.answer_relevancy == 0.85

    def test_metric_exception_handled(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q")
        content = {"answer": "A" * 4000, "contexts": ["ctx"], "error": None}

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "Faithfulness"
        mock_metric.ascore = AsyncMock(side_effect=Exception("LLM error"))

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))
        assert result.faithfulness is None
        assert result.error is None  # metric failure != case error
        assert result.metric_errors["faithfulness"] == "judge failed: llm error"

    def test_truncates_long_answer_and_contexts_before_judge_calls(self):
        case = RAGAsEvalCase(topic="t", difficulty="beginner", surface="learn_path", user_input="q")
        long_answer = "A" * (MAX_EVAL_ANSWER_CHARS + 250)
        long_contexts = [
            "B" * (MAX_EVAL_CONTEXT_CHARS - 10),
            "C" * 100,
        ]
        content = {"answer": long_answer, "contexts": long_contexts, "error": None}

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "Faithfulness"
        metric_result = MagicMock()
        metric_result.value = 0.91
        mock_metric.ascore = AsyncMock(return_value=metric_result)

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))

        kwargs = mock_metric.ascore.await_args.kwargs
        assert 3000 <= len(kwargs["response"]) <= MAX_EVAL_ANSWER_CHARS
        assert len(kwargs["response"]) < len(long_answer)
        assert sum(len(ctx) for ctx in kwargs["retrieved_contexts"]) == MAX_EVAL_CONTEXT_CHARS
        assert result.answer_length_original == len(long_answer)
        assert result.answer_length_evaluated == len(kwargs["response"])
        assert result.contexts_count == 2
        assert result.was_truncated is True
        assert content["eval_answer"] == kwargs["response"]
        assert content["eval_contexts"] == kwargs["retrieved_contexts"]

    def test_metric_length_failure_is_stored_as_judge_failure(self):
        case = RAGAsEvalCase(topic="t", difficulty="d", user_input="q", reference="ref")
        content = {"answer": "A" * 4000, "contexts": ["ctx"], "error": None}

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "AnswerCorrectness"
        mock_metric.ascore = AsyncMock(
            side_effect=Exception("output is incomplete due to a max_tokens length limit")
        )

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))

        assert result.answer_correctness is None
        assert result.metric_errors["answer_correctness"] == "judge failed: max tokens limit"

    def test_short_learn_path_output_is_invalidated_before_metrics(self):
        case = RAGAsEvalCase(
            topic="LLM Basics and Prompt Engineering",
            difficulty="beginner",
            surface="learn_path",
            user_input="q",
        )
        content = {
            "answer": "Short fallback answer" * 10,
            "contexts": ["ctx"],
            "error": None,
            "trace": ["generate_study_guide: cache miss"],
            "generation_failed": False,
        }

        mock_metric = MagicMock()
        type(mock_metric).__name__ = "Faithfulness"
        mock_metric.ascore = AsyncMock()

        result = self._run(_evaluate_single_case(case, content, [mock_metric]))

        assert result.error == "Generated content invalid: answer too short for learn_path (210 chars, min 3000)."
        mock_metric.ascore.assert_not_called()


class TestEvalCaseGeneration:
    @patch("src.graphs.learn_graph.run_learn_workflow")
    def test_generate_learn_content_uses_actual_learn_path_topic_map(self, mock_run):
        from src.eval.ragas_evaluation import _generate_eval_case_content
        from src.schemas import DifficultyLevel, ResponseStyle

        mock_run.return_value = {
            "study_guide": MagicMock(summary="Overview", detailed_notes="A" * 4000),
            "retrieved_docs": [MagicMock(content="ctx")],
            "trace": ["generate_study_guide: success"],
            "generation_failed": False,
        }

        content = _generate_eval_case_content(
            RAGAsEvalCase(
                topic="LLM Basics and Prompt Engineering",
                difficulty="beginner",
                surface="learn_path",
                user_input="Explain the fundamentals of large language models and basic prompt engineering techniques.",
                reference="reference",
            )
        )

        mock_run.assert_called_once_with(
            topic=LEARN_PATH_EVAL_TOPIC_MAP["beginner"],
            difficulty=DifficultyLevel.BEGINNER,
            style=ResponseStyle.DETAILED,
            force_regenerate=True,
        )
        assert content["workflow_topic"] == LEARN_PATH_EVAL_TOPIC_MAP["beginner"]
        assert len(content["answer"]) > 3000

    @patch("src.services.help_assistant.get_help_assistant_app_workflow_context")
    @patch("src.services.help_assistant.get_help_assistant_runtime_defaults")
    @patch("src.services.help_assistant.answer_help_question")
    def test_generate_help_assistant_content_uses_workflow_context_when_no_sources(
        self,
        mock_answer,
        mock_defaults,
        mock_app_context,
    ):
        from src.eval.ragas_evaluation import RAGAsEvalCase, _generate_eval_case_content

        mock_defaults.return_value = {
            "temperature": 0.15,
            "top_p": 0.85,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 1100,
        }
        mock_app_context.return_value = "App workflow context block."
        mock_answer.return_value = {
            "status": "answered",
            "answer_markdown": "Official snapshots are local; live docs are fetched on demand.",
            "sources": [],
        }

        content = _generate_eval_case_content(
            RAGAsEvalCase(
                topic="Help Assistant Official Docs Workflow",
                difficulty="intermediate",
                surface="help_assistant",
                user_input="What is the difference between official snapshots and live docs enrichment?",
                reference="reference",
            )
        )

        assert content["answer"].startswith("Official snapshots")
        assert content["contexts"] == ["App workflow context block."]
        assert content["sources_count"] == 1


class TestRunRagasEvaluation:
    def test_learn_path_readiness_stays_separate_from_diagnostic_cases(self):
        class DummyMetric:
            def __init__(self, *args, **kwargs):
                pass

        class DummyAsyncOpenAI:
            def __init__(self, *args, **kwargs):
                pass

        fake_collections = ModuleType("ragas.metrics.collections")
        fake_collections.Faithfulness = DummyMetric
        fake_collections.AnswerRelevancy = DummyMetric
        fake_collections.ContextPrecision = DummyMetric
        fake_collections.ContextRecall = DummyMetric
        fake_collections.AnswerCorrectness = DummyMetric

        fake_llms = ModuleType("ragas.llms")
        fake_llms.llm_factory = lambda *args, **kwargs: object()

        fake_embeddings = ModuleType("ragas.embeddings.base")
        fake_embeddings.embedding_factory = lambda *args, **kwargs: object()

        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = DummyAsyncOpenAI

        fake_modules = {
            "ragas.metrics.collections": fake_collections,
            "ragas.llms": fake_llms,
            "ragas.embeddings.base": fake_embeddings,
            "openai": fake_openai,
        }

        synthetic_results = [
            RAGAsCaseResult(
                topic="Learn Path 1",
                difficulty="beginner",
                surface="learn_path",
                faithfulness=0.9,
                answer_relevancy=0.8,
                context_precision=0.7,
                context_recall=0.9,
            ),
            RAGAsCaseResult(
                topic="Learn Path 2",
                difficulty="intermediate",
                surface="learn_path",
                faithfulness=0.8,
                answer_relevancy=0.7,
                context_precision=0.6,
                context_recall=0.8,
            ),
            RAGAsCaseResult(
                topic="Topic Mode 1",
                difficulty="",
                surface="topic_mode",
                faithfulness=0.3,
                answer_relevancy=0.2,
                context_precision=0.1,
                context_recall=0.4,
            ),
            RAGAsCaseResult(
                topic="Help Assistant 1",
                difficulty="",
                surface="help_assistant",
                faithfulness=None,
                answer_relevancy=None,
                context_precision=None,
                context_recall=None,
                metric_errors={"faithfulness": "judge failed: max tokens limit"},
            ),
        ]

        cases = [
            RAGAsEvalCase(topic="Learn Path 1", difficulty="beginner", user_input="q", surface="learn_path"),
            RAGAsEvalCase(topic="Learn Path 2", difficulty="intermediate", user_input="q", surface="learn_path"),
            RAGAsEvalCase(topic="Topic Mode 1", difficulty="", user_input="q", surface="topic_mode"),
            RAGAsEvalCase(topic="Help Assistant 1", difficulty="", user_input="q", surface="help_assistant"),
        ]

        async def fake_evaluate(case, content, metrics):
            return synthetic_results.pop(0)

        with patch.dict(sys.modules, fake_modules), patch(
            "src.eval.ragas_evaluation._generate_eval_case_content",
            return_value={"answer": "answer", "contexts": ["ctx"], "sources_count": 1, "error": None},
        ), patch(
            "src.eval.ragas_evaluation._evaluate_single_case",
            side_effect=fake_evaluate,
        ), patch(
            "src.eval.ragas_evaluation.save_ragas_results",
        ):
            report = run_ragas_evaluation(cases=cases)

        assert report.avg_faithfulness == 0.6667
        assert report.avg_answer_relevancy == 0.5667
        assert report.global_averages["avg_faithfulness"] == 0.6667
        assert report.global_averages["avg_answer_relevancy"] == 0.5667
        assert report.category_averages["learn_path"]["avg_faithfulness"] == 0.85
        assert report.category_averages["learn_path"]["avg_answer_relevancy"] == 0.75
        assert report.category_averages["topic_mode"]["avg_answer_relevancy"] == 0.2
        assert report.category_averages["help_assistant"]["avg_faithfulness"] is None


# ---------------------------------------------------------------------------
# Test formatting
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_format_with_scores(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Test Topic", difficulty="beginner",
                    surface="learn_path",
                    faithfulness=0.95, answer_relevancy=0.80,
                    context_precision=0.70, context_recall=1.0,
                    num_contexts=10, answer_length=5000,
                ),
            ],
            avg_faithfulness=0.95,
            avg_answer_relevancy=0.80,
            avg_context_precision=0.70,
            avg_context_recall=1.0,
            category_averages={
                "learn_path": {
                    "avg_faithfulness": 0.95,
                    "avg_answer_relevancy": 0.8,
                    "avg_context_precision": 0.7,
                    "avg_context_recall": 1.0,
                    "avg_answer_correctness": None,
                }
            },
            global_averages={
                "avg_faithfulness": 0.95,
                "avg_answer_relevancy": 0.8,
                "avg_context_precision": 0.7,
                "avg_context_recall": 1.0,
                "avg_answer_correctness": None,
            },
        )
        text = format_ragas_report(report)
        assert "Test Topic" in text
        assert "0.9500" in text
        assert "✅" in text
        assert "(Beginner, Learn Path)" in text
        assert "Learn Path Averages" in text
        assert "Category Averages" in text
        assert "Overall Evaluated Score" in text

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

    def test_format_includes_truncation_and_judge_failure_reason(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Long Case",
                    difficulty="advanced",
                    surface="topic_mode",
                    faithfulness=None,
                    answer_relevancy=0.61,
                    context_precision=0.55,
                    context_recall=0.72,
                    num_contexts=3,
                    answer_length=9000,
                    contexts_count=3,
                    answer_length_original=9000,
                    answer_length_evaluated=MAX_EVAL_ANSWER_CHARS,
                    was_truncated=True,
                    metric_errors={"faithfulness": "judge failed: max tokens limit"},
                )
            ]
        )

        text = format_ragas_report(report)

        assert "Truncated for judge safety" in text
        assert f"Evaluated answer length: {MAX_EVAL_ANSWER_CHARS} chars" in text
        assert "Faithfulness:" in text
        assert "N/A — judge failed: max tokens limit" in text


class TestFmt:
    def test_with_value(self):
        assert _fmt(0.9) == "0.9000"

    def test_with_none(self):
        assert _fmt(None) == "N/A"


class TestMetricErrorFormatting:
    def test_detects_max_tokens_length_failures(self):
        reason = _format_metric_error_reason(
            Exception("output is incomplete due to a max_tokens length limit")
        )
        assert reason == "judge failed: max tokens limit"


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
        assert "faithfulness" in ANSWER_CORRECTNESS_NOTE.lower()


# ---------------------------------------------------------------------------
# Test cache save / load
# ---------------------------------------------------------------------------

class TestCacheSaveLoad:
    def _make_report(self):
        return RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Test", difficulty="beginner",
                    surface="learn_path",
                    faithfulness=0.95, answer_relevancy=0.80,
                    num_contexts=5, answer_length=3000,
                ),
            ],
            avg_faithfulness=0.95,
            avg_answer_relevancy=0.80,
            category_averages={
                "learn_path": {
                    "avg_faithfulness": 0.95,
                    "avg_answer_relevancy": 0.8,
                    "avg_context_precision": None,
                    "avg_context_recall": None,
                    "avg_answer_correctness": None,
                }
            },
            global_averages={
                "avg_faithfulness": 0.95,
                "avg_answer_relevancy": 0.8,
                "avg_context_precision": None,
                "avg_context_recall": None,
                "avg_answer_correctness": None,
            },
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
        assert loaded.category_averages == original.category_averages
        assert loaded.global_averages == original.global_averages
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

    def test_diagnostic_note_not_rendered_in_raw_report(self):
        report = RAGAsReport(results=[])
        text = format_ragas_report(report)
        assert "diagnostic" not in text.lower()

    def test_format_includes_surface_label(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Difference between RAG and Agentic RAG",
                    difficulty="intermediate",
                    surface="help_assistant",
                )
            ]
        )
        text = format_ragas_report(report)
        assert "Help Assistant" in text

    def test_format_with_configured_cases_shows_pending_placeholders(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="LLM Basics and Prompt Engineering",
                    difficulty="beginner",
                    faithfulness=0.95,
                    answer_relevancy=0.8,
                    context_precision=0.7,
                    context_recall=1.0,
                )
            ]
        )
        text = format_ragas_report(report, configured_cases=DEFAULT_CASES)
        assert "LLM Basics and Prompt Engineering (Beginner, Learn Path)" in text
        assert "Status: Pending evaluation" in text

    def test_partial_benchmark_warning_when_some_primary_metrics_are_unavailable(self):
        report = RAGAsReport(
            results=[],
            avg_faithfulness=0.9,
            avg_answer_relevancy=0.8,
            avg_context_precision=None,
            avg_context_recall=0.95,
            category_averages={
                "learn_path": {
                    "avg_faithfulness": 0.9,
                    "avg_answer_relevancy": 0.8,
                    "avg_context_precision": None,
                    "avg_context_recall": 0.95,
                    "avg_answer_correctness": None,
                }
            },
            global_averages={
                "avg_faithfulness": 0.9,
                "avg_answer_relevancy": 0.8,
                "avg_context_precision": None,
                "avg_context_recall": 0.95,
                "avg_answer_correctness": None,
            },
        )

        text = format_ragas_report(report)

        assert "Benchmark partially complete" in text
        assert "Judge-limited metrics: Context Precision." in text

    def test_all_primary_metric_judge_failures_require_review(self):
        report = RAGAsReport(
            results=[],
            avg_faithfulness=None,
            avg_answer_relevancy=None,
            avg_context_precision=None,
            avg_context_recall=None,
        )

        text = format_ragas_report(report)

        assert "Benchmark incomplete" in text

    def test_generation_failures_are_called_out_in_quality_assessment(self):
        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Broken Learn Path",
                    difficulty="beginner",
                    surface="learn_path",
                    error="Generated content invalid: answer too short for learn_path (371 chars, min 3000).",
                )
            ],
            avg_faithfulness=0.8,
            avg_answer_relevancy=0.7,
            avg_context_precision=0.65,
            avg_context_recall=0.9,
        )

        text = format_ragas_report(report)

        assert "Case generation failures: 1 case(s) produced invalid or short output and were not scored." in text
