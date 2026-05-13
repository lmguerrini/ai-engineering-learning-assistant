"""Tests for the RAGAs evaluation section in the Dashboard."""

from unittest.mock import patch, MagicMock

import pytest

from src.ui.dashboard_page import (
    _build_capability_registry_rows,
    _check_ragas_available,
    _display_capability_registry_section,
    _format_feedback_signal_message,
    _metric_color,
    _fmt_metric,
    _format_ragas_case_label,
    _metric_status_label,
    _ragas_snapshot_value,
)


# ---------------------------------------------------------------------------
# _check_ragas_available
# ---------------------------------------------------------------------------

class TestCheckRagasAvailable:
    """Availability check for ragas + OpenAI key."""

    def test_returns_true_when_both_present(self):
        ok, msg = _check_ragas_available()
        # ragas is installed in the test env and settings have a key
        assert ok is True
        assert msg == ""

    def test_returns_false_when_ragas_missing(self):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ragas":
                raise ImportError("no ragas")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            ok, msg = _check_ragas_available()
        assert ok is False
        assert "ragas" in msg.lower()

    def test_returns_false_when_api_key_missing(self):
        mock_settings = MagicMock()
        mock_settings.openai_api_key = ""
        with patch("src.config.get_settings", return_value=mock_settings):
            ok, msg = _check_ragas_available()
        assert ok is False
        assert "API key" in msg


# ---------------------------------------------------------------------------
# _metric_color
# ---------------------------------------------------------------------------

class TestMetricColor:
    """Emoji indicators for metric values."""

    def test_none_returns_neutral(self):
        assert _metric_color(None) == "⬜"

    def test_high_value_green(self):
        assert _metric_color(0.85) == "🟢"

    def test_threshold_boundary_green(self):
        assert _metric_color(0.6) == "🟢"

    def test_medium_value_yellow(self):
        assert _metric_color(0.5) == "🟡"

    def test_low_value_red(self):
        assert _metric_color(0.2) == "🔴"

    def test_zero_value_red(self):
        assert _metric_color(0.0) == "🔴"

    def test_perfect_value_green(self):
        assert _metric_color(1.0) == "🟢"

    def test_custom_threshold(self):
        assert _metric_color(0.7, threshold=0.8) == "🟡"
        assert _metric_color(0.9, threshold=0.8) == "🟢"


# ---------------------------------------------------------------------------
# _fmt_metric
# ---------------------------------------------------------------------------

class TestFmtMetric:
    """Format metric values for display."""

    def test_none_returns_na(self):
        assert _fmt_metric(None) == "N/A"

    def test_float_formatted_to_4_decimals(self):
        assert _fmt_metric(0.8846) == "0.8846"

    def test_zero(self):
        assert _fmt_metric(0.0) == "0.0000"

    def test_one(self):
        assert _fmt_metric(1.0) == "1.0000"

    def test_rounding(self):
        assert _fmt_metric(0.12345) == "0.1235"


# ---------------------------------------------------------------------------
# Dashboard snapshot helpers
# ---------------------------------------------------------------------------

class TestDashboardSnapshotHelpers:
    """Short reviewer-facing snapshot labels should stay clear."""

    def test_ragas_snapshot_value_ready_when_report_present(self):
        assert _ragas_snapshot_value(object()) == "Cached"

    def test_ragas_snapshot_value_not_run_without_report(self):
        assert _ragas_snapshot_value(None) == "Not run"

    def test_metric_status_label_uses_normal_status_dot_for_answer_correctness(self):
        assert _metric_status_label("answer_correctness", 0.1) == "🔴"

    @patch("src.ui.dashboard_page.st")
    def test_trace_snapshot_value_uses_dash_when_no_run_available(self, mock_st):
        from src.ui.dashboard_page import _trace_snapshot_value

        mock_st.session_state = {}
        assert _trace_snapshot_value({}, "last_learn_trace") == "—"

    def test_ragas_case_label_capitalizes_difficulty(self):
        case = MagicMock(topic="AI Agents and Tool Calling", difficulty="advanced")
        assert _format_ragas_case_label(case) == "AI Agents and Tool Calling (Advanced, Learn Path)"

    def test_ragas_case_label_uses_topic_mode_suffix_without_difficulty(self):
        case = MagicMock(topic="LangGraph", difficulty="", surface="topic_mode")
        assert _format_ragas_case_label(case) == "LangGraph (Topic Mode)"

    def test_feedback_signal_message_is_human_readable(self):
        assert _format_feedback_signal_message("increase_difficulty") == (
            "Current feedback signal: increase difficulty based on saved feedback."
        )
        assert _format_feedback_signal_message("simplify") == (
            "Current feedback signal: simplify upcoming explanations based on saved feedback."
        )
        assert _format_feedback_signal_message(None) is None


# ---------------------------------------------------------------------------
# Capability / Tool Registry
# ---------------------------------------------------------------------------

class TestCapabilityRegistry:
    """Reviewer-facing capability registry should stay explicit and accurate."""

    def test_build_capability_registry_rows_covers_expected_capabilities(self):
        rows = _build_capability_registry_rows(
            kb_health={
                "collections": {
                    "curated": {"chunk_count": 257},
                    "official": {"chunk_count": 305},
                }
            },
            tracing_info={"tracing_enabled": True},
            ragas_available=True,
            memory_loaded=False,
            feedback_count=0,
            progressive_streaming=True,
            cache_bypass=False,
            has_api_key=True,
        )

        assert [row["Capability"] for row in rows] == [
            "Curated KB Retrieval",
            "Official Docs Retrieval",
            "Memory Profile",
            "Feedback Logger",
            "Cost Tracker",
            "LangSmith Tracing",
            "RAGAs Evaluator",
            "KB Rebuild Tool",
            "Progressive Streaming",
            "Cache Bypass",
        ]

        official = next(row for row in rows if row["Capability"] == "Official Docs Retrieval")
        memory = next(row for row in rows if row["Capability"] == "Memory Profile")
        feedback = next(row for row in rows if row["Capability"] == "Feedback Logger")
        cost = next(row for row in rows if row["Capability"] == "Cost Tracker")
        tracing = next(row for row in rows if row["Capability"] == "LangSmith Tracing")
        ragas = next(row for row in rows if row["Capability"] == "RAGAs Evaluator")
        rebuild = next(row for row in rows if row["Capability"] == "KB Rebuild Tool")
        progressive = next(row for row in rows if row["Capability"] == "Progressive Streaming")
        bypass = next(row for row in rows if row["Capability"] == "Cache Bypass")

        assert official["Status"] == "Ready"
        assert official["Mode"] == "Optional"
        assert official["User Control"] == "System-managed"
        assert memory["Status"] == "Ready"
        assert feedback["Status"] == "Ready"
        assert cost["User Control"] == "System-managed"
        assert tracing["User Control"] == "Environment-controlled"
        assert ragas["Status"] == "Ready"
        assert rebuild["Status"] == "Ready"
        assert progressive["Status"] == "Active"
        assert progressive["User Control"] == "Controlled in Learn"
        assert bypass["Status"] == "Off"

    @patch("src.ui.dashboard_page.st")
    def test_display_capability_registry_section_renders_table(self, mock_st):
        _display_capability_registry_section(
            kb_health={
                "collections": {
                    "curated": {"chunk_count": 257},
                    "official": {"chunk_count": 305},
                }
            },
            tracing_info={"tracing_enabled": False},
            ragas_available=True,
            memory_loaded=True,
            feedback_count=3,
            progressive_streaming=False,
            cache_bypass=True,
            has_api_key=True,
        )

        mock_st.subheader.assert_called_once_with("Agent Capabilities / Tool Registry")
        mock_st.dataframe.assert_called_once()
        caption_calls = [call.args[0] for call in mock_st.caption.call_args_list if call.args]
        assert caption_calls[0] == (
            "Maps the app's grounded retrieval, personalization, observability, and "
            "manual review tools."
        )
        rows = mock_st.dataframe.call_args.args[0]
        assert len(rows) == 10
        assert rows[0]["Capability"] == "Curated KB Retrieval"
        assert rows[-1]["Capability"] == "Cache Bypass"
        assert list(rows[0].keys()) == [
            "Capability",
            "Status",
            "Mode",
            "User Control",
            "Used By",
            "Description",
        ]
        assert mock_st.dataframe.call_args.kwargs["use_container_width"] is True
        assert mock_st.dataframe.call_args.kwargs["hide_index"] is True


# ---------------------------------------------------------------------------
# _display_ragas_report (integration-level with mocked st)
# ---------------------------------------------------------------------------

class TestDisplayRagasReport:
    """Verify _display_ragas_report calls Streamlit with correct data."""

    def _make_report(self):
        from src.eval.ragas_evaluation import RAGAsReport, RAGAsCaseResult
        return RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="LLM Basics and Prompt Engineering",
                    difficulty="beginner",
                    surface="learn_path",
                    faithfulness=1.0,
                    answer_relevancy=0.81,
                    context_precision=0.69,
                    context_recall=1.0,
                    answer_correctness=0.56,
                    num_contexts=8,
                    answer_length=5000,
                ),
                RAGAsCaseResult(
                    topic="RAG and Vector Databases",
                    difficulty="intermediate",
                    surface="learn_path",
                    faithfulness=0.94,
                    answer_relevancy=0.75,
                    context_precision=0.72,
                    context_recall=1.0,
                    answer_correctness=0.61,
                    num_contexts=10,
                    answer_length=6200,
                ),
                RAGAsCaseResult(
                    topic="AI Agents and Tool Calling",
                    difficulty="advanced",
                    surface="learn_path",
                    faithfulness=None,
                    answer_relevancy=0.70,
                    context_precision=0.54,
                    context_recall=1.0,
                    answer_correctness=0.17,
                    num_contexts=12,
                    answer_length=14000,
                    error=None,
                ),
            ],
            avg_faithfulness=1.0,
            avg_answer_relevancy=0.755,
            avg_context_precision=0.615,
            avg_context_recall=1.0,
            avg_answer_correctness=0.365,
            timestamp="2025-05-09T00:00:00Z",
            model="gpt-4o-mini",
            case_count=3,
        )

    def _setup_mock_st(self, mock_st):
        """Configure mock_st.columns to return correct count per call."""
        mock_st.columns.side_effect = lambda *a, **kw: [MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_metadata(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        # Should show timestamp/model/case_count via st.caption
        caption_calls = [str(c) for c in mock_st.caption.call_args_list]
        meta_shown = any("2025-05-09" in c and "gpt-4o-mini" in c for c in caption_calls)
        assert meta_shown, f"Metadata not shown in caption calls: {caption_calls}"

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_success_when_passing(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        mock_st.success.assert_called_once()
        assert "primary" in mock_st.success.call_args[0][0].lower()

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_warning_when_failing(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        report.avg_faithfulness = 0.3  # below threshold
        _display_ragas_report(report)

        mock_st.warning.assert_called()
        warning_text = mock_st.warning.call_args[0][0]
        assert "Faithfulness" in warning_text

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_benchmark_coverage_summary(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        assert any("Learn Path: 3 cached evaluated case(s)" in text for text in markdown_calls)
        caption_calls = [str(c) for c in mock_st.caption.call_args_list]
        assert all("Pending cases will be scored after running a fresh RAGAs evaluation." not in c for c in caption_calls)

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_borderline_variance_note(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        caption_calls = [str(c) for c in mock_st.caption.call_args_list]
        assert any("borderline yellow" in c.lower() for c in caption_calls)

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_pending_placeholders_for_unevaluated_cases(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        info_calls = [call.args[0] for call in mock_st.info.call_args_list if call.args]
        assert info_calls == []

    @patch("src.ui.dashboard_page.st")
    def test_display_restores_answer_correctness_only_inside_per_case_tables(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        table_calls = [text for text in markdown_calls if "| Metric | Score | Status | Role |" in text]
        assert any(
            "| Answer Correctness |" in text
            and "| Diagnostic |" in text
            for text in table_calls
        )
        assert all("#### Diagnostic Metric" not in text for text in markdown_calls)

    @patch("src.ui.dashboard_page.st")
    def test_display_shows_error_case(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report
        from src.eval.ragas_evaluation import RAGAsReport, RAGAsCaseResult

        self._setup_mock_st(mock_st)

        report = RAGAsReport(
            results=[
                RAGAsCaseResult(
                    topic="Broken",
                    difficulty="beginner",
                    error="LLM call failed",
                ),
            ],
            avg_faithfulness=None,
            avg_answer_relevancy=None,
            avg_context_precision=None,
            avg_context_recall=None,
            avg_answer_correctness=None,
        )
        _display_ragas_report(report)

        mock_st.error.assert_called_once()
        assert "Broken" in mock_st.error.call_args[0][0]


# ---------------------------------------------------------------------------
# _render_ragas_section (integration-level with mocked st)
# ---------------------------------------------------------------------------

class TestRenderRagasSection:
    """Verify render flow for the RAGAs section."""

    @patch("src.ui.dashboard_page.st")
    def test_shows_warning_when_ragas_unavailable(self, mock_st):
        from src.ui.dashboard_page import _render_ragas_section

        with patch(
            "src.ui.dashboard_page._check_ragas_available",
            return_value=(False, "ragas not installed"),
        ):
            _render_ragas_section()

        mock_st.warning.assert_any_call("ragas not installed")

    @patch("src.ui.dashboard_page.st")
    def test_shows_cost_warning_when_available(self, mock_st):
        from src.ui.dashboard_page import _render_ragas_section

        mock_st.button.return_value = False
        mock_st.session_state = {}

        with patch(
            "src.ui.dashboard_page._check_ragas_available",
            return_value=(True, ""),
        ), patch(
            "src.eval.ragas_evaluation.load_ragas_results",
            return_value=None,
        ):
            _render_ragas_section()

        # Should show cost warning (second st.warning call)
        warning_calls = [
            call for call in mock_st.warning.call_args_list
            if "cost" in str(call).lower() or "LLM judge" in str(call)
        ]
        assert len(warning_calls) >= 1
        warning_text = " ".join(str(call) for call in warning_calls)
        assert "5–10" in warning_text or "5-10" in warning_text
        assert "💰" not in warning_text

    @patch("src.ui.dashboard_page.st")
    def test_loads_cached_results_on_render(self, mock_st):
        from src.ui.dashboard_page import _render_ragas_section
        from src.eval.ragas_evaluation import RAGAsReport

        cached = RAGAsReport(
            timestamp="2025-05-09T00:00:00Z",
            model="gpt-4o-mini",
            case_count=3,
            avg_faithfulness=0.9,
        )
        mock_st.button.return_value = False
        mock_st.session_state = {}
        mock_st.columns.side_effect = lambda *a, **kw: [MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "src.ui.dashboard_page._check_ragas_available",
            return_value=(True, ""),
        ), patch(
            "src.eval.ragas_evaluation.load_ragas_results",
            return_value=cached,
        ):
            _render_ragas_section()

        # Cached report should be loaded into session state
        assert "ragas_report" in mock_st.session_state

    @patch("src.ui.dashboard_page.st")
    def test_shows_cached_info_message(self, mock_st):
        from src.ui.dashboard_page import _render_ragas_section

        mock_st.button.return_value = False
        mock_st.session_state = {}

        with patch(
            "src.ui.dashboard_page._check_ragas_available",
            return_value=(True, ""),
        ), patch(
            "src.eval.ragas_evaluation.load_ragas_results",
            return_value=None,
        ):
            _render_ragas_section()

        info_calls = [str(c) for c in mock_st.info.call_args_list]
        has_cached_msg = any("latest saved benchmark" in c for c in info_calls)
        assert has_cached_msg, f"Cached info message not found in: {info_calls}"
        assert all("💡" not in c for c in info_calls)


class TestDashboardUsageTables:
    """Session usage tables should clearly separate latest-run and all-session views."""

    @patch("src.ui.dashboard_page.st")
    def test_latest_run_context_caption_clarifies_scope(self, mock_st):
        from src.ui.dashboard_page import _display_latest_run_contexts

        mock_st.session_state = {
            "last_learn_mode": "Topic",
            "last_learn_depth": "Deep Study",
            "last_learn_progressive_streaming": True,
            "last_learn_force_regenerate": False,
            "quiz_selected_topic": "AI Agents",
        }
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]

        learn_result = {
            "trace": ["Cache miss"],
            "token_usage": {"total_tokens": 1200},
            "usage_records": [{"estimated_cost_usd": 0.0012}],
        }
        quiz_result = {
            "topic": "AI Agents",
            "trace": ["Cache hit"],
            "token_usage": {"total_tokens": 300},
            "usage_records": [{"estimated_cost_usd": 0.0003}],
        }
        _display_latest_run_contexts(learn_result, quiz_result)

        caption_calls = [str(c) for c in mock_st.caption.call_args_list]
        assert any("latest learn run and latest quiz run only" in c.lower() for c in caption_calls)

    @patch("src.ui.dashboard_page.st")
    def test_session_cost_summary_shows_context_columns(self, mock_st):
        from src.ui.dashboard_page import _display_session_cost_summary

        mock_st.session_state = {
            "session_usage_records": [
                {
                    "model": "gpt-4o-mini",
                    "operation": "learn_guide_overview_generation",
                    "total_tokens": 1200,
                    "prompt_tokens": 800,
                    "completion_tokens": 400,
                    "estimated_cost_usd": 0.0012,
                    "learning_mode": "Topic",
                    "learning_depth": "Deep Study",
                    "progressive_streaming": True,
                    "cache_bypass": False,
                    "cache_hit": False,
                },
                {
                    "model": "gpt-4o-mini",
                    "operation": "quiz_generation",
                    "total_tokens": 300,
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "estimated_cost_usd": 0.0003,
                    "learning_mode": None,
                    "learning_depth": None,
                    "progressive_streaming": None,
                    "cache_bypass": None,
                    "cache_hit": True,
                },
            ]
        }
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        _display_session_cost_summary()

        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        assert any("#### All Session Operations" in text for text in markdown_calls)

        mock_st.dataframe.assert_called_once()
        dataframe_rows = mock_st.dataframe.call_args.args[0]
        assert dataframe_rows[0]["Type"] == "Learn"
        assert dataframe_rows[0]["Mode"] == "Topic"
        assert dataframe_rows[0]["Depth"] == "Deep Study"
        assert dataframe_rows[0]["Stream"] == "On"
        assert dataframe_rows[0]["Bypass"] == "Off"
        assert dataframe_rows[0]["Cache"] == "Off"
        assert dataframe_rows[1]["Mode"] == "—"
        assert dataframe_rows[1]["Cache"] == "On"
        assert mock_st.dataframe.call_args.kwargs["use_container_width"] is True
        assert mock_st.dataframe.call_args.kwargs["hide_index"] is True

        caption_calls = [str(c) for c in mock_st.caption.call_args_list]
        assert any("all tracked llm operations" in c.lower() for c in caption_calls)
        assert any("session total:" in c.lower() for c in caption_calls)
