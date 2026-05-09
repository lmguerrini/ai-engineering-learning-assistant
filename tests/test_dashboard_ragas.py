"""Tests for the RAGAs evaluation section in the Dashboard."""

from unittest.mock import patch, MagicMock

import pytest

from src.ui.dashboard_page import (
    _check_ragas_available,
    _metric_color,
    _fmt_metric,
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
        assert _ragas_snapshot_value(object()) == "Ready"

    def test_ragas_snapshot_value_not_run_without_report(self):
        assert _ragas_snapshot_value(None) == "Not run"


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
                    topic="LLM Basics",
                    difficulty="beginner",
                    faithfulness=1.0,
                    answer_relevancy=0.81,
                    context_precision=0.69,
                    context_recall=1.0,
                    answer_correctness=0.56,
                    num_contexts=8,
                    answer_length=5000,
                ),
                RAGAsCaseResult(
                    topic="AI Agents",
                    difficulty="intermediate",
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
            case_count=2,
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
    def test_display_shows_diagnostic_note(self, mock_st):
        from src.ui.dashboard_page import _display_ragas_report

        self._setup_mock_st(mock_st)
        report = self._make_report()
        _display_ragas_report(report)

        # Should show diagnostic note about Answer Correctness
        caption_calls = [str(c) for c in mock_st.caption.call_args_list]
        has_diagnostic = any("diagnostic" in c.lower() for c in caption_calls)
        assert has_diagnostic, f"Diagnostic note not found in caption calls: {caption_calls}"

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
