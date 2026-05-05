"""Tests for observability utilities and display helpers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.services.observability import (
    TracingStatus,
    configure_langsmith_tracing,
    format_tracing_status,
    get_tracing_status,
)
from src.ui.display_helpers import (
    format_error_message,
    format_graph_state_summary,
    format_memory_transparency,
    format_source_display,
    format_sources_summary,
)


# ---------------------------------------------------------------------------
# Observability config / tracing status
# ---------------------------------------------------------------------------


class TestTracingStatus:
    def test_default_status(self):
        status = TracingStatus()
        assert status.enabled is False
        assert status.project == ""
        assert status.has_api_key is False
        assert status.issues == []

    def test_format_tracing_status_disabled(self):
        status = TracingStatus(enabled=False, project="test-proj", issues=["Not enabled."])
        result = format_tracing_status(status)
        assert result["tracing_enabled"] is False
        assert result["status_label"] == "❌ Disabled"
        assert result["project"] == "test-proj"
        assert "Not enabled." in result["issues"]

    def test_format_tracing_status_enabled(self):
        status = TracingStatus(enabled=True, project="prod", has_api_key=True)
        result = format_tracing_status(status)
        assert result["tracing_enabled"] is True
        assert result["status_label"] == "✅ Active"


class TestConfigureLangsmithTracing:
    @patch("src.services.observability.get_settings")
    def test_tracing_disabled(self, mock_settings):
        settings = MagicMock()
        settings.langchain_tracing_v2 = False
        settings.langchain_api_key = ""
        settings.langchain_project = "test"
        mock_settings.return_value = settings

        status = configure_langsmith_tracing()
        assert status.enabled is False
        assert any("not enabled" in i.lower() for i in status.issues)

    @patch("src.services.observability.get_settings")
    def test_tracing_enabled_no_key(self, mock_settings):
        settings = MagicMock()
        settings.langchain_tracing_v2 = True
        settings.langchain_api_key = ""
        settings.langchain_project = "test"
        mock_settings.return_value = settings

        status = configure_langsmith_tracing()
        assert status.enabled is False
        assert any("API_KEY" in i for i in status.issues)

    @patch("src.services.observability.get_settings")
    def test_tracing_enabled_with_key(self, mock_settings):
        settings = MagicMock()
        settings.langchain_tracing_v2 = True
        settings.langchain_api_key = "ls-key-123"
        settings.langchain_project = "my-proj"
        mock_settings.return_value = settings

        # Clean env to avoid interference
        env_backup = {k: os.environ.pop(k, None) for k in [
            "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT",
        ]}
        try:
            status = configure_langsmith_tracing()
            assert status.enabled is True
            assert status.has_api_key is True
            assert status.project == "my-proj"
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    @patch("src.services.observability.get_settings", side_effect=RuntimeError("boom"))
    def test_settings_unavailable(self, mock_settings):
        status = configure_langsmith_tracing()
        assert status.enabled is False
        assert any("unavailable" in i.lower() for i in status.issues)


class TestGetTracingStatus:
    @patch("src.services.observability.get_settings")
    def test_returns_disabled_when_not_configured(self, mock_settings):
        settings = MagicMock()
        settings.langchain_tracing_v2 = False
        settings.langchain_api_key = ""
        settings.langchain_project = "test"
        mock_settings.return_value = settings

        status = get_tracing_status()
        assert status.enabled is False

    @patch("src.services.observability.get_settings", side_effect=Exception("fail"))
    def test_graceful_on_settings_error(self, mock_settings):
        status = get_tracing_status()
        assert status.enabled is False
        assert len(status.issues) > 0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


class TestFormatSourceDisplay:
    def test_with_full_source(self):
        src = MagicMock()
        src.title = "AI Agents Overview"
        src.content_snippet = "Agents are autonomous..."
        src.relevance_score = 0.85
        src.metadata = {"topic": "AI Agents", "filename": "agents.md"}

        result = format_source_display(src)
        assert result["title"] == "AI Agents Overview"
        assert result["snippet"] == "Agents are autonomous..."
        assert result["relevance"] == 0.85
        assert result["relevance_label"] == "0.8"
        assert ("Topic", "AI Agents") in result["metadata_items"]
        assert ("File", "agents.md") in result["metadata_items"]

    def test_with_empty_source(self):
        src = MagicMock()
        src.title = ""
        src.content_snippet = ""
        src.relevance_score = 0.0
        src.metadata = {}

        result = format_source_display(src)
        assert result["title"] == "Untitled"
        assert result["snippet"] == "_No preview available._"
        assert result["relevance_label"] == "N/A"
        assert result["metadata_items"] == []

    def test_with_no_attributes(self):
        result = format_source_display(object())
        assert result["title"] == "Untitled"
        assert result["relevance_label"] == "N/A"


class TestFormatSourcesSummary:
    def test_no_sources(self):
        assert format_sources_summary([]) == "No sources used."
        assert format_sources_summary(None) == "No sources used."

    def test_with_sources(self):
        assert format_sources_summary(["a", "b", "c"]) == "3 source(s) used."


class TestFormatGraphStateSummary:
    def test_full_result(self):
        result = {
            "topic": "LangGraph",
            "difficulty": MagicMock(value="intermediate"),
            "style": MagicMock(value="detailed"),
            "retrieved_docs": [1, 2, 3],
            "attempts": 2,
            "query_refined": True,
            "memory_profile": {"recent_topics": ["X"]},
            "token_usage": {"total_tokens": 1500},
        }
        fields = format_graph_state_summary(result)
        labels = [f["label"] for f in fields]
        assert "Topic" in labels
        assert "Sources Retrieved" in labels
        assert "Retrieval Attempts" in labels
        assert "Query Refined" in labels
        assert "Memory Profile" in labels
        assert "Total Tokens" in labels

    def test_empty_result(self):
        fields = format_graph_state_summary({})
        labels = [f["label"] for f in fields]
        assert "Sources Retrieved" in labels
        # Memory profile should show "Not available"
        mem_field = next(f for f in fields if f["label"] == "Memory Profile")
        assert mem_field["value"] == "Not available"

    def test_no_token_usage(self):
        fields = format_graph_state_summary({"token_usage": {}})
        labels = [f["label"] for f in fields]
        assert "Total Tokens" not in labels


class TestFormatMemoryTransparency:
    def test_no_profile(self):
        result = format_memory_transparency(None)
        assert result["loaded"] is False
        assert "message" in result

    def test_empty_dict_profile(self):
        """An empty dict is falsy — treated as no profile."""
        result = format_memory_transparency({})
        assert result["loaded"] is False

    def test_minimal_profile(self):
        result = format_memory_transparency({"recent_topics": []})
        assert result["loaded"] is True
        assert result["recent_topics"] == []
        assert result["weak_areas"] == []

    def test_full_profile(self):
        profile = {
            "recent_topics": ["A", "B"],
            "recurring_weak_areas": ["X"],
            "average_score": 75.0,
            "suggested_focus_topics": ["C"],
            "preferred_style": "detailed",
        }
        result = format_memory_transparency(profile)
        assert result["loaded"] is True
        assert result["recent_topics"] == ["A", "B"]
        assert result["weak_areas"] == ["X"]
        assert result["average_score"] == 75.0
        assert result["suggested_focus"] == ["C"]


class TestFormatErrorMessage:
    def test_known_error_types(self):
        for error_type in [
            "no_api_key", "retrieval_failure", "no_sources",
            "quiz_generation_failure", "incomplete_answers",
            "memory_save_failure", "empty_progress",
        ]:
            result = format_error_message(error_type)
            assert "icon" in result
            assert "title" in result
            assert "message" in result
            assert "suggestion" in result

    def test_unknown_error_type(self):
        result = format_error_message("unknown_type_xyz")
        assert result["title"] == "Something Went Wrong"
        assert "suggestion" in result
