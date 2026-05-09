"""Tests for the dashboard Knowledge Base Health section."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _setup_mock_st(mock_st):
    mock_st.columns.side_effect = lambda *a, **kw: [
        MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
    ]
    mock_st.spinner.return_value.__enter__ = MagicMock()
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)


class TestDisplayKbHealthSection:
    @patch("src.ui.dashboard_page.st")
    @patch("src.kb.index_health.get_kb_index_health")
    @patch("src.config.get_settings")
    def test_renders_health_summary_and_warning(self, mock_settings, mock_health, mock_st):
        from src.ui.dashboard_page import _display_kb_health_section

        _setup_mock_st(mock_st)
        mock_settings.return_value = SimpleNamespace(openai_api_key="test-key")
        mock_health.return_value = {
            "status": "up_to_date",
            "status_label": "Up to date",
            "reindex_required": False,
            "raw_docs_count": 10,
            "official_docs_count": 12,
            "embedding_model": "text-embedding-3-small",
            "last_rebuild_at": "2026-05-09T10:00:00+00:00",
            "collections": {
                "curated": {"chunk_count": 120, "source_count": 10},
                "official": {"chunk_count": 85, "source_count": 12},
            },
            "notes": [],
        }
        mock_st.button.return_value = False
        mock_st.session_state = {}

        _display_kb_health_section()

        mock_st.subheader.assert_any_call("Knowledge Base Health")
        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        assert any("| Index Status | Up to date |" in text for text in markdown_calls)
        assert any("| Curated Chunks / Sources | 120 / 10 |" in text for text in markdown_calls)
        warning_calls = [str(call) for call in mock_st.warning.call_args_list]
        assert any("may take time" in call.lower() for call in warning_calls)

    @patch("src.ui.dashboard_page.st")
    @patch("src.kb.index_health.rebuild_kb_index")
    @patch("src.kb.index_health.get_kb_index_health")
    @patch("src.config.get_settings")
    def test_rebuild_runs_only_on_button_click(
        self,
        mock_settings,
        mock_health,
        mock_rebuild,
        mock_st,
    ):
        from src.ui.dashboard_page import _display_kb_health_section

        _setup_mock_st(mock_st)
        mock_settings.return_value = SimpleNamespace(openai_api_key="test-key")
        mock_health.return_value = {
            "status": "outdated",
            "status_label": "Outdated",
            "reindex_required": True,
            "raw_docs_count": 10,
            "official_docs_count": 12,
            "embedding_model": "text-embedding-3-small",
            "last_rebuild_at": "2026-05-09T10:00:00+00:00",
            "collections": {
                "curated": {"chunk_count": 120, "source_count": 10},
                "official": {"chunk_count": 85, "source_count": 12},
            },
            "notes": ["Markdown source files changed after the last KB rebuild."],
        }
        mock_st.button.return_value = True
        mock_st.session_state = {}

        _display_kb_health_section()

        mock_rebuild.assert_called_once()
        assert mock_st.session_state["kb_rebuild_notice"] == "KB index rebuilt successfully."
        mock_st.rerun.assert_called_once()
