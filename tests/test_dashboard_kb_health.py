"""Tests for the dashboard Knowledge Base Health section."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


def _setup_mock_st(mock_st):
    mock_st.columns.side_effect = lambda *a, **kw: [
        MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
    ]
    mock_st.spinner.return_value.__enter__ = MagicMock()
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)


class TestDisplayKbHealthSection:
    def test_dashboard_timestamp_helper_formats_iso_timestamp(self):
        from src.ui.dashboard_page import _format_dashboard_timestamp

        assert (
            _format_dashboard_timestamp("2026-05-11T13:57:26.448797+00:00")
            == "2026-05-11 13:57 UTC"
        )

    @patch("src.ui.dashboard_page.st")
    @patch("src.kb.index_health.get_kb_index_health")
    @patch("src.config.get_settings")
    def test_renders_health_summary_and_manual_rebuild_warning(self, mock_settings, mock_health, mock_st):
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
        mock_st.success.assert_any_call("✅ KB index is up to date.")
        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        assert any("| Index Status | Up to date |" in text for text in markdown_calls)
        assert any("| Curated Chunks / Sources | 120 / 10 |" in text for text in markdown_calls)
        assert any("| Last Rebuild | 2026-05-09 10:00 UTC |" in text for text in markdown_calls)
        warning_calls = [str(call) for call in mock_st.warning.call_args_list]
        assert any("may take time" in call.lower() for call in warning_calls)

    @patch("src.ui.dashboard_page.st")
    @patch("src.kb.index_health.get_kb_index_health")
    @patch("src.config.get_settings")
    def test_rebuild_notice_renders_inside_section_before_status(self, mock_settings, mock_health, mock_st):
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
            "last_rebuild_at": "2026-05-11T13:43:28.471971+00:00",
            "collections": {
                "curated": {"chunk_count": 120, "source_count": 10},
                "official": {"chunk_count": 85, "source_count": 12},
            },
            "notes": [],
        }
        mock_st.button.return_value = False
        mock_st.session_state = {"kb_rebuild_notice": "KB index rebuilt successfully."}

        _display_kb_health_section()

        assert "kb_rebuild_notice" not in mock_st.session_state
        calls = mock_st.method_calls
        assert calls.index(call.subheader("Knowledge Base Health")) < calls.index(
            call.caption(
                "Tracks whether the local Chroma index matches the current curated and official markdown files."
            )
        )
        assert calls.index(
            call.caption(
                "Tracks whether the local Chroma index matches the current curated and official markdown files."
            )
        ) < calls.index(call.success("KB index rebuilt successfully."))
        assert calls.index(call.success("KB index rebuilt successfully.")) < calls.index(
            call.success("✅ KB index is up to date.")
        )

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
            "status_label": "Rebuild recommended",
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
        mock_st.warning.assert_any_call("⚠️ KB index is outdated. Rebuild recommended.")
        assert mock_st.session_state["kb_rebuild_notice"] == "KB index rebuilt successfully."
        mock_st.rerun.assert_called_once()

    @patch("src.ui.dashboard_page.st")
    @patch("src.kb.index_health.get_kb_index_health")
    @patch("src.config.get_settings")
    def test_metadata_missing_shows_explanatory_copy(self, mock_settings, mock_health, mock_st):
        from src.ui.dashboard_page import _display_kb_health_section

        _setup_mock_st(mock_st)
        mock_settings.return_value = SimpleNamespace(openai_api_key="test-key")
        mock_health.return_value = {
            "status": "metadata_missing",
            "status_label": "Rebuild recommended",
            "reindex_required": True,
            "raw_docs_count": 10,
            "official_docs_count": 12,
            "embedding_model": "text-embedding-3-small",
            "last_rebuild_at": None,
            "collections": {
                "curated": {"chunk_count": 120, "source_count": 10},
                "official": {"chunk_count": 85, "source_count": 12},
            },
            "notes": [
                "Chroma collections already exist, but no KB health metadata baseline was found yet. Rebuild once to enable freshness tracking."
            ],
        }
        mock_st.button.return_value = False
        mock_st.session_state = {}

        _display_kb_health_section()

        mock_st.info.assert_called()
        info_calls = [str(call) for call in mock_st.info.call_args_list]
        assert any("no kb health metadata baseline" in call.lower() for call in info_calls)


class TestDisplayExternalDocsUpdaterSection:
    @patch("src.ui.dashboard_page.st")
    @patch("src.services.external_docs_updater.get_external_docs_source_status")
    def test_renders_updater_section_and_rebuild_copy(self, mock_status, mock_st):
        from src.ui.dashboard_page import _display_external_docs_updater_section

        _setup_mock_st(mock_st)
        mock_status.return_value = [
            {
                "Source": "OpenAI API & Structured Outputs",
                "File": "openai_api_structured_outputs.md",
                "Primary Domain": "platform.openai.com",
                "Local File": "Present",
                "Last Refreshed": "2025-05-08",
                "URLs": "1",
            }
        ]
        mock_st.button.return_value = False
        mock_st.session_state = {}

        _display_external_docs_updater_section()

        mock_st.subheader.assert_called_once_with("External Docs / API Updater")
        mock_st.info.assert_any_call(
            "Run Rebuild KB Index after a successful docs update to make refreshed docs available to retrieval."
        )
        mock_st.dataframe.assert_called_once()
        rows = mock_st.dataframe.call_args.args[0]
        assert rows[0]["Source"] == "OpenAI API & Structured Outputs"
        assert rows[0]["Primary Domain"] == "platform.openai.com"

    @patch("src.ui.dashboard_page.st")
    @patch("src.services.external_docs_updater.update_external_docs")
    @patch("src.services.external_docs_updater.get_external_docs_source_status")
    def test_manual_update_runs_only_on_button_click(self, mock_status, mock_update, mock_st):
        from src.ui.dashboard_page import _display_external_docs_updater_section

        _setup_mock_st(mock_st)
        mock_status.return_value = []
        mock_update.return_value = {
            "checked_sources": 2,
            "updated_files": 1,
            "partial_files": 0,
            "skipped_files": 1,
            "error_count": 0,
            "results": [
                {
                    "Source": "OpenAI API & Structured Outputs",
                    "File": "openai_api_structured_outputs.md",
                    "Status": "Updated",
                    "URLs Checked": 1,
                    "URLs Fetched": 1,
                    "Result": "Fetched 1/1 URL(s) and refreshed the local Markdown file.",
                }
            ],
        }
        mock_st.button.return_value = True
        mock_st.session_state = {}

        _display_external_docs_updater_section()

        mock_update.assert_called_once()
        mock_st.button.assert_called_once_with(
            "Update External Official Docs", key="btn_update_external_docs"
        )
        assert mock_st.session_state["external_docs_update_result"]["updated_files"] == 1
        mock_st.rerun.assert_called_once()

    @patch("src.ui.dashboard_page.st")
    @patch("src.services.external_docs_updater.get_external_docs_source_status")
    def test_displays_stored_update_results(self, mock_status, mock_st):
        from src.ui.dashboard_page import _display_external_docs_updater_section

        _setup_mock_st(mock_st)
        mock_status.return_value = []
        metric_cols = [MagicMock() for _ in range(5)]
        mock_st.columns.side_effect = lambda *a, **kw: metric_cols
        mock_st.button.return_value = False
        mock_st.session_state = {
            "external_docs_update_result": {
                "checked_sources": 2,
                "updated_files": 1,
                "partial_files": 0,
                "skipped_files": 1,
                "error_count": 0,
                "results": [
                    {
                        "Source": "OpenAI API & Structured Outputs",
                        "File": "openai_api_structured_outputs.md",
                        "Status": "Updated",
                        "URLs Checked": 1,
                        "URLs Fetched": 1,
                        "Result": "Fetched 1/1 URL(s) and refreshed the local Markdown file.",
                    },
                    {
                        "Source": "LangChain Core & Tools",
                        "File": "langchain_core_tools.md",
                        "Status": "Skipped",
                        "URLs Checked": 1,
                        "URLs Fetched": 0,
                        "Result": "Fetch failed; kept existing local file unchanged. Failed: docs.langchain.com (timeout).",
                    }
                ],
            }
        }

        _display_external_docs_updater_section()

        mock_st.success.assert_called_once_with(
            "External docs updated. Run Rebuild KB Index to make refreshed docs available to retrieval."
        )
        mock_st.columns.assert_called_once_with(5)
        metric_cols[0].metric.assert_called_once_with("Checked Sources", "2")
        metric_cols[1].metric.assert_called_once_with("Updated Files", "1")
        metric_cols[2].metric.assert_called_once_with("Partial Files", "0")
        metric_cols[3].metric.assert_called_once_with("Skipped Files", "1")
        metric_cols[4].metric.assert_called_once_with("Failed Files", "0")
        assert mock_st.dataframe.call_count == 2

    @patch("src.ui.dashboard_page.st")
    @patch("src.services.external_docs_updater.get_external_docs_source_status")
    def test_displays_partial_update_results_safely(self, mock_status, mock_st):
        from src.ui.dashboard_page import _display_external_docs_updater_section

        _setup_mock_st(mock_st)
        mock_status.return_value = []
        metric_cols = [MagicMock() for _ in range(5)]
        mock_st.columns.side_effect = lambda *a, **kw: metric_cols
        mock_st.button.return_value = False
        mock_st.session_state = {
            "external_docs_update_result": {
                "checked_sources": 1,
                "updated_files": 0,
                "partial_files": 1,
                "skipped_files": 0,
                "error_count": 0,
                "results": [
                    {
                        "Source": "OpenAI API & Structured Outputs",
                        "File": "openai_api_structured_outputs.md",
                        "Status": "Partial",
                        "URLs Checked": 2,
                        "URLs Fetched": 1,
                        "Result": "Fetched 1/2 URL(s) and refreshed the local Markdown file. Partial success. Failed URLs: platform.openai.com (HTTP 403 Forbidden).",
                    }
                ],
            }
        }

        _display_external_docs_updater_section()

        mock_st.warning.assert_called_once_with(
            "External docs partially updated. Review failed URLs, then run Rebuild KB Index to make refreshed docs available to retrieval."
        )
        mock_st.columns.assert_called_once_with(5)
        metric_cols[0].metric.assert_called_once_with("Checked Sources", "1")
        metric_cols[1].metric.assert_called_once_with("Updated Files", "0")
        metric_cols[2].metric.assert_called_once_with("Partial Files", "1")
        metric_cols[3].metric.assert_called_once_with("Skipped Files", "0")
        metric_cols[4].metric.assert_called_once_with("Failed Files", "0")
        assert mock_st.dataframe.call_count == 2
