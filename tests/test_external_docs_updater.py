"""Tests for the manual external docs updater service."""

from pathlib import Path


class TestExternalDocsUpdater:
    def test_preview_text_strips_html_head_scripts_styles_and_attributes(self):
        from src.services.external_docs_updater import _preview_text

        html = """
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <title>Hidden Title</title>
            <link rel="preload" href="/bundle.js">
            <style>.hero { color: red; }</style>
            <script>window.__DATA__ = {"x": 1};</script>
          </head>
          <body data-theme="dark">
            <main id="app">
              <h1 class="hero">Visible Heading</h1>
              <p data-id="123">Readable paragraph with <a href="/docs" class="nav">docs link</a>.</p>
            </main>
          </body>
        </html>
        """

        preview = _preview_text(html)

        assert "Visible Heading" in preview
        assert "Readable paragraph with docs link." in preview
        assert "<!DOCTYPE html>" not in preview
        assert "<head>" not in preview
        assert "<script>" not in preview
        assert "<style>" not in preview
        assert "<meta" not in preview
        assert "<link" not in preview
        assert "window.__DATA__" not in preview
        assert "data-theme" not in preview
        assert "class=" not in preview
        assert "href=" not in preview

    def test_fetch_external_doc_url_returns_reason_for_failed_fetch_result_dict(self):
        from src.services.external_docs_updater import _normalize_fetch_result

        result = _normalize_fetch_result(
            {
                "ok": False,
                "content": None,
                "error": "HTTP 403 Forbidden",
            },
            source_url="https://example.com/a",
        )

        assert result["ok"] is False
        assert result["error"] == "HTTP 403 Forbidden"
        assert result["source_url"] == "https://example.com/a"

    def test_update_external_docs_preserves_existing_body_and_adds_preview(self, tmp_path):
        from src.services.external_docs_updater import (
            AUTO_PREVIEW_START,
            update_external_docs,
        )

        filepath = tmp_path / "test_source.md"
        filepath.write_text(
            "# Test Source\n\n"
            "- **Official source**: https://old.example/\n"
            "- **Last refreshed**: 2025-01-01\n"
            "- **source_type**: official_docs\n\n"
            "## Key Concepts\n\n"
            "Original curated notes.\n",
            encoding="utf-8",
        )
        source = {
            "name": "Test Source",
            "filename": "test_source.md",
            "urls": ["https://example.com/a", "https://example.com/b"],
        }
        fetch_map = {
            "https://example.com/a": "<html>Alpha docs preview</html>",
            "https://example.com/b": "<html>Beta docs preview</html>",
        }

        result = update_external_docs(
            directory=tmp_path,
            sources=[source],
            fetcher=lambda url: fetch_map[url],
        )

        assert result["checked_sources"] == 1
        assert result["updated_files"] == 1
        assert result["partial_files"] == 0
        assert result["skipped_files"] == 0
        assert result["error_count"] == 0
        row = result["results"][0]
        assert row["Status"] == "Updated"
        assert row["URLs Checked"] == 2
        assert row["URLs Fetched"] == 2

        content = filepath.read_text(encoding="utf-8")
        assert "Original curated notes." in content
        assert AUTO_PREVIEW_START in content
        assert "https://example.com/a" in content
        assert "https://example.com/b" in content
        assert "<html>" not in content
        assert "<body>" not in content
        assert "<script>" not in content
        assert "Run Rebuild KB Index in the Dashboard" in content
        assert "- **Last refreshed**:" in content

    def test_update_external_docs_skips_existing_file_when_fetch_fails(self, tmp_path):
        from src.services.external_docs_updater import update_external_docs

        filepath = tmp_path / "test_source.md"
        original = (
            "# Test Source\n\n"
            "- **Official source**: https://example.com/a\n"
            "- **Last refreshed**: 2025-01-01\n"
            "- **source_type**: official_docs\n\n"
            "Original content.\n"
        )
        filepath.write_text(original, encoding="utf-8")
        source = {
            "name": "Test Source",
            "filename": "test_source.md",
            "urls": ["https://example.com/a"],
        }

        result = update_external_docs(
            directory=tmp_path,
            sources=[source],
            fetcher=lambda _url: None,
        )

        assert result["updated_files"] == 0
        assert result["partial_files"] == 0
        assert result["skipped_files"] == 1
        assert result["error_count"] == 0
        assert filepath.read_text(encoding="utf-8") == original
        assert result["results"][0]["Status"] == "Skipped"
        assert "Failed:" in result["results"][0]["Result"]
        assert "example.com" in result["results"][0]["Result"]

    def test_update_external_docs_reports_error_when_file_missing_and_fetch_fails(self, tmp_path):
        from src.services.external_docs_updater import update_external_docs

        source = {
            "name": "Missing Source",
            "filename": "missing.md",
            "urls": ["https://example.com/missing"],
        }

        result = update_external_docs(
            directory=tmp_path,
            sources=[source],
            fetcher=lambda _url: None,
        )

        assert result["updated_files"] == 0
        assert result["partial_files"] == 0
        assert result["skipped_files"] == 0
        assert result["error_count"] == 1
        assert result["results"][0]["Status"] == "Error"
        assert "Failed:" in result["results"][0]["Result"]
        assert not (tmp_path / "missing.md").exists()

    def test_update_external_docs_includes_failure_reason_in_result_output(self, tmp_path):
        from src.services.external_docs_updater import update_external_docs

        filepath = tmp_path / "test_source.md"
        filepath.write_text("# Existing\n", encoding="utf-8")
        source = {
            "name": "Test Source",
            "filename": "test_source.md",
            "urls": ["https://example.com/a"],
        }

        result = update_external_docs(
            directory=tmp_path,
            sources=[source],
            fetcher=lambda _url: {
                "ok": False,
                "content": None,
                "error": "SSL error: certificate verify failed",
            },
        )

        assert result["partial_files"] == 0
        assert result["results"][0]["Status"] == "Skipped"
        assert "SSL error" in result["results"][0]["Result"]

    def test_update_external_docs_partial_success_updates_file_and_reports_failure_reason(self, tmp_path):
        from src.services.external_docs_updater import update_external_docs

        filepath = tmp_path / "test_source.md"
        filepath.write_text(
            "# Test Source\n\n"
            "- **Official source**: https://example.com/a\n"
            "- **Last refreshed**: 2025-01-01\n"
            "- **source_type**: official_docs\n\n"
            "Original content.\n",
            encoding="utf-8",
        )
        source = {
            "name": "Test Source",
            "filename": "test_source.md",
            "urls": ["https://example.com/a", "https://example.com/b"],
        }

        def _fetch(url):
            if url.endswith("/a"):
                return {
                    "ok": True,
                    "content": "<html>Updated docs</html>",
                    "final_url": url,
                    "status_code": 200,
                    "content_type": "text/html",
                }
            return {
                "ok": False,
                "content": None,
                "error": "HTTP 403 Forbidden",
                "source_url": url,
            }

        result = update_external_docs(
            directory=tmp_path,
            sources=[source],
            fetcher=_fetch,
        )

        assert result["updated_files"] == 0
        assert result["partial_files"] == 1
        assert result["skipped_files"] == 0
        assert result["error_count"] == 0
        assert result["results"][0]["Status"] == "Partial"
        assert "Partial success." in result["results"][0]["Result"]
        assert "HTTP 403 Forbidden" in result["results"][0]["Result"]
        content = filepath.read_text(encoding="utf-8")
        assert "Updated docs" in content
        assert "<html>" not in content

    def test_source_status_reads_existing_refresh_dates(self, tmp_path):
        from src.services.external_docs_updater import get_external_docs_source_status

        filepath = tmp_path / "test_source.md"
        filepath.write_text(
            "# Test Source\n\n"
            "- **Official source**: https://example.com/a\n"
            "- **Last refreshed**: 2025-05-08\n"
            "- **source_type**: official_docs\n",
            encoding="utf-8",
        )
        rows = get_external_docs_source_status(
            directory=tmp_path,
            sources=[
                {
                    "name": "Test Source",
                    "filename": "test_source.md",
                    "urls": ["https://example.com/a", "https://example.com/b"],
                }
            ],
        )

        assert rows == [
            {
                "Source": "Test Source",
                "File": "test_source.md",
                "Primary Domain": "example.com",
                "Local File": "Present",
                "Last Refreshed": "2025-05-08",
                "URLs": "2",
            }
        ]

    def test_latest_external_docs_refresh_date_returns_most_recent_value(self, tmp_path):
        from src.services.external_docs_updater import get_latest_external_docs_refresh_date

        (tmp_path / "older.md").write_text(
            "# Older\n\n"
            "- **Official source**: https://example.com/older\n"
            "- **Last refreshed**: 2025-05-08\n"
            "- **source_type**: official_docs\n",
            encoding="utf-8",
        )
        (tmp_path / "newer.md").write_text(
            "# Newer\n\n"
            "- **Official source**: https://example.com/newer\n"
            "- **Last refreshed**: 2025-05-10\n"
            "- **source_type**: official_docs\n",
            encoding="utf-8",
        )

        latest = get_latest_external_docs_refresh_date(
            directory=tmp_path,
            sources=[
                {
                    "name": "Older",
                    "filename": "older.md",
                    "urls": ["https://example.com/older"],
                },
                {
                    "name": "Newer",
                    "filename": "newer.md",
                    "urls": ["https://example.com/newer"],
                },
            ],
        )

        assert latest == "2025-05-10"
