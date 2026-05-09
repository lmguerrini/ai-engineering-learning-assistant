"""Tests for KB index freshness and manual rebuild helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.kb.index_health import (
    build_source_snapshot,
    get_kb_index_health,
    load_index_metadata,
    rebuild_kb_index,
    save_index_metadata,
)
from src.kb.vector_store import DEFAULT_COLLECTION


def _make_settings(tmp_path: Path, *, embedding_model: str = "text-embedding-3-small"):
    raw_dir = tmp_path / "raw"
    official_dir = tmp_path / "official"
    chroma_dir = tmp_path / "chroma"
    raw_dir.mkdir()
    official_dir.mkdir()
    chroma_dir.mkdir()
    return SimpleNamespace(
        raw_documents_dir=str(raw_dir),
        official_docs_dir=str(official_dir),
        chroma_persist_dir=str(chroma_dir),
        embedding_model=embedding_model,
        official_docs_collection="official_docs",
        chunk_size=500,
        chunk_overlap=50,
        openai_api_key="test-key",
    )


def _write_docs(settings) -> None:
    Path(settings.raw_documents_dir, "topic_a.md").write_text("# Topic A\nContent")
    Path(settings.official_docs_dir, "official_a.md").write_text("# Official\nReference")


class TestBuildSourceSnapshot:
    def test_snapshot_uses_path_size_and_mtime(self, tmp_path: Path):
        settings = _make_settings(tmp_path)
        _write_docs(settings)

        snapshot = build_source_snapshot(
            raw_documents_dir=settings.raw_documents_dir,
            official_docs_dir=settings.official_docs_dir,
        )

        assert len(snapshot["raw"]) == 1
        assert len(snapshot["official"]) == 1
        raw_entry = snapshot["raw"][0]
        assert raw_entry["path"].endswith("topic_a.md")
        assert raw_entry["size"] > 0
        assert raw_entry["modified_ns"] > 0


class TestGetKbIndexHealth:
    @patch("src.kb.index_health._get_collection_stats")
    @patch("src.kb.index_health.get_settings")
    def test_missing_when_collections_absent(self, mock_settings, mock_stats, tmp_path: Path):
        settings = _make_settings(tmp_path)
        _write_docs(settings)
        mock_settings.return_value = settings
        mock_stats.side_effect = [
            {"name": DEFAULT_COLLECTION, "available": False, "chunk_count": None, "source_count": None},
            {"name": settings.official_docs_collection, "available": False, "chunk_count": None, "source_count": None},
        ]

        health = get_kb_index_health()

        assert health["status"] == "missing"
        assert health["status_label"] == "Missing"
        assert health["reindex_required"] is True
        assert health["raw_docs_count"] == 1
        assert health["official_docs_count"] == 1

    @patch("src.kb.index_health._get_collection_stats")
    @patch("src.kb.index_health.get_settings")
    def test_metadata_missing_when_collections_exist_without_baseline(
        self,
        mock_settings,
        mock_stats,
        tmp_path: Path,
    ):
        settings = _make_settings(tmp_path)
        _write_docs(settings)
        mock_settings.return_value = settings
        mock_stats.side_effect = [
            {"name": DEFAULT_COLLECTION, "available": True, "chunk_count": 3, "source_count": 1},
            {"name": settings.official_docs_collection, "available": True, "chunk_count": 4, "source_count": 1},
        ]

        health = get_kb_index_health()

        assert health["status"] == "metadata_missing"
        assert health["status_label"] == "Rebuild recommended"
        assert health["reindex_required"] is True
        assert any("no kb health metadata baseline" in note.lower() for note in health["notes"])

    @patch("src.kb.index_health._get_collection_stats")
    @patch("src.kb.index_health.get_settings")
    def test_up_to_date_when_snapshot_and_counts_match(self, mock_settings, mock_stats, tmp_path: Path):
        settings = _make_settings(tmp_path)
        _write_docs(settings)
        mock_settings.return_value = settings
        snapshot = build_source_snapshot(
            raw_documents_dir=settings.raw_documents_dir,
            official_docs_dir=settings.official_docs_dir,
        )
        save_index_metadata(
            {
                "rebuilt_at": "2026-05-09T10:00:00+00:00",
                "embedding_model": settings.embedding_model,
                "snapshot": snapshot,
                "collections": {
                    "curated": {"name": DEFAULT_COLLECTION, "chunk_count": 3, "source_count": 1},
                    "official": {"name": settings.official_docs_collection, "chunk_count": 4, "source_count": 1},
                },
            },
            persist_dir=settings.chroma_persist_dir,
        )
        mock_stats.side_effect = [
            {"name": DEFAULT_COLLECTION, "available": True, "chunk_count": 3, "source_count": 1},
            {"name": settings.official_docs_collection, "available": True, "chunk_count": 4, "source_count": 1},
        ]

        health = get_kb_index_health()

        assert health["status"] == "up_to_date"
        assert health["status_label"] == "Up to date"
        assert health["reindex_required"] is False
        assert health["last_rebuild_at"] == "2026-05-09T10:00:00+00:00"
        assert health["embedding_model"] == settings.embedding_model

    @patch("src.kb.index_health._get_collection_stats")
    @patch("src.kb.index_health.get_settings")
    def test_outdated_when_source_snapshot_changes(self, mock_settings, mock_stats, tmp_path: Path):
        settings = _make_settings(tmp_path)
        _write_docs(settings)
        mock_settings.return_value = settings
        original_snapshot = build_source_snapshot(
            raw_documents_dir=settings.raw_documents_dir,
            official_docs_dir=settings.official_docs_dir,
        )
        save_index_metadata(
            {
                "rebuilt_at": "2026-05-09T10:00:00+00:00",
                "embedding_model": settings.embedding_model,
                "snapshot": original_snapshot,
                "collections": {
                    "curated": {"name": DEFAULT_COLLECTION, "chunk_count": 3, "source_count": 1},
                    "official": {"name": settings.official_docs_collection, "chunk_count": 4, "source_count": 1},
                },
            },
            persist_dir=settings.chroma_persist_dir,
        )
        Path(settings.raw_documents_dir, "topic_a.md").write_text("# Topic A\nUpdated content")
        mock_stats.side_effect = [
            {"name": DEFAULT_COLLECTION, "available": True, "chunk_count": 3, "source_count": 1},
            {"name": settings.official_docs_collection, "available": True, "chunk_count": 4, "source_count": 1},
        ]

        health = get_kb_index_health()

        assert health["status"] == "outdated"
        assert health["status_label"] == "Rebuild recommended"
        assert health["reindex_required"] is True
        assert any("source files changed" in note.lower() for note in health["notes"])


class TestRebuildKbIndex:
    @patch("src.kb.index_health._get_collection_stats")
    @patch("src.kb.index_health.ingest_official_docs")
    @patch("src.kb.index_health.run_ingestion")
    @patch("src.kb.index_health.delete_collection")
    @patch("src.kb.index_health.get_chroma_client")
    @patch("src.kb.index_health.get_settings")
    def test_rebuild_resets_collections_and_saves_metadata(
        self,
        mock_settings,
        mock_client,
        mock_delete,
        mock_ingest,
        mock_official,
        mock_stats,
        tmp_path: Path,
    ):
        settings = _make_settings(tmp_path)
        _write_docs(settings)
        mock_settings.return_value = settings
        mock_client.return_value = MagicMock()
        mock_ingest.return_value = 6
        mock_official.return_value = 8
        mock_stats.side_effect = [
            {"name": DEFAULT_COLLECTION, "available": True, "chunk_count": 6, "source_count": 1},
            {"name": settings.official_docs_collection, "available": True, "chunk_count": 8, "source_count": 1},
        ]

        health = rebuild_kb_index()

        mock_delete.assert_any_call(DEFAULT_COLLECTION, client=mock_client.return_value)
        mock_delete.assert_any_call(settings.official_docs_collection, client=mock_client.return_value)
        metadata = load_index_metadata(settings.chroma_persist_dir)
        assert metadata is not None
        assert metadata["embedding_model"] == settings.embedding_model
        assert metadata["collections"]["curated"]["chunk_count"] == 6
        assert metadata["collections"]["official"]["chunk_count"] == 8
        assert health["status"] == "up_to_date"
        assert health["collections"]["curated"]["chunk_count"] == 6
        assert health["collections"]["official"]["chunk_count"] == 8
