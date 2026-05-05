"""Tests for the ingestion pipeline.

Uses mocked embeddings to avoid requiring an OpenAI API key.
"""

from pathlib import Path
from unittest.mock import patch

from src.kb.ingestion import run_ingestion

FAKE_EMBEDDING = [0.1] * 128


def _fake_get_embeddings(texts, model=None):
    """Return fake embeddings for testing."""
    return [FAKE_EMBEDDING] * len(texts)


class TestIngestion:
    @patch("src.kb.vector_store.get_embeddings", side_effect=_fake_get_embeddings)
    def test_full_pipeline(self, mock_embed, tmp_path: Path):
        """Test the full ingestion pipeline with mock embeddings."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "topic_a.md").write_text("# Topic A\n\nContent about topic A." * 5)
        (docs_dir / "topic_b.txt").write_text("Topic B content.\n" * 10)

        chroma_dir = str(tmp_path / "chroma")

        added = run_ingestion(
            documents_dir=docs_dir,
            persist_dir=chroma_dir,
            chunk_size=100,
            chunk_overlap=20,
        )

        assert added > 0
        assert mock_embed.called

    @patch("src.kb.vector_store.get_embeddings", side_effect=_fake_get_embeddings)
    def test_empty_directory(self, mock_embed, tmp_path: Path):
        """Test ingestion with no documents."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        added = run_ingestion(
            documents_dir=empty_dir,
            persist_dir=str(tmp_path / "chroma"),
        )

        assert added == 0
        assert not mock_embed.called

    @patch("src.kb.vector_store.get_embeddings", side_effect=_fake_get_embeddings)
    def test_missing_directory(self, mock_embed, tmp_path: Path):
        """Test ingestion with nonexistent directory."""
        added = run_ingestion(
            documents_dir=tmp_path / "nonexistent",
            persist_dir=str(tmp_path / "chroma"),
        )

        assert added == 0
