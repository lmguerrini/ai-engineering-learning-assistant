"""Tests for document retrieval.

Uses mocked embeddings to avoid requiring an OpenAI API key.
"""

from pathlib import Path
from unittest.mock import patch

from src.kb.ingestion import run_ingestion
from src.kb.retrieval import retrieve_documents

FAKE_EMBEDDING = [0.1] * 128


def _fake_get_embeddings(texts, model=None):
    """Return fake embeddings for testing."""
    return [FAKE_EMBEDDING] * len(texts)


class TestRetrieval:
    @patch("src.kb.vector_store.get_embeddings", side_effect=_fake_get_embeddings)
    def test_retrieve_returns_results(self, mock_embed, tmp_path: Path):
        """Test retrieval returns documents after ingestion."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "agents.md").write_text(
            "# AI Agents\n\nAgents use LLMs for reasoning and tool calling." * 5
        )
        chroma_dir = str(tmp_path / "chroma")

        run_ingestion(
            documents_dir=docs_dir,
            persist_dir=chroma_dir,
            chunk_size=100,
            chunk_overlap=20,
        )

        results = retrieve_documents(
            query="What are AI agents?",
            top_k=3,
            persist_dir=chroma_dir,
        )

        assert len(results) > 0
        assert results[0].content
        assert "filename" in results[0].metadata

    @patch("src.kb.vector_store.get_embeddings", side_effect=_fake_get_embeddings)
    def test_retrieve_respects_top_k(self, mock_embed, tmp_path: Path):
        """Test that top_k limits the number of results."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "big.md").write_text("Content. " * 500)
        chroma_dir = str(tmp_path / "chroma")

        run_ingestion(
            documents_dir=docs_dir,
            persist_dir=chroma_dir,
            chunk_size=50,
            chunk_overlap=10,
        )

        results = retrieve_documents(
            query="content",
            top_k=2,
            persist_dir=chroma_dir,
        )

        assert len(results) <= 2

    def test_retrieve_empty_collection(self, tmp_path: Path):
        """Test retrieval from empty collection returns empty list."""
        chroma_dir = str(tmp_path / "empty_chroma")

        results = retrieve_documents(
            query="anything",
            persist_dir=chroma_dir,
        )

        assert results == []

    def test_retrieve_empty_query(self, tmp_path: Path):
        """Test empty query returns empty list."""
        results = retrieve_documents(
            query="",
            persist_dir=str(tmp_path / "chroma"),
        )

        assert results == []

    def test_retrieve_whitespace_query(self, tmp_path: Path):
        """Test whitespace-only query returns empty list."""
        results = retrieve_documents(
            query="   ",
            persist_dir=str(tmp_path / "chroma"),
        )

        assert results == []
