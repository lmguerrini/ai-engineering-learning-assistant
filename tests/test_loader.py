"""Tests for the document loader."""

from pathlib import Path

import pytest

from src.kb.loader import Document, load_document, load_documents, _infer_topic


class TestInferTopic:
    def test_underscores(self):
        assert _infer_topic("ai_agents.md") == "Ai Agents"

    def test_hyphens(self):
        assert _infer_topic("lang-graph-basics.txt") == "Lang Graph Basics"

    def test_simple(self):
        assert _infer_topic("overview.md") == "Overview"


class TestLoadDocument:
    def test_load_md(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("# Hello\nWorld", encoding="utf-8")
        doc = load_document(f)
        assert isinstance(doc, Document)
        assert "Hello" in doc.content
        assert doc.metadata["filename"] == "test.md"
        assert doc.metadata["topic"] == "Test"

    def test_load_txt(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("Some notes", encoding="utf-8")
        doc = load_document(f)
        assert doc.content == "Some notes"

    def test_unsupported_extension(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(f)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "missing.md")


class TestLoadDocuments:
    def test_load_from_directory(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("Doc A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Doc B", encoding="utf-8")
        (tmp_path / "c.csv").write_text("ignored", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 2

    def test_empty_directory(self, tmp_path: Path):
        docs = load_documents(tmp_path)
        assert docs == []

    def test_missing_directory(self, tmp_path: Path):
        docs = load_documents(tmp_path / "nonexistent")
        assert docs == []

    def test_load_real_data(self):
        """Verify the example documents in data/raw/ can be loaded."""
        docs = load_documents(Path("data/raw"))
        assert len(docs) >= 2
        filenames = {d.metadata["filename"] for d in docs}
        assert "ai_agents.md" in filenames
        assert "langgraph_basics.md" in filenames
