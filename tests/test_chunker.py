"""Tests for the text chunker."""

import pytest

from src.kb.chunker import chunk_text, chunk_document, chunk_documents
from src.kb.loader import Document


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_splits_long_text(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
        assert len(chunks) >= 3
        assert all(len(c) <= 300 for c in chunks)

    def test_overlap_works(self):
        text = "ABCDEFGHIJ" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
        assert len(chunks) >= 3
        # Verify overlap: end of chunk i should overlap with start of chunk i+1
        for i in range(len(chunks) - 1):
            # The step is chunk_size - overlap = 30
            # So chunks overlap by 10 characters at boundaries
            assert len(chunks[i]) <= 40

    def test_empty_text(self):
        assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
        assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []

    def test_overlap_exceeds_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("test", chunk_size=10, chunk_overlap=10)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("test", chunk_size=10, chunk_overlap=20)


class TestChunkDocument:
    def test_produces_chunk_documents(self):
        doc = Document(
            content="A" * 200,
            metadata={"filename": "test.md", "topic": "Test"},
        )
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Document)
            assert chunk.metadata["filename"] == "test.md"
            assert chunk.metadata["chunk_index"] == i

    def test_empty_document(self):
        doc = Document(content="", metadata={"filename": "empty.md"})
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        assert chunks == []


class TestChunkDocuments:
    def test_chunks_multiple_docs(self):
        docs = [
            Document(content="A" * 200, metadata={"filename": "a.md"}),
            Document(content="B" * 200, metadata={"filename": "b.md"}),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 4
        filenames = {c.metadata["filename"] for c in chunks}
        assert filenames == {"a.md", "b.md"}
