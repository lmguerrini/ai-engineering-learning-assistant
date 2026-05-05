"""Tests for official docs KB: loading, retrieval, fallback, isolation, and refresh."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.kb.loader import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def official_docs_dir(tmp_path):
    """Create a temporary directory with sample official docs."""
    doc1 = tmp_path / "openai_api_structured_outputs.md"
    doc1.write_text(
        "# OpenAI API\n\n- **source_type**: official_docs\n\n"
        "Use chat completions for structured outputs."
    )
    doc2 = tmp_path / "langgraph_state_orchestration.md"
    doc2.write_text(
        "# LangGraph State\n\n- **source_type**: official_docs\n\n"
        "StateGraph for multi-step agent workflows."
    )
    return tmp_path


@pytest.fixture
def curated_docs_dir(tmp_path):
    """Create a temporary directory with sample curated KB docs."""
    d = tmp_path / "raw"
    d.mkdir()
    (d / "rag_basics.md").write_text("# RAG Basics\n\nRAG overview content here.")
    return d


@pytest.fixture
def empty_dir(tmp_path):
    """Create an empty temporary directory."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Data isolation tests
# ---------------------------------------------------------------------------

class TestDataIsolation:
    """Verify official docs are not loaded from data/raw/ and vice versa."""

    def test_raw_dir_does_not_contain_official_docs(self):
        """data/raw/ should not contain files from data/official_docs/."""
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            pytest.skip("data/raw/ not present")
        raw_files = {f.name for f in raw_dir.iterdir() if f.suffix == ".md"}
        official_dir = Path("data/official_docs")
        if not official_dir.exists():
            pytest.skip("data/official_docs/ not present")
        official_files = {f.name for f in official_dir.iterdir() if f.suffix == ".md"}
        overlap = raw_files & official_files
        assert not overlap, f"Files present in both raw and official_docs: {overlap}"

    def test_official_docs_dir_exists_separately(self):
        """data/official_docs/ should exist as a separate directory."""
        assert Path("data/official_docs").is_dir()

    def test_loader_does_not_load_official_docs_from_raw(self):
        """Loading from data/raw/ should not include official docs files."""
        from src.kb.loader import load_documents

        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            pytest.skip("data/raw/ not present")
        docs = load_documents(raw_dir)
        official_filenames = {
            "openai_api_structured_outputs.md",
            "langchain_core_tools.md",
            "langgraph_state_orchestration.md",
            "langsmith_observability.md",
            "chroma_vector_store.md",
            "ragas_evaluation.md",
            "streamlit_app_patterns.md",
            "pydantic_validation_settings.md",
            "loguru_logging.md",
        }
        loaded_filenames = {d.metadata.get("filename") for d in docs}
        overlap = loaded_filenames & official_filenames
        assert not overlap, f"Official docs loaded from raw: {overlap}"


# ---------------------------------------------------------------------------
# Official docs loading tests
# ---------------------------------------------------------------------------

class TestLoadOfficialDocs:
    """Test loading official docs with source_type tagging."""

    def test_load_official_docs_tags_source_type(self, official_docs_dir):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(official_docs_dir)
        assert len(docs) == 2
        for doc in docs:
            assert doc.metadata["source_type"] == "official_docs"

    def test_load_official_docs_tags_doc_type(self, official_docs_dir):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(official_docs_dir)
        for doc in docs:
            assert doc.metadata["doc_type"] == "official_reference"

    def test_load_official_docs_tags_domain(self, official_docs_dir):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(official_docs_dir)
        domains = {d.metadata["filename"]: d.metadata["domain"] for d in docs}
        assert domains["openai_api_structured_outputs.md"] == "openai"
        assert domains["langgraph_state_orchestration.md"] == "langgraph"

    def test_load_official_docs_preserves_filename(self, official_docs_dir):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(official_docs_dir)
        filenames = {d.metadata["filename"] for d in docs}
        assert "openai_api_structured_outputs.md" in filenames
        assert "langgraph_state_orchestration.md" in filenames

    def test_load_official_docs_empty_dir(self, empty_dir):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(empty_dir)
        assert docs == []

    def test_load_official_docs_nonexistent_dir(self, tmp_path):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(tmp_path / "nonexistent")
        assert docs == []


# ---------------------------------------------------------------------------
# Official docs retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieveOfficialDocs:
    """Test retrieval from the official docs collection."""

    def test_retrieve_empty_query_returns_empty(self):
        from src.kb.official_docs import retrieve_official_docs

        result = retrieve_official_docs("   ")
        assert result == []

    @patch("src.kb.official_docs.similarity_search")
    @patch("src.kb.official_docs.create_collection")
    @patch("src.kb.official_docs.get_chroma_client")
    def test_retrieve_tags_source_type(self, mock_client, mock_coll, mock_search):
        from src.kb.official_docs import retrieve_official_docs

        mock_search.return_value = [
            Document(content="LangGraph info", metadata={"filename": "langgraph.md"})
        ]
        results = retrieve_official_docs("langgraph", top_k=3)
        assert len(results) == 1
        assert results[0].metadata["source_type"] == "official_docs"

    @patch("src.kb.official_docs.similarity_search", side_effect=Exception("DB error"))
    @patch("src.kb.official_docs.create_collection")
    @patch("src.kb.official_docs.get_chroma_client")
    def test_retrieve_error_returns_empty(self, mock_client, mock_coll, mock_search):
        from src.kb.official_docs import retrieve_official_docs

        results = retrieve_official_docs("test query")
        assert results == []


# ---------------------------------------------------------------------------
# Fallback retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieveWithFallback:
    """Test the fallback/enrichment logic."""

    def test_curated_sufficient_skips_fallback(self):
        from src.kb.official_docs import retrieve_with_fallback

        curated = [
            Document(content="A" * 150, metadata={"filename": "a.md"}),
            Document(content="B" * 150, metadata={"filename": "b.md"}),
        ]
        result = retrieve_with_fallback(
            query="test", curated_docs=curated,
            min_sources=2, min_content_chars=200,
        )
        assert len(result) == 2
        for doc in result:
            assert doc.metadata["source_type"] == "curated_kb"

    @patch("src.kb.official_docs.retrieve_official_docs")
    def test_curated_insufficient_triggers_fallback(self, mock_official):
        from src.kb.official_docs import retrieve_with_fallback

        mock_official.return_value = [
            Document(content="Official info", metadata={"filename": "official.md", "source_type": "official_docs"})
        ]
        curated = [
            Document(content="Short", metadata={"filename": "a.md"}),
        ]
        result = retrieve_with_fallback(
            query="test", curated_docs=curated,
            min_sources=2, min_content_chars=200,
        )
        assert len(result) == 2
        assert result[0].metadata["source_type"] == "curated_kb"
        assert result[1].metadata["source_type"] == "official_docs"

    @patch("src.kb.official_docs.retrieve_official_docs")
    def test_empty_curated_triggers_fallback(self, mock_official):
        from src.kb.official_docs import retrieve_with_fallback

        mock_official.return_value = [
            Document(content="Fallback content", metadata={"filename": "fb.md", "source_type": "official_docs"})
        ]
        result = retrieve_with_fallback(
            query="test", curated_docs=[],
            min_sources=2, min_content_chars=200,
        )
        assert len(result) == 1
        assert result[0].metadata["source_type"] == "official_docs"

    @patch("src.kb.official_docs.retrieve_official_docs", return_value=[])
    def test_fallback_returns_curated_when_official_empty(self, mock_official):
        from src.kb.official_docs import retrieve_with_fallback

        curated = [Document(content="Short", metadata={"filename": "a.md"})]
        result = retrieve_with_fallback(
            query="test", curated_docs=curated,
            min_sources=2, min_content_chars=200,
        )
        assert len(result) == 1
        assert result[0].metadata["source_type"] == "curated_kb"

    def test_curated_docs_get_source_type_tag(self):
        from src.kb.official_docs import retrieve_with_fallback

        curated = [
            Document(content="A" * 200, metadata={"filename": "a.md"}),
            Document(content="B" * 200, metadata={"filename": "b.md"}),
        ]
        result = retrieve_with_fallback(
            query="test", curated_docs=curated,
            min_sources=2, min_content_chars=200,
        )
        for doc in result:
            assert "source_type" in doc.metadata


# ---------------------------------------------------------------------------
# Metadata source_type tests
# ---------------------------------------------------------------------------

class TestSourceTypeMetadata:
    """Verify source_type is set correctly in all paths."""

    def test_official_docs_have_source_type(self, official_docs_dir):
        from src.kb.official_docs import load_official_docs

        docs = load_official_docs(official_docs_dir)
        for doc in docs:
            assert doc.metadata["source_type"] == "official_docs"

    def test_curated_docs_tagged_by_fallback(self):
        from src.kb.official_docs import retrieve_with_fallback

        curated = [
            Document(content="A" * 300, metadata={"filename": "a.md"}),
            Document(content="B" * 300, metadata={"filename": "b.md"}),
        ]
        retrieve_with_fallback(
            query="test", curated_docs=curated,
            min_sources=2, min_content_chars=200,
        )
        for doc in curated:
            assert doc.metadata["source_type"] == "curated_kb"


# ---------------------------------------------------------------------------
# Domain inference tests
# ---------------------------------------------------------------------------

class TestDomainInference:
    """Test domain inference from filenames and query detection."""

    def test_infer_domain_known_files(self):
        from src.kb.official_docs import infer_domain

        assert infer_domain("openai_api_structured_outputs.md") == "openai"
        assert infer_domain("langchain_core_tools.md") == "langchain"
        assert infer_domain("langgraph_state_orchestration.md") == "langgraph"
        assert infer_domain("langsmith_observability.md") == "langsmith"
        assert infer_domain("chroma_vector_store.md") == "chroma"
        assert infer_domain("ragas_evaluation.md") == "ragas"
        assert infer_domain("streamlit_app_patterns.md") == "streamlit"
        assert infer_domain("pydantic_validation_settings.md") == "pydantic"
        assert infer_domain("loguru_logging.md") == "loguru"

    def test_infer_domain_unknown_file(self):
        from src.kb.official_docs import infer_domain

        assert infer_domain("random_notes.md") == "general"
        assert infer_domain("") == "general"

    def test_detect_query_domains_langgraph(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("How does LangGraph StateGraph work?")
        assert "langgraph" in domains

    def test_detect_query_domains_openai(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("OpenAI structured output with GPT")
        assert "openai" in domains

    def test_detect_query_domains_chroma(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("How to use Chroma vector store?")
        assert "chroma" in domains

    def test_detect_query_domains_ragas(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("RAGAs evaluation metrics")
        assert "ragas" in domains

    def test_detect_query_domains_streamlit(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("Streamlit sidebar layout")
        assert "streamlit" in domains

    def test_detect_query_domains_pydantic(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("Pydantic BaseModel validation")
        assert "pydantic" in domains

    def test_detect_query_domains_loguru(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("Loguru logging setup")
        assert "loguru" in domains

    def test_detect_query_domains_langsmith(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("LangSmith tracing observability")
        assert "langsmith" in domains

    def test_detect_query_domains_langchain(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("LangChain tools and chains")
        assert "langchain" in domains

    def test_detect_query_domains_no_match(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("generic question about nothing specific")
        assert domains == []

    def test_detect_query_domains_multiple(self):
        from src.kb.official_docs import detect_query_domains

        domains = detect_query_domains("LangGraph with OpenAI and Chroma")
        assert "langgraph" in domains
        assert "openai" in domains
        assert "chroma" in domains


# ---------------------------------------------------------------------------
# Domain-aware reranking tests
# ---------------------------------------------------------------------------

class TestDomainReranking:
    """Test domain-aware reranking of retrieval results."""

    def test_rerank_prioritizes_matching_domain(self):
        from src.kb.official_docs import _rerank_by_domain

        docs = [
            Document(content="A", metadata={"domain": "openai"}),
            Document(content="B", metadata={"domain": "langgraph"}),
            Document(content="C", metadata={"domain": "chroma"}),
        ]
        result = _rerank_by_domain(docs, ["langgraph"], top_k=3)
        assert result[0].metadata["domain"] == "langgraph"

    def test_rerank_trims_to_top_k(self):
        from src.kb.official_docs import _rerank_by_domain

        docs = [
            Document(content="A", metadata={"domain": "openai"}),
            Document(content="B", metadata={"domain": "langgraph"}),
            Document(content="C", metadata={"domain": "chroma"}),
        ]
        result = _rerank_by_domain(docs, ["langgraph"], top_k=2)
        assert len(result) == 2
        assert result[0].metadata["domain"] == "langgraph"

    def test_rerank_multiple_preferred_domains(self):
        from src.kb.official_docs import _rerank_by_domain

        docs = [
            Document(content="A", metadata={"domain": "streamlit"}),
            Document(content="B", metadata={"domain": "langgraph"}),
            Document(content="C", metadata={"domain": "openai"}),
            Document(content="D", metadata={"domain": "chroma"}),
        ]
        result = _rerank_by_domain(docs, ["langgraph", "openai"], top_k=4)
        matched_domains = [d.metadata["domain"] for d in result[:2]]
        assert "langgraph" in matched_domains
        assert "openai" in matched_domains

    def test_rerank_no_matches_preserves_order(self):
        from src.kb.official_docs import _rerank_by_domain

        docs = [
            Document(content="A", metadata={"domain": "openai"}),
            Document(content="B", metadata={"domain": "chroma"}),
        ]
        result = _rerank_by_domain(docs, ["ragas"], top_k=2)
        assert len(result) == 2
        assert result[0].metadata["domain"] == "openai"

    def test_rerank_empty_docs(self):
        from src.kb.official_docs import _rerank_by_domain

        result = _rerank_by_domain([], ["langgraph"], top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# Retrieval metadata enrichment tests
# ---------------------------------------------------------------------------

class TestRetrievalMetadata:
    """Test that retrieval results carry full metadata."""

    @patch("src.kb.official_docs.similarity_search")
    @patch("src.kb.official_docs.create_collection")
    @patch("src.kb.official_docs.get_chroma_client")
    def test_retrieve_enriches_doc_type(self, mock_client, mock_coll, mock_search):
        from src.kb.official_docs import retrieve_official_docs

        mock_search.return_value = [
            Document(content="info", metadata={"filename": "langgraph_state_orchestration.md"})
        ]
        results = retrieve_official_docs("langgraph state", top_k=3)
        assert results[0].metadata["doc_type"] == "official_reference"

    @patch("src.kb.official_docs.similarity_search")
    @patch("src.kb.official_docs.create_collection")
    @patch("src.kb.official_docs.get_chroma_client")
    def test_retrieve_enriches_domain(self, mock_client, mock_coll, mock_search):
        from src.kb.official_docs import retrieve_official_docs

        mock_search.return_value = [
            Document(content="info", metadata={"filename": "langgraph_state_orchestration.md"})
        ]
        results = retrieve_official_docs("langgraph state", top_k=3)
        assert results[0].metadata["domain"] == "langgraph"

    @patch("src.kb.official_docs.similarity_search")
    @patch("src.kb.official_docs.create_collection")
    @patch("src.kb.official_docs.get_chroma_client")
    def test_retrieve_domain_filtering_for_langgraph_query(self, mock_client, mock_coll, mock_search):
        """A LangGraph query should prioritize langgraph docs over others."""
        from src.kb.official_docs import retrieve_official_docs

        mock_search.return_value = [
            Document(content="openai stuff", metadata={"filename": "openai_api.md"}),
            Document(content="langgraph stuff", metadata={"filename": "langgraph_state_orchestration.md"}),
            Document(content="chroma stuff", metadata={"filename": "chroma_vector_store.md"}),
        ]
        results = retrieve_official_docs("How does LangGraph StateGraph work?", top_k=2)
        assert results[0].metadata["domain"] == "langgraph"

    @patch("src.kb.official_docs.similarity_search")
    @patch("src.kb.official_docs.create_collection")
    @patch("src.kb.official_docs.get_chroma_client")
    def test_curated_kb_source_type_preserved_in_combined(self, mock_client, mock_coll, mock_search):
        """When fallback combines curated + official, source_types are distinct."""
        from src.kb.official_docs import retrieve_with_fallback

        mock_search.return_value = [
            Document(content="official", metadata={"filename": "langgraph.md", "source_type": "official_docs"})
        ]
        curated = [Document(content="short", metadata={"filename": "a.md"})]
        result = retrieve_with_fallback(
            query="test", curated_docs=curated,
            min_sources=2, min_content_chars=200,
        )
        source_types = [d.metadata["source_type"] for d in result]
        assert "curated_kb" in source_types
        assert "official_docs" in source_types


# ---------------------------------------------------------------------------
# Refresh pipeline tests
# ---------------------------------------------------------------------------

class TestRefreshPipeline:
    """Test the refresh script's registry and graceful failure."""

    def test_source_registry_exists(self):
        from scripts.refresh_official_docs import get_source_registry

        registry = get_source_registry()
        assert len(registry) == 9

    def test_source_registry_has_required_fields(self):
        from scripts.refresh_official_docs import get_source_registry

        for src in get_source_registry():
            assert "name" in src
            assert "filename" in src
            assert "url" in src
            assert "topics" in src
            assert src["url"].startswith("http")
            assert src["filename"].endswith(".md")

    def test_source_registry_filenames_are_unique(self):
        from scripts.refresh_official_docs import get_source_registry

        filenames = [s["filename"] for s in get_source_registry()]
        assert len(filenames) == len(set(filenames))

    def test_check_existing_docs(self, official_docs_dir):
        from scripts.refresh_official_docs import check_existing_docs, OFFICIAL_SOURCES

        status = check_existing_docs(official_docs_dir)
        assert "openai_api_structured_outputs.md" in status
        assert status["openai_api_structured_outputs.md"] is True
        # Most others should not exist in our fixture
        assert status["chroma_vector_store.md"] is False

    def test_fetch_url_graceful_failure(self):
        """fetch_url_content should return None on network failure."""
        from scripts.refresh_official_docs import fetch_url_content

        result = fetch_url_content("http://localhost:1/nonexistent")
        assert result is None

    def test_refresh_single_dry_run(self, tmp_path):
        from scripts.refresh_official_docs import refresh_single_source

        source = {
            "name": "Test Source",
            "filename": "test_source.md",
            "url": "https://example.com/",
            "topics": "testing",
        }
        result = refresh_single_source(source, tmp_path, dry_run=True)
        assert result is True
        assert not (tmp_path / "test_source.md").exists()

    @patch("scripts.refresh_official_docs.fetch_url_content", return_value=None)
    def test_refresh_preserves_existing_on_failure(self, mock_fetch, tmp_path):
        from scripts.refresh_official_docs import refresh_single_source

        existing = tmp_path / "test.md"
        existing.write_text("existing content")
        source = {
            "name": "Test", "filename": "test.md",
            "url": "https://example.com/", "topics": "test",
        }
        result = refresh_single_source(source, tmp_path, dry_run=False)
        assert result is False
        assert existing.read_text() == "existing content"

    @patch("scripts.refresh_official_docs.fetch_url_content", return_value="<html>Hello</html>")
    def test_refresh_writes_file_on_success(self, mock_fetch, tmp_path):
        from scripts.refresh_official_docs import refresh_single_source

        source = {
            "name": "Test Source", "filename": "test_new.md",
            "url": "https://example.com/", "topics": "testing",
        }
        result = refresh_single_source(source, tmp_path, dry_run=False)
        assert result is True
        content = (tmp_path / "test_new.md").read_text()
        assert "Test Source" in content
        assert "official_docs" in content
        assert "https://example.com/" in content


# ---------------------------------------------------------------------------
# Learn graph integration (no regression)
# ---------------------------------------------------------------------------

class TestLearnGraphIntegration:
    """Verify retrieve_sources node still works with fallback integration."""

    @patch("src.graphs.learn_nodes.retrieve_with_fallback")
    @patch("src.graphs.learn_nodes.retrieve_documents")
    def test_retrieve_sources_calls_fallback(self, mock_retrieve, mock_fallback):
        from src.graphs.learn_nodes import retrieve_sources

        curated = [Document(content="curated content", metadata={"filename": "a.md"})]
        mock_retrieve.return_value = curated
        mock_fallback.return_value = curated

        state = {"topic": "RAG", "query": "RAG", "trace": [], "attempts": 0}
        result = retrieve_sources(state)

        mock_retrieve.assert_called_once()
        mock_fallback.assert_called_once()
        assert result["attempts"] == 1
        assert len(result["retrieved_docs"]) == 1

    @patch("src.graphs.learn_nodes.retrieve_with_fallback")
    @patch("src.graphs.learn_nodes.retrieve_documents")
    def test_retrieve_sources_traces_official_docs_fallback(self, mock_retrieve, mock_fallback):
        from src.graphs.learn_nodes import retrieve_sources

        curated = [Document(content="short", metadata={"filename": "a.md"})]
        official = [Document(content="official", metadata={"filename": "o.md", "source_type": "official_docs"})]
        mock_retrieve.return_value = curated
        mock_fallback.return_value = curated + official

        state = {"topic": "test", "query": "test", "trace": [], "attempts": 0}
        result = retrieve_sources(state)

        assert any("official doc chunks as fallback" in t for t in result["trace"])
        assert len(result["retrieved_docs"]) == 2
