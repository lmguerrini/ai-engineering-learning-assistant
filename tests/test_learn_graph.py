"""Tests for the Learn LangGraph workflow."""

from unittest.mock import MagicMock, patch

import pytest

from src.graphs.learn_graph import build_learn_graph, compile_learn_graph, run_learn_workflow
from src.graphs.learn_nodes import (
    _MIN_CONTENT_CHARS,
    _MIN_SOURCES,
    _build_fallback_guide,
    _build_memory_context,
    _build_sources_list,
    _extract_summary_from_markdown,
    _generate_deep_study_learn_path,
    assess_source_quality,
    generate_study_guide,
    load_user_memory,
    persist_learning_event_placeholder,
    quality_check,
    refine_query_if_needed,
    retrieve_sources,
    return_output,
    validate_input,
)
from src.graphs.learn_prompts import is_deep_study_learn_path
from src.graphs.learn_state import LearningState
from src.kb.loader import Document
from src.schemas import DifficultyLevel, ResponseStyle, StudyGuide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(n: int, content_len: int = 150) -> list[Document]:
    """Create n dummy Document objects."""
    return [
        Document(
            content="x" * content_len,
            metadata={"filename": f"doc_{i}.md", "topic": f"Topic {i}"},
        )
        for i in range(n)
    ]


def _base_state(**overrides) -> LearningState:
    """Return a minimal valid LearningState with optional overrides."""
    state: LearningState = {
        "topic": "AI Agents",
        "difficulty": DifficultyLevel.INTERMEDIATE,
        "style": ResponseStyle.DETAILED,
        "trace": [],
        "token_usage": {},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_valid_input(self):
        result = validate_input(_base_state())
        assert result["error"] is None
        assert result["topic"] == "AI Agents"
        assert result["query"] == "AI Agents"

    def test_empty_topic(self):
        result = validate_input(_base_state(topic=""))
        assert result["error"] is not None
        assert "topic" in result["error"].lower()

    def test_whitespace_topic(self):
        result = validate_input(_base_state(topic="   "))
        assert result["error"] is not None

    def test_defaults_applied(self):
        state: LearningState = {"topic": "RAG", "trace": []}
        result = validate_input(state)
        assert result["difficulty"] == DifficultyLevel.INTERMEDIATE
        assert result["style"] == ResponseStyle.DETAILED


class TestLoadUserMemory:
    @patch("src.memory.memory_service.get_user_profile_summary")
    def test_loads_profile(self, mock_profile):
        mock_profile.return_value = {
            "recent_topics": ["RAG"],
            "recurring_weak_areas": ["embeddings"],
            "average_score": 65.0,
            "preferred_style": None,
            "suggested_focus_topics": ["embeddings"],
        }
        result = load_user_memory(_base_state())
        assert result["memory_profile"]["recent_topics"] == ["RAG"]
        assert any("profile loaded" in t for t in result["trace"])

    @patch("src.memory.memory_service.get_user_profile_summary", side_effect=Exception("DB error"))
    def test_fallback_on_error(self, mock_profile):
        result = load_user_memory(_base_state())
        assert result["memory_profile"]["recent_topics"] == []
        assert any("no memory data" in t for t in result["trace"])


class TestRetrieveSources:
    @patch("src.graphs.learn_nodes.retrieve_documents")
    def test_retrieves_docs_deep_study(self, mock_retrieve):
        docs = _make_docs(3)
        mock_retrieve.return_value = docs
        result = retrieve_sources(_base_state(query="AI Agents", style=ResponseStyle.DETAILED))
        assert result["retrieved_docs"] == docs
        mock_retrieve.assert_called_once_with(query="AI Agents", top_k=10)

    @patch("src.graphs.learn_nodes.retrieve_documents")
    def test_retrieves_docs_summary(self, mock_retrieve):
        docs = _make_docs(3)
        mock_retrieve.return_value = docs
        result = retrieve_sources(_base_state(query="AI Agents", style=ResponseStyle.CONCISE))
        assert result["retrieved_docs"] == docs
        mock_retrieve.assert_called_once_with(query="AI Agents", top_k=6)

    @patch("src.graphs.learn_nodes.retrieve_documents")
    def test_empty_retrieval(self, mock_retrieve):
        mock_retrieve.return_value = []
        result = retrieve_sources(_base_state(query="unknown"))
        assert result["retrieved_docs"] == []


class TestAssessSourceQuality:
    def test_sufficient_sources(self):
        docs = _make_docs(_MIN_SOURCES, content_len=_MIN_CONTENT_CHARS)
        result = assess_source_quality(_base_state(retrieved_docs=docs))
        assert result["source_quality_ok"] is True

    def test_insufficient_count(self):
        docs = _make_docs(1, content_len=500)
        result = assess_source_quality(_base_state(retrieved_docs=docs))
        assert result["source_quality_ok"] is False

    def test_insufficient_content(self):
        docs = _make_docs(3, content_len=10)
        result = assess_source_quality(_base_state(retrieved_docs=docs))
        assert result["source_quality_ok"] is False

    def test_empty_docs(self):
        result = assess_source_quality(_base_state(retrieved_docs=[]))
        assert result["source_quality_ok"] is False


class TestRefineQueryIfNeeded:
    def test_refines_query(self):
        result = refine_query_if_needed(_base_state(query="AI Agents"))
        assert "overview" in result["query"]
        assert result["query_refined"] is True
        assert "retrieved_docs" not in result


@patch("src.graphs.learn_nodes.get_cached_value", return_value=None)
@patch("src.graphs.learn_nodes.set_cached_value")
class TestGenerateStudyGuide:
    @patch("src.graphs.learn_nodes.get_settings")
    def test_fallback_when_no_api_key(self, mock_settings, _cache_set, _cache_get):
        mock_settings.return_value = MagicMock(openai_api_key="")
        docs = _make_docs(3)
        state = _base_state(retrieved_docs=docs)
        result = generate_study_guide(state)
        guide = result["study_guide"]
        assert isinstance(guide, StudyGuide)
        assert "overview" in guide.summary.lower()
        assert any("no API key" in t for t in result["trace"])

    @patch("src.graphs.learn_nodes.get_settings")
    def test_fallback_on_llm_error(self, mock_settings, _cache_set, _cache_get):
        mock_settings.return_value = MagicMock(
            openai_api_key="sk-test",
            app_default_model="gpt-4o-mini",
        )
        docs = _make_docs(2)
        state = _base_state(retrieved_docs=docs)
        with patch("src.graphs.learn_nodes.OpenAI", side_effect=Exception("API down")):
            result = generate_study_guide(state)
        guide = result["study_guide"]
        assert isinstance(guide, StudyGuide)
        assert any("error" in t.lower() or "fallback" in t.lower() for t in result["trace"])

    @patch("src.graphs.learn_nodes.get_settings")
    def test_successful_llm_call(self, mock_settings, _cache_set, _cache_get):
        mock_settings.return_value = MagicMock(
            openai_api_key="sk-test",
            app_default_model="gpt-4o-mini",
        )
        import json
        guide_json = json.dumps({
            "topic": "AI Agents",
            "difficulty": "intermediate",
            "summary": "Overview of AI Agents",
            "key_concepts": ["autonomy", "planning"],
            "detailed_notes": "Agents are autonomous systems.",
            "sources": [{"title": "Doc 1", "content_snippet": "...", "relevance_score": 0.9}],
        })
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=guide_json))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.graphs.learn_nodes.OpenAI", return_value=mock_client):
            result = generate_study_guide(_base_state(retrieved_docs=_make_docs(2)))

        guide = result["study_guide"]
        assert isinstance(guide, StudyGuide)
        assert guide.topic == "AI Agents"
        assert result["token_usage"]["total_tokens"] == 150

    @patch("src.graphs.learn_nodes.get_settings")
    def test_malformed_json_fallback(self, mock_settings, _cache_set, _cache_get):
        mock_settings.return_value = MagicMock(
            openai_api_key="sk-test",
            app_default_model="gpt-4o-mini",
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json{{{"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.graphs.learn_nodes.OpenAI", return_value=mock_client):
            result = generate_study_guide(_base_state(retrieved_docs=_make_docs(2)))

        guide = result["study_guide"]
        assert isinstance(guide, StudyGuide)
        assert any("malformed" in t.lower() for t in result["trace"])


class TestQualityCheck:
    def test_passes_good_guide(self):
        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Good summary",
            key_concepts=["concept1"],
            detailed_notes="Detailed content here.",
        )
        result = quality_check(_base_state(study_guide=guide))
        assert result["quality_passed"] is True

    def test_fails_empty_guide(self):
        guide = StudyGuide(topic="X", difficulty=DifficultyLevel.BEGINNER)
        result = quality_check(_base_state(study_guide=guide))
        assert result["quality_passed"] is False

    def test_fails_no_guide(self):
        result = quality_check(_base_state(study_guide=None))
        assert result["quality_passed"] is False


class TestPersistPlaceholder:
    def test_returns_trace(self):
        result = persist_learning_event_placeholder(_base_state())
        assert any("placeholder" in t for t in result["trace"])


class TestReturnOutput:
    def test_returns_trace(self):
        result = return_output(_base_state())
        assert any("done" in t for t in result["trace"])


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestLearnGraph:
    def test_graph_compiles(self):
        app = compile_learn_graph()
        assert app is not None

    def test_graph_has_all_nodes(self):
        graph = build_learn_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "validate_input",
            "load_user_memory",
            "retrieve_sources",
            "assess_source_quality",
            "refine_query_if_needed",
            "generate_study_guide",
            "quality_check",
            "persist_learning_event_placeholder",
            "return_output",
        }
        assert expected.issubset(node_names)


# ---------------------------------------------------------------------------
# Integration / routing tests
# ---------------------------------------------------------------------------

class TestLearnWorkflowRouting:
    @patch("src.graphs.learn_nodes.retrieve_documents")
    @patch("src.graphs.learn_nodes.get_settings")
    def test_sufficient_sources_skip_refinement(self, mock_settings, mock_retrieve):
        """When sources are sufficient, refinement is skipped."""
        mock_settings.return_value = MagicMock(openai_api_key="")
        docs = _make_docs(3, content_len=200)
        mock_retrieve.return_value = docs

        result = run_learn_workflow("AI Agents")
        trace = result.get("trace", [])
        trace_text = " ".join(trace)
        assert "refine_query_if_needed" not in trace_text
        assert result.get("study_guide") is not None

    @patch("src.graphs.learn_nodes.retrieve_documents")
    @patch("src.graphs.learn_nodes.get_settings")
    def test_insufficient_sources_trigger_refinement(self, mock_settings, mock_retrieve):
        """When sources are insufficient, query refinement is triggered and re-retrieval happens."""
        mock_settings.return_value = MagicMock(openai_api_key="")
        insufficient = _make_docs(1, content_len=10)
        sufficient = _make_docs(3, content_len=200)
        mock_retrieve.side_effect = [insufficient, sufficient]

        result = run_learn_workflow("AI Agents")
        trace = result.get("trace", [])
        trace_text = " ".join(trace)
        assert "refine_query_if_needed" in trace_text
        assert result.get("study_guide") is not None
        assert mock_retrieve.call_count == 2

    def test_empty_topic_returns_error(self):
        result = run_learn_workflow("")
        assert result.get("error") is not None
        assert "topic" in result["error"].lower()

    @patch("src.graphs.learn_nodes.get_cached_value", return_value=None)
    @patch("src.graphs.learn_nodes.retrieve_documents")
    @patch("src.graphs.learn_nodes.get_settings")
    def test_fallback_guide_has_sources(self, mock_settings, mock_retrieve, _mock_cache):
        """Fallback guide includes source information from retrieved docs."""
        mock_settings.return_value = MagicMock(openai_api_key="")
        docs = _make_docs(3, content_len=200)
        mock_retrieve.return_value = docs

        result = run_learn_workflow("AI Agents")
        guide = result.get("study_guide")
        assert guide is not None
        assert len(guide.sources) > 0


class TestBuildFallbackGuide:
    def test_builds_from_docs(self):
        docs = _make_docs(3)
        state = _base_state(retrieved_docs=docs)
        guide = _build_fallback_guide(state)
        assert guide.topic == "AI Agents"
        assert len(guide.sources) == 3
        assert "overview" in guide.summary.lower()

    def test_builds_from_empty_docs(self):
        state = _base_state(retrieved_docs=[])  
        guide = _build_fallback_guide(state)
        assert guide.topic == "AI Agents"
        assert "could not be fully generated" in guide.detailed_notes


# ---------------------------------------------------------------------------
# Deep Study Learn Path tests
# ---------------------------------------------------------------------------

_LEARN_PATH_TOPIC = (
    "Building Applications with LangChain, RAGs, and Streamlit: LangChain chains, "
    "retrieval-augmented generation, function calling, tool integration, "
    "Streamlit UI, and evaluation"
)


class TestIsDeepStudyLearnPath:
    def test_detects_deep_learn_path(self):
        state = _base_state(topic=_LEARN_PATH_TOPIC, style=ResponseStyle.DETAILED)
        assert is_deep_study_learn_path(state) is True

    def test_rejects_summary_learn_path(self):
        state = _base_state(topic=_LEARN_PATH_TOPIC, style=ResponseStyle.CONCISE)
        assert is_deep_study_learn_path(state) is False

    def test_rejects_deep_single_topic(self):
        state = _base_state(topic="AI Agents", style=ResponseStyle.DETAILED)
        assert is_deep_study_learn_path(state) is False


class TestExtractSummaryFromMarkdown:
    def test_extracts_first_paragraph(self):
        md = "# Handbook\n\nThis is the overview paragraph.\n\n## Section 1\nMore content."
        result = _extract_summary_from_markdown(md)
        assert "overview paragraph" in result

    def test_fallback_on_empty(self):
        result = _extract_summary_from_markdown("")
        assert "handbook" in result.lower()


class TestBuildSourcesList:
    def test_builds_sources(self):
        docs = _make_docs(3)
        state = _base_state(retrieved_docs=docs)
        sources = _build_sources_list(state)
        assert len(sources) == 3
        assert sources[0].title == "Topic 0"

    def test_limits_to_five(self):
        docs = _make_docs(8)
        state = _base_state(retrieved_docs=docs)
        sources = _build_sources_list(state)
        assert len(sources) == 5


@patch("src.graphs.learn_nodes.get_cached_value", return_value=None)
@patch("src.graphs.learn_nodes.set_cached_value")
class TestDeepStudyLearnPathFlow:
    @patch("src.graphs.learn_nodes.get_settings")
    def test_returns_markdown_guide(self, mock_settings, _cache_set, _cache_get):
        """Deep Study Learn Path returns a StudyGuide built from raw markdown."""
        mock_settings.return_value = MagicMock(
            openai_api_key="sk-test",
            app_default_model="gpt-4o-mini",
        )
        handbook_md = (
            "# Professional Curriculum Handbook\n\n"
            "This handbook covers LangChain and RAG topics in depth.\n\n"
            "## LangChain Chains\n\n### Theory\nLangChain provides...\n"
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=handbook_md))]
        mock_response.usage = MagicMock(prompt_tokens=500, completion_tokens=2000, total_tokens=2500)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        state = _base_state(
            topic=_LEARN_PATH_TOPIC,
            style=ResponseStyle.DETAILED,
            retrieved_docs=_make_docs(3),
        )

        with patch("src.graphs.learn_nodes.OpenAI", return_value=mock_client):
            result = generate_study_guide(state)

        guide = result["study_guide"]
        assert isinstance(guide, StudyGuide)
        # Should contain raw markdown in detailed_notes, NOT parsed JSON
        assert "LangChain" in guide.detailed_notes
        assert "handbook" in guide.detailed_notes.lower() or "chains" in guide.detailed_notes.lower()
        # key_concepts extracted from topic string
        assert len(guide.key_concepts) > 0
        # Sources built from retrieved docs
        assert len(guide.sources) > 0
        # Trace shows markdown flow
        assert any("deep_study_learn_path" in t for t in result["trace"])
        assert any("markdown" in t.lower() for t in result["trace"])

    @patch("src.graphs.learn_nodes.get_settings")
    def test_fallback_on_error(self, mock_settings, _cache_set, _cache_get):
        """Deep Study Learn Path falls back gracefully on LLM error."""
        mock_settings.return_value = MagicMock(
            openai_api_key="sk-test",
            app_default_model="gpt-4o-mini",
        )
        state = _base_state(
            topic=_LEARN_PATH_TOPIC,
            style=ResponseStyle.DETAILED,
            retrieved_docs=_make_docs(2),
        )
        with patch("src.graphs.learn_nodes.OpenAI", side_effect=Exception("API down")):
            result = generate_study_guide(state)

        guide = result["study_guide"]
        assert isinstance(guide, StudyGuide)
        assert any("error" in t.lower() for t in result["trace"])
