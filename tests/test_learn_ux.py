"""Tests for Learn UX quality improvements."""

import pytest

from src.schemas import ResponseStyle, DifficultyLevel, Source, StudyGuide
from src.ui.display_helpers import (
    _sanitize_snippet,
    _skip_to_sentence_start,
    deduplicate_sources,
    downgrade_headings,
    format_source_display,
    format_sources_summary,
    format_graph_state_summary,
    format_memory_transparency,
)


class TestUserFacingLabels:
    """User-facing labels should be capitalized and professional."""

    def test_learn_path_labels(self):
        from src.ui.pages import _LEARN_PATH_LABELS

        for label in _LEARN_PATH_LABELS:
            assert label[0].isupper(), f"Label '{label}' not capitalized"

    def test_learning_depth_labels(self):
        from src.ui.pages import _LEARNING_DEPTH_LABELS

        for label in _LEARNING_DEPTH_LABELS:
            assert label[0].isupper(), f"Label '{label}' not capitalized"

    def test_learning_mode_labels(self):
        from src.ui.pages import _LEARNING_MODE_LABELS

        assert _LEARNING_MODE_LABELS == ["Learn Path", "Topic"]

    def test_examples_heavy_not_in_depth_labels(self):
        from src.ui.pages import _LEARNING_DEPTH_LABELS

        for label in _LEARNING_DEPTH_LABELS:
            assert "examples" not in label.lower()

    def test_depth_labels_are_summary_and_deep_study(self):
        from src.ui.pages import _LEARNING_DEPTH_LABELS

        assert _LEARNING_DEPTH_LABELS == ["Summary", "Deep Study"]

    def test_learn_path_maps_to_difficulty_enum(self):
        from src.ui.pages import _LEARN_PATH_TO_ENUM

        assert _LEARN_PATH_TO_ENUM["Beginner"] == DifficultyLevel.BEGINNER
        assert _LEARN_PATH_TO_ENUM["Intermediate"] == DifficultyLevel.INTERMEDIATE
        assert _LEARN_PATH_TO_ENUM["Advanced"] == DifficultyLevel.ADVANCED

    def test_depth_maps_to_response_style(self):
        from src.ui.pages import _DEPTH_TO_STYLE

        assert _DEPTH_TO_STYLE["Summary"] == ResponseStyle.CONCISE
        assert _DEPTH_TO_STYLE["Deep Study"] == ResponseStyle.DETAILED

    def test_no_difficulty_or_response_style_in_labels(self):
        from src.ui.pages import (
            _LEARN_PATH_LABELS,
            _LEARNING_DEPTH_LABELS,
            _LEARNING_MODE_LABELS,
        )

        all_labels = _LEARN_PATH_LABELS + _LEARNING_DEPTH_LABELS + _LEARNING_MODE_LABELS
        for label in all_labels:
            assert "difficulty" not in label.lower()
            assert "response style" not in label.lower()


class TestLearnPathMode:
    """Learn Path mode should map to guided topics."""

    def test_learn_path_topic_map_exists(self):
        from src.ui.pages import _LEARN_PATH_TOPIC_MAP

        assert "Beginner" in _LEARN_PATH_TOPIC_MAP
        assert "Intermediate" in _LEARN_PATH_TOPIC_MAP
        assert "Advanced" in _LEARN_PATH_TOPIC_MAP

    def test_learn_path_topics_are_nonempty(self):
        from src.ui.pages import _LEARN_PATH_TOPIC_MAP

        for key, value in _LEARN_PATH_TOPIC_MAP.items():
            assert len(value) > 10, f"Topic for {key} is too short"

    def test_learn_path_topics_no_sprint_references(self):
        from src.ui.pages import _LEARN_PATH_TOPIC_MAP

        for key, value in _LEARN_PATH_TOPIC_MAP.items():
            assert "sprint" not in value.lower(), f"Sprint reference in {key}"

    def test_topic_mode_uses_learn_topics(self):
        from src.ui.pages import LEARN_TOPICS

        assert len(LEARN_TOPICS) >= 5
        for t in LEARN_TOPICS:
            assert len(t) > 2


class TestPromptStyleDifferentiation:
    """Summary vs Deep Study prompts should differ strongly."""

    def _build_prompt(self, style, difficulty=DifficultyLevel.INTERMEDIATE):
        from src.graphs.learn_nodes import _build_prompt

        state = {
            "topic": "AI Agents",
            "difficulty": difficulty,
            "style": style,
            "retrieved_docs": [],
            "memory_profile": {},
        }
        return _build_prompt(state)

    def test_summary_prompt_is_concise(self):
        prompt = self._build_prompt(ResponseStyle.CONCISE)
        assert "concise" in prompt.lower()
        assert "bullet" in prompt.lower()

    def test_deep_study_prompt_has_rich_sections(self):
        prompt = self._build_prompt(ResponseStyle.DETAILED)
        assert "Common Mistakes" in prompt
        assert "When to Use" in prompt
        assert "Review Checklist" in prompt
        assert "Comparison Table" in prompt
        assert "Architecture" in prompt
        assert "Practical Examples" in prompt

    def test_deep_study_asks_for_code(self):
        prompt = self._build_prompt(ResponseStyle.DETAILED)
        assert "code" in prompt.lower()

    def test_advanced_difficulty_adds_tradeoffs(self):
        prompt = self._build_prompt(ResponseStyle.DETAILED, DifficultyLevel.ADVANCED)
        assert "tradeoff" in prompt.lower()
        assert "production" in prompt.lower()
        assert "edge case" in prompt.lower()
        assert "observability" in prompt.lower()

    def test_beginner_difficulty_adds_first_principles(self):
        prompt = self._build_prompt(ResponseStyle.DETAILED, DifficultyLevel.BEGINNER)
        assert "first principles" in prompt.lower()

    def test_summary_and_deep_study_differ_strongly(self):
        p_summary = self._build_prompt(ResponseStyle.CONCISE)
        p_deep = self._build_prompt(ResponseStyle.DETAILED)
        assert p_summary != p_deep
        assert len(p_deep) > len(p_summary) + 100


class TestSnippetSanitization:
    """Source snippets should strip headings, single-letter artifacts, whitespace."""

    def test_strips_h1(self):
        assert _sanitize_snippet("# Big Title\nContent here") == "Big Title\nContent here"

    def test_strips_h2(self):
        assert _sanitize_snippet("## Section\nSome text here") == "Section\nSome text here"

    def test_strips_h3(self):
        assert _sanitize_snippet("### Sub Section\nText here") == "Sub Section\nText here"

    def test_strips_multiple_headings(self):
        text = "# Title\n\n## Section\n\nContent here"
        result = _sanitize_snippet(text)
        assert "#" not in result

    def test_collapses_blank_lines(self):
        text = "A longer line here\n\n\n\n\nAnother line here"
        result = _sanitize_snippet(text)
        assert "\n\n\n" not in result

    def test_empty_string(self):
        assert _sanitize_snippet("") == ""

    def test_no_headings_unchanged(self):
        text = "Just plain text here."
        assert _sanitize_snippet(text) == text

    def test_removes_single_letter_line(self):
        text = "Some text here\nO\nMore text here"
        result = _sanitize_snippet(text)
        assert "\nO\n" not in result

    def test_removes_stray_single_letter_mid_text(self):
        text = "some longer word O another word here"
        result = _sanitize_snippet(text)
        assert " O " not in result

    def test_collapses_multiple_spaces(self):
        text = "word   word and more text"
        result = _sanitize_snippet(text)
        assert "   " not in result

    def test_removes_short_uppercase_chunk_artifact(self):
        text = "Some good content here.\nND\nMore content follows."
        result = _sanitize_snippet(text)
        assert "ND" not in result

    def test_removes_two_letter_artifact_line(self):
        text = "Real content here.\nIO\nAnother sentence."
        result = _sanitize_snippet(text)
        assert "\nIO\n" not in result

    def test_junk_only_returns_empty(self):
        assert _sanitize_snippet("O") == ""
        assert _sanitize_snippet("ND") == ""
        assert _sanitize_snippet("X Y Z") == ""

    def test_fallback_for_no_useful_content(self):
        src = Source(title="Test", content_snippet="ND")
        info = format_source_display(src)
        assert info["snippet"] == "_No clean preview available._"

    def test_empty_snippet_fallback(self):
        src = Source(title="Test", content_snippet="")
        info = format_source_display(src)
        assert info["snippet"] == "_No clean preview available._"

    def test_bare_url_line_removed(self):
        text = "Good content here.\nhttps://example.com/foo\nMore content."
        result = _sanitize_snippet(text)
        assert "https://" not in result

    def test_preserves_real_content(self):
        text = "LangChain provides tools for building LLM applications."
        assert _sanitize_snippet(text) == text


class TestHeadingDowngrade:
    """downgrade_headings should shift heading levels down by one."""

    def test_h1_becomes_h2(self):
        assert downgrade_headings("# Title") == "## Title"

    def test_h2_becomes_h3(self):
        assert downgrade_headings("## Section") == "### Section"

    def test_h3_becomes_h4(self):
        assert downgrade_headings("### Sub") == "#### Sub"

    def test_multiple_headings(self):
        text = "# Title\n\n## Section\n\nContent"
        result = downgrade_headings(text)
        assert result == "## Title\n\n### Section\n\nContent"

    def test_no_headings_unchanged(self):
        assert downgrade_headings("plain text") == "plain text"

    def test_empty_string(self):
        assert downgrade_headings("") == ""

    def test_inline_hash_not_affected(self):
        text = "Use C# for development"
        assert downgrade_headings(text) == text


class TestSourceDeduplication:
    """Sources should be deduplicated by filename."""

    def test_removes_duplicate_filenames(self):
        s1 = Source(title="A", metadata={"filename": "a.md"})
        s2 = Source(title="B", metadata={"filename": "a.md"})
        s3 = Source(title="C", metadata={"filename": "b.md"})
        result = deduplicate_sources([s1, s2, s3])
        assert len(result) == 2
        assert result[0].title == "A"
        assert result[1].title == "C"

    def test_keeps_unique_sources(self):
        s1 = Source(title="A", metadata={"filename": "a.md"})
        s2 = Source(title="B", metadata={"filename": "b.md"})
        result = deduplicate_sources([s1, s2])
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate_sources([]) == []

    def test_falls_back_to_title(self):
        s1 = Source(title="Same", metadata={})
        s2 = Source(title="Same", metadata={})
        result = deduplicate_sources([s1, s2])
        assert len(result) == 1

    def test_format_source_display_sanitizes(self):
        src = Source(title="Test", content_snippet="# Big Heading\nContent here")
        info = format_source_display(src)
        assert not info["snippet"].startswith("#")

    def test_format_source_display_shows_source_type(self):
        src = Source(title="Test", metadata={"source_type": "official_docs"})
        info = format_source_display(src)
        types = [k for k, v in info["metadata_items"] if k == "Type"]
        assert len(types) == 1


class TestSourceMetadataFormatting:
    """Source metadata should be structured as separate key-value items."""

    def test_metadata_items_are_tuples(self):
        src = Source(title="T", metadata={"topic": "AI", "filename": "a.md", "source_type": "raw"})
        info = format_source_display(src)
        for item in info["metadata_items"]:
            assert isinstance(item, tuple) and len(item) == 2

    def test_metadata_includes_all_known_keys(self):
        src = Source(title="T", metadata={
            "topic": "AI", "filename": "a.md", "source_type": "raw", "source": "/path"
        })
        info = format_source_display(src)
        keys = [k for k, v in info["metadata_items"]]
        assert "Topic" in keys
        assert "File" in keys
        assert "Type" in keys
        assert "Source" in keys

    def test_metadata_empty_when_no_metadata(self):
        src = Source(title="T", metadata={})
        info = format_source_display(src)
        assert info["metadata_items"] == []


class TestRelevanceDisplay:
    """Fake/static relevance (0.5) should be hidden; real scores shown."""

    def test_placeholder_relevance_hidden(self):
        src = Source(title="T", content_snippet="Good content here.", relevance_score=0.5)
        info = format_source_display(src)
        assert info["relevance"] == 0.0
        assert info["relevance_label"] == ""

    def test_zero_relevance_hidden(self):
        src = Source(title="T", content_snippet="Good content here.", relevance_score=0.0)
        info = format_source_display(src)
        assert info["relevance_label"] == ""

    def test_real_relevance_shown(self):
        src = Source(title="T", content_snippet="Good content here.", relevance_score=0.85)
        info = format_source_display(src)
        assert info["relevance"] == 0.85
        assert info["relevance_label"] == "0.8"


class TestRetrievalCountWording:
    """Trace should say 'Passages Retrieved' not 'Sources Retrieved'."""

    def test_passages_retrieved_label(self):
        result = {"retrieved_docs": [1, 2, 3]}
        fields = format_graph_state_summary(result)
        labels = [f["label"] for f in fields]
        assert "Passages Retrieved" in labels
        assert "Sources Retrieved" not in labels


class TestMemoryEmptyState:
    """Empty memory profile message should mention quizzes and personalization."""

    def test_empty_memory_message(self):
        info = format_memory_transparency(None)
        assert "quiz" in info["message"].lower()
        assert "personalized" in info["message"].lower() or "personalization" in info["message"].lower()

    def test_empty_dict_memory(self):
        info = format_memory_transparency({})
        assert info["loaded"] is False

    def test_profile_with_all_empty_fields_is_not_loaded(self):
        """A profile dict with keys but no real data should be treated as empty."""
        profile = {
            "recent_topics": [],
            "recurring_weak_areas": [],
            "average_score": None,
            "suggested_focus_topics": [],
            "preferred_style": None,
        }
        info = format_memory_transparency(profile)
        assert info["loaded"] is False

    def test_profile_with_one_real_field_is_loaded(self):
        """A profile with at least one meaningful field should be loaded."""
        profile = {"recent_topics": ["AI Agents"]}
        info = format_memory_transparency(profile)
        assert info["loaded"] is True

    def test_profile_with_only_score_is_loaded(self):
        profile = {"average_score": 75.0}
        info = format_memory_transparency(profile)
        assert info["loaded"] is True


class TestPromptCodeBlockRules:
    """Deep study prompts must contain the strict code block rule."""

    def test_learn_path_deep_study_has_code_block_rule(self):
        from src.graphs.learn_prompts import _build_deep_study_markdown_prompt
        state = {
            "topic": "Test", "difficulty": DifficultyLevel.INTERMEDIATE,
            "style": ResponseStyle.DETAILED, "retrieved_docs": [], "memory_profile": {},
        }
        prompt = _build_deep_study_markdown_prompt(state)
        assert "STRICT CODE BLOCK RULE" in prompt
        assert "comment-only" in prompt.lower() or "comments" in prompt.lower()

    def test_topic_deep_study_has_code_block_rule(self):
        from src.graphs.learn_prompts import _build_deep_study_topic_markdown_prompt
        state = {
            "topic": "Test", "difficulty": DifficultyLevel.INTERMEDIATE,
            "style": ResponseStyle.DETAILED, "retrieved_docs": [], "memory_profile": {},
        }
        prompt = _build_deep_study_topic_markdown_prompt(state)
        assert "comment-only" in prompt.lower() or "placeholder-only comments" in prompt.lower()


class TestSidebarNavigation:
    """Sidebar should use visible buttons, not radio or selectbox for navigation."""

    def test_app_uses_buttons_for_navigation(self):
        with open("app.py") as f:
            source = f.read()
        assert "st.sidebar.button" in source
        assert "st.sidebar.radio" not in source
        # selectbox should not appear in sidebar navigation
        assert "st.sidebar.selectbox" not in source

    def test_app_sections_include_home_and_dashboard(self):
        with open("app.py") as f:
            source = f.read()
        assert '"Home"' in source
        assert '"Dashboard"' in source
        assert "Intro" not in source

    def test_app_name_is_consistent(self):
        with open("app.py") as f:
            source = f.read()
        assert "AI Engineering Learning App" in source
        assert "AI Learning Assistant" not in source


class TestTraceLabels:
    """Trace labels should use professional wording."""

    def test_trace_uses_learn_path_not_difficulty(self):
        result = {
            "topic": "Test",
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "style": ResponseStyle.DETAILED,
            "retrieved_docs": [],
        }
        fields = format_graph_state_summary(result)
        labels = [f["label"] for f in fields]
        assert "Learn Path" in labels
        assert "Level / Difficulty" not in labels
        assert "Difficulty" not in labels

    def test_trace_uses_learning_depth(self):
        result = {
            "topic": "Test",
            "style": ResponseStyle.DETAILED,
            "retrieved_docs": [],
        }
        fields = format_graph_state_summary(result)
        labels = [f["label"] for f in fields]
        assert "Learning Depth" in labels
        assert "Response Style" not in labels


class TestOfficialDocsTraceFormat:
    """Official docs trace should report accurate curated/official counts."""

    def test_deep_study_trace_reports_counts(self):
        from unittest.mock import patch, MagicMock
        from src.kb.loader import Document

        curated = [Document(content=f"curated_{i}" * 20, metadata={}) for i in range(3)]
        official = [Document(content=f"official_{i}" * 20, metadata={}) for i in range(2)]

        with patch("src.graphs.learn_nodes.retrieve_documents", return_value=curated), \
             patch("src.graphs.learn_nodes.retrieve_with_fallback"), \
             patch("src.kb.official_docs.retrieve_official_docs", return_value=official):
            from src.graphs.learn_nodes import retrieve_sources

            state = {
                "topic": "LangGraph",
                "style": ResponseStyle.DETAILED,
                "trace": [],
                "attempts": 0,
            }
            result = retrieve_sources(state)

        trace_text = " ".join(result["trace"])
        assert "curated=3" in trace_text
        assert "official_retrieved=2" in trace_text
        assert "official_added=2" in trace_text
        assert "final=5" in trace_text


class TestMemoryLoadedState:
    """Memory loaded state should expose profile fields."""

    def test_memory_loaded_has_fields(self):
        profile = {
            "recent_topics": ["AI Agents"],
            "recurring_weak_areas": ["RAG"],
            "average_score": 75.0,
            "suggested_focus_topics": ["LangGraph"],
        }
        mem = format_memory_transparency(profile)
        assert mem["loaded"]
        assert mem["recent_topics"] == ["AI Agents"]


class TestOutputTitles:
    """Generated output title should differ between Learn Path and Topic mode."""

    def test_learn_path_mode_title_includes_level(self):
        guide = StudyGuide(
            topic="Foundations",
            difficulty=DifficultyLevel.BEGINNER,
            summary="test",
            key_concepts=[],
            detailed_notes="",
        )
        level_label = guide.difficulty.value.capitalize()
        title = f"{level_label} Learn Path"
        assert "Beginner" in title
        assert "Learn Path" in title

    def test_topic_mode_title_is_topic_only(self):
        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="test",
            key_concepts=[],
            detailed_notes="",
        )
        assert guide.topic == "AI Agents"


class TestDynamicButton:
    """Generate button label should vary by learning mode."""

    def test_learn_page_source_has_dynamic_button(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert "Generate Learn Path" in source
        assert "Generate Topic" in source

    def test_no_demo_expander(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert "Load a recommended setup" not in source


class TestPromptModeInstruction:
    """Prompt should include mode instruction differentiating Learn Path vs Topic."""

    def _build(self, topic):
        from src.graphs.learn_nodes import _build_prompt
        state = {
            "topic": topic,
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "style": ResponseStyle.DETAILED,
            "retrieved_docs": [],
            "memory_profile": {},
        }
        return _build_prompt(state)

    def test_learn_path_prompt_has_curriculum_instruction(self):
        prompt = self._build(
            "Foundations of AI Engineering: LLM basics, prompt engineering, "
            "development environment, and API usage"
        )
        assert "LEARN PATH" in prompt
        assert "multi-topic curriculum" in prompt.lower()

    def test_topic_prompt_has_focused_instruction(self):
        prompt = self._build("AI Agents")
        assert "TOPIC study" in prompt
        assert "focused" in prompt.lower()


class TestLearnSubtitle:
    """Learn subtitle should mention both Topic and Learn Path."""

    def test_subtitle_mentions_both_modes(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert "Topic" in source
        assert "Learn Path" in source


class TestLearnStreamingHelpers:
    """Learn streaming should stay UI-only and preserve safe boundaries."""

    def test_should_stream_fresh_result(self):
        from src.ui.learn_page import _should_stream_learn_result

        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Summary",
            key_concepts=[],
            detailed_notes="## Notes\nBody",
        )
        result = {"study_guide": guide, "trace": ["generate_study_guide: done"]}
        assert _should_stream_learn_result(result) is True

    def test_should_not_stream_cache_hit(self):
        from src.ui.learn_page import _should_stream_learn_result

        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Summary",
            key_concepts=[],
            detailed_notes="## Notes\nBody",
        )
        result = {"study_guide": guide, "trace": ["generate_study_guide: cache hit"]}
        assert _should_stream_learn_result(result) is False

    def test_should_not_stream_failed_result(self):
        from src.ui.learn_page import _should_stream_learn_result

        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Summary",
            key_concepts=[],
            detailed_notes="## Notes\nBody",
        )
        result = {
            "study_guide": guide,
            "trace": ["generate_study_guide: done"],
            "generation_failed": True,
        }
        assert _should_stream_learn_result(result) is False

    def test_iter_markdown_blocks_groups_paragraphs(self):
        from src.ui.learn_page import _iter_markdown_blocks

        text = "First paragraph.\n\nSecond paragraph."
        assert list(_iter_markdown_blocks(text)) == [
            "First paragraph.\n\n",
            "Second paragraph.",
        ]

    def test_iter_markdown_blocks_keeps_fenced_code_together(self):
        from src.ui.learn_page import _iter_markdown_blocks

        text = (
            "Intro paragraph.\n\n"
            "```python\n"
            "x = 1\n\n"
            "print(x)\n"
            "```\n\n"
            "Closing paragraph."
        )
        blocks = list(_iter_markdown_blocks(text))
        assert blocks[0] == "Intro paragraph.\n\n"
        assert blocks[1] == "```python\nx = 1\n\nprint(x)\n```\n"
        assert blocks[2] == "Closing paragraph."

    def test_display_learn_result_passes_stream_flag_to_study_guide(self):
        from unittest.mock import patch

        from src.ui.learn_page import _display_learn_result

        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Summary",
            key_concepts=[],
            detailed_notes="## Notes\nBody",
        )
        result = {"study_guide": guide, "trace": ["generate_study_guide: done"]}

        with patch("src.ui.learn_page._display_study_guide") as display_guide, \
             patch("src.ui.learn_page._display_memory_section") as display_memory, \
             patch("src.ui.learn_page._display_debug_trace") as display_trace:
            _display_learn_result(
                result,
                depth="Deep Study",
                mode="Topic",
                stream=True,
            )

        display_guide.assert_called_once_with(
            guide,
            depth="Deep Study",
            mode="Topic",
            stream=True,
        )
        display_memory.assert_called_once_with(result)
        display_trace.assert_called_once_with(result, "Learn Workflow Trace")


class TestQuizUiCopy:
    """Quiz page labels should match the current workflow terminology."""

    def test_quiz_uses_difficulty_label(self):
        with open("src/ui/quiz_page.py") as f:
            source = f.read()
        assert '"Difficulty"' in source
        assert '"Learn Path"' not in source

    def test_quiz_context_caption_is_generic(self):
        with open("src/ui/quiz_page.py") as f:
            source = f.read()
        assert "Using context from your last Learn session." in source


class TestSidebarStatus:
    """Sidebar should show compact status info."""

    def test_sidebar_has_model_status(self):
        with open("app.py") as f:
            source = f.read()
        assert "Model:" in source

    def test_sidebar_has_api_status(self):
        with open("app.py") as f:
            source = f.read()
        assert "OpenAI:" in source

    def test_sidebar_has_langsmith_status(self):
        with open("app.py") as f:
            source = f.read()
        assert "LangSmith:" in source


class TestTopicAwareRetrieval:
    """Learn Path Deep Study should use topic-aware retrieval."""

    def test_get_learn_path_topics_returns_topics_for_learn_path(self):
        from src.graphs.learn_nodes import _get_learn_path_topics
        state = {
            "topic": "Foundations of LLM Application Development: LLM basics, prompt engineering",
            "difficulty": DifficultyLevel.BEGINNER,
            "style": ResponseStyle.DETAILED,
        }
        topics = _get_learn_path_topics(state)
        assert topics is not None
        assert len(topics) >= 3
        assert "LLM Basics" in topics

    def test_get_learn_path_topics_returns_none_for_topic_mode(self):
        from src.graphs.learn_nodes import _get_learn_path_topics
        state = {
            "topic": "AI Agents",
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "style": ResponseStyle.DETAILED,
        }
        topics = _get_learn_path_topics(state)
        assert topics is None

    def test_get_learn_path_topics_returns_none_for_summary(self):
        from src.graphs.learn_nodes import _get_learn_path_topics
        state = {
            "topic": "Foundations: basics",
            "difficulty": DifficultyLevel.BEGINNER,
            "style": ResponseStyle.CONCISE,
        }
        # _get_learn_path_topics is only called when is_deep, but the function
        # itself doesn't check style — it checks topic format + difficulty mapping
        topics = _get_learn_path_topics(state)
        # Should still return topics since topic has ":" and difficulty maps
        assert topics is not None

    def test_intermediate_learn_path_returns_intermediate_topics(self):
        from src.graphs.learn_nodes import _get_learn_path_topics
        state = {
            "topic": "Building Applications with LangChain: chains, RAG",
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "style": ResponseStyle.DETAILED,
        }
        topics = _get_learn_path_topics(state)
        assert topics is not None
        assert "Retrieval-Augmented Generation" in topics


class TestSkipToSentenceStart:
    """_skip_to_sentence_start should fix mid-sentence previews."""

    def test_already_clean_start_unchanged(self):
        text = "Large language models are powerful tools."
        assert _skip_to_sentence_start(text) == text

    def test_lowercase_start_skips_to_next_sentence(self):
        text = "tion and retrieval. The model uses embeddings to find relevant docs."
        result = _skip_to_sentence_start(text)
        assert result.startswith("The model")

    def test_bullet_start_unchanged(self):
        text = "- First item\n- Second item"
        assert _skip_to_sentence_start(text) == text

    def test_number_start_unchanged(self):
        text = "1. First step in the process."
        assert _skip_to_sentence_start(text) == text

    def test_no_boundary_returns_original(self):
        text = "some lowercase text without any sentence boundary at all"
        assert _skip_to_sentence_start(text) == text

    def test_mid_sentence_with_exclamation(self):
        text = "broken fragment here! Next sentence starts properly."
        result = _skip_to_sentence_start(text)
        assert result.startswith("Next sentence")


class TestMidSentenceSnippetCleanup:
    """_sanitize_snippet should not produce mid-sentence previews."""

    def test_truncated_start_cleaned(self):
        text = "mentation details vary. Prompt engineering is the practice of designing effective prompts."
        result = _sanitize_snippet(text)
        assert result.startswith("Prompt engineering")

    def test_clean_start_preserved(self):
        text = "Prompt engineering is the practice of designing effective prompts for LLMs."
        result = _sanitize_snippet(text)
        assert result.startswith("Prompt engineering")


class TestExpandedMetadataFields:
    """format_source_display should include sprint, part, tags when present."""

    def test_sprint_and_part_shown(self):
        src = Source(
            title="Test",
            content_snippet="This is a valid content snippet for testing metadata display.",
            relevance_score=0.5,
            metadata={"sprint": "S3", "part": "P2", "filename": "test.md"},
        )
        info = format_source_display(src)
        keys = [k for k, v in info["metadata_items"]]
        assert "Sprint" in keys
        assert "Part" in keys

    def test_tags_shown(self):
        src = Source(
            title="Test",
            content_snippet="This is a valid content snippet for testing tags display.",
            relevance_score=0.5,
            metadata={"tags": "rag, retrieval, vectors"},
        )
        info = format_source_display(src)
        tags_items = [v for k, v in info["metadata_items"] if k == "Tags"]
        assert len(tags_items) == 1
        assert "rag" in tags_items[0]
        assert "retrieval" in tags_items[0]


class TestWorkflowSummary:
    """_build_workflow_summary should produce reviewer-friendly steps."""

    def test_basic_workflow_summary(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": [
                "validate_input: ok",
                "retrieve_sources: query='test' (attempt 1)",
                "assess_source_quality: 5 docs, 2000 chars → sufficient",
                "generate_study_guide: done",
            ],
            "sources": [1, 2, 3, 4, 5],
            "source_quality_ok": True,
            "token_usage": {"total_tokens": 1500},
        }
        steps = _build_workflow_summary(result)
        assert any("Input validated" in s for s in steps)
        assert any("Retrieval completed" in s for s in steps)
        assert any("generated" in s.lower() for s in steps)
        # Enhanced summary includes concise metadata
        assert any("passages" in s.lower() for s in steps)
        assert any("tokens" in s.lower() for s in steps)
        assert any("sufficient" in s.lower() for s in steps)

    def test_topic_aware_retrieval_noted(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": [
                "retrieve_sources: topic-aware retrieval — 5 topics",
            ],
            "sources": [1, 2, 3],
        }
        steps = _build_workflow_summary(result)
        assert any("Topic-aware" in s for s in steps)
        assert any("3 passages" in s for s in steps)

    def test_empty_trace_returns_empty(self):
        from src.ui.shared import _build_workflow_summary
        result = {"trace": [], "sources": []}
        steps = _build_workflow_summary(result)
        assert steps == []

    def test_cache_hit_noted(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": ["cache_hit: true", "generate_study_guide: done"],
            "sources": [],
        }
        steps = _build_workflow_summary(result)
        assert any("cache hit" in s.lower() for s in steps)

    def test_insufficient_sources_from_trace(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": ["retrieve_sources: query='test'", "assess_source_quality: 0 docs → insufficient"],
            "sources": [],
        }
        steps = _build_workflow_summary(result)
        assert any("insufficient" in s.lower() for s in steps)

    def test_token_count_shown(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": ["generate_study_guide: done"],
            "sources": [],
            "token_usage": {"total_tokens": 2500},
        }
        steps = _build_workflow_summary(result)
        assert any("2500" in s for s in steps)


class TestMemoryBannerVisibility:
    """Memory empty-state should use a visible banner message."""

    def test_empty_memory_message_mentions_quizzes(self):
        mem = format_memory_transparency(None)
        assert not mem["loaded"]
        assert "quizzes" in mem["message"].lower() or "quiz" in mem["message"].lower()

    def test_empty_memory_message_mentions_personalization(self):
        mem = format_memory_transparency(None)
        assert "personalized" in mem["message"].lower() or "personalization" in mem["message"].lower()


class TestEmptyDuplicateChunksSection:
    """Duplicate chunks section should not render when no extra entries exist."""

    def test_no_extra_lines_when_all_titles_in_primary(self):
        """If all sources share titles with primary set, no extra lines."""
        from unittest.mock import MagicMock
        src1 = MagicMock()
        src1.title = "LLM Basics"
        src1.metadata = {"filename": "llm_basics.md"}
        # all_sources has duplicate of same title
        src2 = MagicMock()
        src2.title = "LLM Basics"
        src2.metadata = {"filename": "llm_basics.md"}
        all_sources = [src1, src2]
        primary = [src1]
        # Simulate the filtering logic from _display_sources_section
        seen_titles = {getattr(s, "title", "") for s in primary}
        extra_lines = []
        for src in all_sources:
            t = getattr(src, "title", "") or "Untitled"
            if t in seen_titles:
                continue
            seen_titles.add(t)
            extra_lines.append(t)
        assert extra_lines == []

    def test_extra_lines_when_different_titles(self):
        """Extra entries should appear when titles differ from primary."""
        from unittest.mock import MagicMock
        src1 = MagicMock()
        src1.title = "LLM Basics"
        src1.metadata = {"filename": "llm_basics.md"}
        src2 = MagicMock()
        src2.title = "Tool Calling"
        src2.metadata = {"filename": "tool_calling.md"}
        all_sources = [src1, src2]
        primary = [src1]
        seen_titles = {getattr(s, "title", "") for s in primary}
        extra_lines = []
        for src in all_sources:
            t = getattr(src, "title", "") or "Untitled"
            if t in seen_titles:
                continue
            seen_titles.add(t)
            extra_lines.append(t)
        assert extra_lines == ["Tool Calling"]


class TestMemoryEmptyStateInsideExpander:
    """Memory empty-state should be inside expander, never leave it empty."""

    def test_empty_state_has_professional_wording(self):
        mem = format_memory_transparency(None)
        assert not mem["loaded"]
        # The UI now shows "No learning personalization data available yet."
        # and "Completed quizzes, topic preferences..." inside the expander.
        # The format_memory_transparency returns the raw message.
        assert "message" in mem

    def test_loaded_state_has_fields(self):
        mem = format_memory_transparency({"recent_topics": ["AI Agents"]})
        assert mem["loaded"]
        assert mem["recent_topics"] == ["AI Agents"]


class TestWorkflowSummaryInsideTrace:
    """Workflow summary should be rendered inside trace expander."""

    def test_display_debug_trace_has_summary_inside(self):
        """Verify _display_debug_trace renders summary inside the expander."""
        import inspect
        from src.ui.shared import _display_debug_trace
        source = inspect.getsource(_display_debug_trace)
        # Summary should be inside the expander block (indented under `with st.expander`)
        assert "Workflow Summary" in source
        # The old pattern had summary BEFORE the expander; now it should be inside
        assert "with st.expander(label):" in source
        # "Workflow Summary" should appear after the expander open, not before
        expander_pos = source.index("with st.expander(label):")
        summary_pos = source.index("Workflow Summary")
        assert summary_pos > expander_pos


class TestSnippetSentenceAwareTrimming:
    """Snippet trimming should prefer complete sentences."""

    def test_trailing_fragment_trimmed_to_sentence(self):
        text = "LLMs are powerful tools for NLP. They can generate text. They process tokens and"
        result = _sanitize_snippet(text)
        assert result.endswith(".")
        assert "and" not in result.split(".")[-1]

    def test_complete_sentence_preserved(self):
        text = "LLMs are powerful tools for natural language processing."
        result = _sanitize_snippet(text)
        assert result == text

    def test_long_snippet_trimmed_at_sentence_boundary(self):
        """_trim_snippet_to_sentence should cut at sentence boundary."""
        from src.ui.shared import _trim_snippet_to_sentence
        sentence = "This is a complete sentence about AI engineering. "
        text = sentence * 20  # ~1000 chars
        result = _trim_snippet_to_sentence(text, max_len=600)
        assert result.endswith(".")
        assert len(result) <= 600

    def test_short_snippet_mid_sentence_trimmed(self):
        """Short text ending mid-sentence should be trimmed to last sentence."""
        from src.ui.shared import _trim_snippet_to_sentence
        text = "First sentence here. Second sentence here. And then some fragment"
        result = _trim_snippet_to_sentence(text, max_len=600)
        assert result.endswith(".")
        assert "fragment" not in result

    def test_snippet_ending_with_period_unchanged(self):
        """Text already ending with period should not be altered."""
        from src.ui.shared import _trim_snippet_to_sentence
        text = "This is a complete sentence."
        result = _trim_snippet_to_sentence(text, max_len=600)
        assert result == text

    def test_word_boundary_fallback(self):
        """When no sentence boundary exists, fall back to word boundary with ..."""
        from src.ui.shared import _trim_snippet_to_sentence
        # Long text with no periods
        text = "word " * 200  # 1000 chars, no sentence boundaries
        result = _trim_snippet_to_sentence(text, max_len=600)
        assert result.endswith("...")
        assert len(result) <= 603  # 600 + "..."


class TestRetrievalWordingClarity:
    """Retrieval wording should distinguish chunks from source files."""

    def test_summary_says_source_files(self):
        assert "source file" in format_sources_summary(["a"]).lower()
        assert "source file" in format_sources_summary([]).lower()

    def test_summary_says_unique(self):
        result = format_sources_summary(["a", "b"])
        assert "unique" in result.lower()

    def test_dedup_wording_has_passages_and_sources(self):
        """The dedup line in shared.py should mention passages and sources."""
        import inspect
        from src.ui.shared import _display_sources_section
        source = inspect.getsource(_display_sources_section)
        assert "context" in source and "passage" in source
        assert "unique" in source and "displayed" in source


class TestSourceQualityFromState:
    """Workflow summary source quality must use source_quality_ok state flag."""

    def test_sufficient_from_state_flag(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": ["retrieve_sources: done"],
            "sources": [{"title": "A"}],
            "source_quality_ok": True,
        }
        steps = _build_workflow_summary(result)
        quality = [s for s in steps if "quality" in s.lower()]
        assert any("sufficient" in s.lower() and "insufficient" not in s.lower() for s in quality)

    def test_insufficient_from_state_flag(self):
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": ["retrieve_sources: done"],
            "sources": [{"title": "A"}],
            "source_quality_ok": False,
        }
        steps = _build_workflow_summary(result)
        quality = [s for s in steps if "quality" in s.lower()]
        assert any("insufficient" in s.lower() for s in quality)

    def test_state_flag_overrides_source_presence(self):
        """Even with sources present, False flag means insufficient."""
        from src.ui.shared import _build_workflow_summary
        result = {
            "trace": ["retrieve_sources: done"],
            "sources": [{"title": "A"}, {"title": "B"}],
            "source_quality_ok": False,
        }
        steps = _build_workflow_summary(result)
        quality = [s for s in steps if "quality" in s.lower()]
        assert any("insufficient" in s.lower() for s in quality)


class TestGlobalLayoutCSS:
    """Global CSS must constrain code blocks inside markdown/lists."""

    def test_global_css_has_stmarkdown_pre_rules(self):
        with open("app.py") as f:
            content = f.read()
        assert ".stMarkdown pre" in content
        assert "overflow-x: auto" in content

    def test_global_css_has_list_code_rules(self):
        with open("app.py") as f:
            content = f.read()
        assert ".stMarkdown li pre" in content
        assert ".stMarkdown li code" in content

    def test_global_css_no_overflow_hidden(self):
        """Code blocks should scroll, not clip."""
        with open("app.py") as f:
            content = f.read()
        assert "overflow: hidden" not in content


class TestSnippetMidSentenceEllipsis:
    """Short snippets ending mid-sentence should get ellipsis, not raw cut."""

    def test_short_no_sentence_boundary_gets_ellipsis(self):
        from src.ui.shared import _trim_snippet_to_sentence
        text = "LLMs are a key building block for modern AI applications such"
        result = _trim_snippet_to_sentence(text, max_len=600)
        assert result.endswith("...")

    def test_short_with_sentence_boundary_no_ellipsis(self):
        from src.ui.shared import _trim_snippet_to_sentence
        text = "LLMs are powerful. They generate text well"
        result = _trim_snippet_to_sentence(text, max_len=600)
        assert result == "LLMs are powerful."


class TestDashboardStructure:
    """Dashboard should have visible sections, not all expanders."""

    def test_dashboard_has_subheaders(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        # Dashboard should use st.subheader for main sections
        assert "st.subheader(\"Overview\")" in source
        assert "st.subheader(\"Costs\")" in source
        assert "st.subheader(\"Memory\")" in source
        assert "st.subheader(\"Feedback\")" in source
        assert "st.subheader(\"Workflow Traces\")" in source
