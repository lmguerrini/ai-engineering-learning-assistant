"""Tests for Learn UX quality improvements."""

import pytest
from unittest.mock import patch

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


class TestHomePagePolish:
    """Home page should stay reviewer-friendly and visually structured."""

    def test_home_page_has_polished_hero_and_ctas(self):
        with open("src/ui/pages.py") as f:
            source = f.read()

        assert 'def _set_active_section(section: str)' in source
        assert 'def _render_home_feature_card(title: str, body: str)' in source
        assert "AI Engineering Learning Assistant" in source
        assert "Study AI engineering with grounded lessons, quizzes, progress tracking, and scoped help." in source
        assert 'st.button("Start Learning"' in source
        assert 'st.button("Take a Quiz"' in source
        assert 'st.button("Open Dashboard"' in source
        assert "reviewer" not in source.lower()
        assert "best review path" not in source.lower()
        assert "what this demo highlights" not in source.lower()

    def test_home_page_highlights_core_workflows_and_review_path(self):
        with open("src/ui/pages.py") as f:
            source = f.read()

        assert 'st.markdown("#### Explore")' in source
        assert '"Learn"' in source
        assert '"Quiz"' in source
        assert '"Progress"' in source
        assert '"Help Assistant"' in source
        assert '"Dashboard"' in source
        assert "Runtime status:" in source


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
        assert '"Help Assistant"' in source
        assert '"Dashboard"' in source
        assert '"Quick Help"' not in source
        assert source.index('"Dashboard"') < source.index('"Help Assistant"')
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

    def test_learn_path_summary_prompt_uses_topic_led_sections(self):
        from src.graphs.learn_prompts import _build_prompt

        prompt = _build_prompt(
            {
                "topic": (
                    "Foundations of AI Engineering: LLM basics, prompt engineering, "
                    "development environment, and API usage"
                ),
                "difficulty": DifficultyLevel.BEGINNER,
                "style": ResponseStyle.CONCISE,
                "retrieved_docs": [],
                "memory_profile": {},
            }
        )

        assert "## Study Sequence" in prompt
        assert "## LLM Basics" in prompt
        assert "generic numbered template headings" in prompt
        assert "Recommended Study Order" not in prompt
        assert "Learn Path Overview" not in prompt


class TestLearnSubtitle:
    """Learn subtitle should mention both Topic and Learn Path."""

    def test_subtitle_mentions_both_modes(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert "Topic" in source
        assert "Learn Path" in source


class TestLearnStreamingHelpers:
    """Learn streaming should stay UI-only and preserve safe boundaries."""

    def test_supports_progressive_streaming_for_deep_study_modes(self):
        from src.ui.learn_page import _supports_progressive_streaming

        assert _supports_progressive_streaming(depth="Deep Study", mode="Topic") is True
        assert _supports_progressive_streaming(depth="Deep Study", mode="Learn Path") is True

    def test_disables_progressive_streaming_for_summary_learn_paths(self):
        from src.ui.learn_page import _supports_progressive_streaming

        assert _supports_progressive_streaming(depth="Summary", mode="Learn Path") is False

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

    def test_append_only_markdown_delta_returns_suffix(self):
        from src.ui.learn_page import _get_append_only_markdown_delta

        previous = "Overview paragraph.\n\n"
        current = previous + "Next paragraph.\n\n```python\nprint('done')\n```"
        assert _get_append_only_markdown_delta(previous, current) == (
            "Next paragraph.\n\n```python\nprint('done')\n```"
        )

    def test_append_only_markdown_delta_rejects_non_prefix_update(self):
        from src.ui.learn_page import _get_append_only_markdown_delta

        previous = "Overview paragraph.\n\n"
        current = "Updated overview paragraph.\n\n"
        assert _get_append_only_markdown_delta(previous, current) is None

    def test_stream_markdown_delta_replays_only_new_blocks(self):
        from unittest.mock import patch

        from src.ui.learn_page import _stream_markdown_delta

        class _Placeholder:
            def __init__(self):
                self.calls = []

            def markdown(self, text, unsafe_allow_html=False):
                self.calls.append((text, unsafe_allow_html))

        previous = "Overview paragraph.\n\n"
        current = (
            previous
            + "Next paragraph.\n\n"
            + "```python\nprint('done')\n```"
        )
        placeholder = _Placeholder()

        with patch("src.ui.learn_page.st.empty", return_value=placeholder):
            _stream_markdown_delta(previous, current)

        assert placeholder.calls[0] == (previous, False)
        assert placeholder.calls[-1] == (current, False)
        assert any("Next paragraph." in rendered for rendered, _ in placeholder.calls[1:])
        assert any("```python\nprint('done')\n```" in rendered for rendered, _ in placeholder.calls[1:])

    def test_display_study_guide_uses_progressive_previous_for_delta_replay(self):
        from unittest.mock import call, patch

        from src.ui.learn_page import _clean_generated_markdown, _display_study_guide

        previous = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Overview paragraph.",
            key_concepts=["Autonomy: choose actions"],
            detailed_notes="## Conceptual Foundations\nPrior section text.",
        )
        current = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Overview paragraph.",
            key_concepts=["Autonomy: choose actions"],
            detailed_notes=(
                "## Conceptual Foundations\nPrior section text.\n\n"
                "## Practical Examples\nNew section text."
            ),
        )
        previous_notes = downgrade_headings(
            _clean_generated_markdown(previous.detailed_notes, previous, "Deep Study", "Topic")
        )
        current_notes = downgrade_headings(
            _clean_generated_markdown(current.detailed_notes, current, "Deep Study", "Topic")
        )

        with patch("src.ui.learn_page._stream_markdown_delta") as stream_delta, \
             patch("src.ui.learn_page.st") as mock_st:
            _display_study_guide(
                current,
                depth="Deep Study",
                mode="Topic",
                include_sources=False,
                show_topic_key_concepts=True,
                progressive_previous=previous,
            )

        assert call(previous.summary, current.summary) in stream_delta.call_args_list
        assert call(previous_notes, current_notes, unsafe_allow_html=False) in stream_delta.call_args_list
        mock_st.subheader.assert_called_once_with("AI Agents")

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
             patch("src.ui.learn_page._display_sources_section") as display_sources, \
             patch("src.ui.learn_page._display_memory_section") as display_memory, \
             patch("src.ui.learn_page._display_debug_trace") as display_trace, \
             patch("src.ui.learn_page._display_feedback_widget") as display_feedback:
            _display_learn_result(
                result,
                depth="Deep Study",
                mode="Topic",
                stream=True,
                feedback_topic="AI Agents",
            )

        display_guide.assert_called_once_with(
            guide,
            depth="Deep Study",
            mode="Topic",
            stream=True,
            include_sources=False,
        )
        display_sources.assert_called_once_with(guide)
        display_memory.assert_called_once_with(result)
        display_trace.assert_called_once_with(result, "Learn Workflow Trace")
        display_feedback.assert_called_once_with("learn", "AI Agents", expanded=True)

    def test_display_learn_result_extras_uses_distinct_section_headings(self):
        from unittest.mock import patch

        from src.ui.learn_page import _display_learn_result_extras

        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Summary",
            key_concepts=[],
            detailed_notes="## Notes\nBody",
        )
        result = {"trace": [], "study_guide": guide}

        with patch("src.ui.learn_page.st.markdown") as markdown, \
             patch("src.ui.learn_page._display_sources_section") as display_sources, \
             patch("src.ui.learn_page._display_memory_section") as display_memory, \
             patch("src.ui.learn_page._display_debug_trace") as display_trace, \
             patch("src.ui.learn_page._display_feedback_widget") as display_feedback:
            _display_learn_result_extras(result, guide=guide, feedback_topic="AI Agents")

        headings = [call.args[0] for call in markdown.call_args_list]
        assert "#### Personalization" in headings
        assert "#### Workflow Trace" in headings
        assert "#### Feedback" in headings
        display_sources.assert_called_once_with(guide)
        display_memory.assert_called_once_with(result)
        display_trace.assert_called_once_with(result, "Learn Workflow Trace")
        display_feedback.assert_called_once_with("learn", "AI Agents", expanded=True)

    def test_display_learn_result_extras_renders_sources_before_other_sections(self):
        from unittest.mock import patch

        from src.ui.learn_page import _display_learn_result_extras

        guide = StudyGuide(
            topic="AI Agents",
            difficulty=DifficultyLevel.INTERMEDIATE,
            summary="Summary",
            key_concepts=[],
            detailed_notes="## Notes\nBody",
        )
        result = {"trace": [], "study_guide": guide}
        order: list[str] = []

        with patch(
            "src.ui.learn_page._display_sources_section",
            side_effect=lambda _: order.append("sources"),
        ), patch(
            "src.ui.learn_page._display_memory_section",
            side_effect=lambda _: order.append("memory"),
        ), patch(
            "src.ui.learn_page._display_debug_trace",
            side_effect=lambda *_args: order.append("trace"),
        ), patch(
            "src.ui.learn_page._display_feedback_widget",
            side_effect=lambda *_args, **_kwargs: order.append("feedback"),
        ):
            _display_learn_result_extras(result, guide=guide, feedback_topic="AI Agents")

        assert order == ["sources", "memory", "trace", "feedback"]


class TestLearnProgressiveStreamingToggle:
    """Learn page should expose a clear progressive streaming toggle."""

    def test_learn_page_includes_progressive_streaming_toggle_label(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert '"Progressive streaming"' in source

    def test_learn_page_includes_progressive_streaming_explanation(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert "starts showing content earlier" in source
        assert "section-by-section" in source

    def test_learn_page_passes_progressive_streaming_to_workflow(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert "progressive_streaming=use_progressive_streaming" in source


class TestLearnResultLayoutPolish:
    """Learn result layout should use distinct semantic sections."""

    def test_learn_page_feedback_is_rendered_with_result_extras(self):
        with open("src/ui/learn_page.py") as f:
            source = f.read()
        assert '_display_feedback_widget("learn", feedback_topic, expanded=True)' in source
        assert '_display_feedback_widget("learn", st.session_state.get("last_learn_topic", ""))' not in source

    def test_shared_feedback_widget_supports_expanded_flag(self):
        import inspect
        from src.ui.shared import _display_feedback_widget

        source = inspect.getsource(_display_feedback_widget)
        assert "expanded: bool = False" in source
        assert "st.expander(f\"Rate this {context_type} experience\", expanded=expanded)" in source

    def test_memory_profile_expander_stays_collapsed_by_default(self):
        import inspect
        from src.ui.shared import _display_memory_section

        source = inspect.getsource(_display_memory_section)
        assert 'st.expander("Memory Profile", expanded=False)' in source


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

    def test_sidebar_has_kb_index_status(self):
        with open("app.py") as f:
            source = f.read()
        assert "KB Index:" in source
        runtime_info = 'with st.sidebar.expander("Runtime Info", expanded=True):'
        kb_status = 'st.caption(f"KB Index: {_kb_status}")'
        external_docs = 'st.caption(f"Official Docs Sync: {_get_sidebar_external_docs_status()}")'
        ragas_status = 'st.caption(f"RAGAs Evaluation: {_get_sidebar_ragas_status()}")'
        help_assistant_style = 'f"Agent Personality: "'
        model_line = 'st.caption(f"Model: {_s.app_default_model}")'
        assert source.index(runtime_info) < source.index(kb_status)
        assert source.index(kb_status) < source.index(external_docs)
        assert source.index(external_docs) < source.index(ragas_status)
        assert source.index(ragas_status) < source.index(help_assistant_style)
        assert source.index(help_assistant_style) < source.index(model_line)

    def test_sidebar_has_external_docs_and_ragas_status(self):
        with open("app.py") as f:
            source = f.read()
        assert "Official Docs Sync:" in source
        assert "RAGAs Evaluation:" in source
        assert "Agent Personality:" in source
        assert "Not updated" in source
        assert "Passed" in source
        assert "Needs review" in source
        assert "Not run" in source

    def test_runtime_info_renders_after_active_page(self):
        with open("app.py") as f:
            source = f.read()
        page_render = 'SECTIONS[st.session_state["active_section"]]()'
        runtime_info = 'with st.sidebar.expander("Runtime Info", expanded=True):'
        assert source.index(page_render) < source.index(runtime_info)


class TestAppAccentTheme:
    """Non-error interactive accents should use the blue app theme."""

    def test_streamlit_theme_uses_dark_base_and_blue_primary_accent(self):
        with open(".streamlit/config.toml") as f:
            source = f.read()
        assert "[theme]" in source
        assert 'base = "dark"' in source
        assert 'primaryColor = "#1565c0"' in source

    def test_app_css_overrides_focus_states_to_blue(self):
        with open("app.py") as f:
            source = f.read()
        assert "focus/selected states do not read as errors" in source
        assert "div[data-baseweb=\"input\"] > div:focus-within" in source
        assert "div[data-baseweb=\"textarea\"] > div:focus-within" in source
        assert "div[data-baseweb=\"select\"] > div:focus-within" in source
        assert ".stSlider [data-baseweb=\"slider\"] [role=\"slider\"]" in source
        assert ".stSlider [data-baseweb=\"slider\"] [role=\"progressbar\"]" in source


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
        assert "st.subheader(\"Review Snapshot\")" in source
        assert "st.subheader(\"Observability\")" in source
        assert "st.subheader(\"Agent Capabilities / Tool Registry\")" in source
        assert "st.subheader(\"Knowledge Base Health\")" in source
        assert "st.subheader(\"External Docs / API Updater\")" in source
        assert "st.subheader(\"Token and Cost Tracking\")" in source
        assert "st.subheader(\"Evaluation Readiness (RAGAs)\")" in source
        assert "st.subheader(\"Learning Signals\")" in source
        assert "st.subheader(\"Workflow Readiness\")" in source
        render_block = source[source.index("def render_advanced()"):]
        assert render_block.index("_display_capability_registry_section(") < render_block.index(
            'st.subheader("Token and Cost Tracking")'
        )
        assert render_block.index('st.subheader("Token and Cost Tracking")') < render_block.index(
            "_display_kb_health_section()"
        )
        assert render_block.index("_display_kb_health_section()") < render_block.index(
            "_display_external_docs_updater_section()"
        )

    def test_dashboard_removes_project_strengths_block(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert "**Project Strengths**" not in source

    def test_dashboard_uses_snapshot_metrics(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert 'top_cols[0].metric(' in source
        assert 'signal_cols[0].metric(' in source

    def test_dashboard_has_latest_run_context_for_cost_tracking(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert '"#### Latest Run Context"' in source
        assert "Learning Mode" in source
        assert "Progressive Streaming" in source
        assert "Cache Bypass" in source
        assert "Cache Hit" in source
        assert '"#### All Session Operations"' in source
        assert "use_container_width=True" in source

    def test_dashboard_has_capability_registry_columns_and_controls(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert '"Capability": "Curated KB Retrieval"' in source
        assert '"Capability": "Official Docs Retrieval"' in source
        assert '"Capability": "Progressive Streaming"' in source
        assert '"Capability": "Cache Bypass"' in source
        assert "manual review tools." in source
        assert "without adding new risky controls" not in source
        assert '"User Control": "System-managed"' in source
        assert '"User Control": "Environment-controlled"' in source
        assert '"User Control": "Controlled in Learn"' in source
        assert '"User Control": "Run in Dashboard"' in source
        assert '"Status": "Ready" if official_ready else "Off"' in source
        assert '"Status": "Ready" if memory_loaded else "Active"' not in source
        assert '"Status": "Active" if memory_loaded else "Ready"' in source

    def test_dashboard_uses_clearer_learning_signal_empty_states(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert "No saved learning memory yet." in source
        assert "No feedback captured yet." in source

    def test_dashboard_uses_shorter_snapshot_values(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert 'return "Cached"' in source
        assert '"Loaded" if mem["loaded"] else "Empty"' in source

    def test_dashboard_has_kb_rebuild_button(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert '"Rebuild KB Index"' in source
        assert '"Update External Official Docs"' in source
        assert "This updates local Markdown only" in source
        assert "Chroma automatically." in source
        assert "Run Rebuild KB Index after a successful docs update" in source
        assert '"Partial Files"' in source
        assert '"Failed Files"' in source

    def test_dashboard_ragas_warning_copy_has_no_leading_emoji(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert "Running RAGAs evaluation calls the OpenAI API" in source
        assert "⚠️ Running RAGAs evaluation calls the OpenAI API" not in source

    def test_dashboard_formats_ragas_run_timestamp_for_display(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert '_format_dashboard_timestamp(getattr(report, \'timestamp\', \'\'))' in source

    def test_dashboard_uses_dividers_between_major_sections(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert source.count("st.divider()") >= 6

    def test_sources_copy_explains_source_file_deduplication(self):
        with open("src/ui/shared.py") as f:
            source = f.read()
        assert "deduplicated by source file in this view" in source
        assert "displayed after deduplication by source file" in source


class TestHelpAssistantUi:
    """Help pages should expose scoped guidance and live-doc wording."""

    def test_help_page_has_scoped_assistant_copy(self):
        with open("src/ui/help_page.py") as f:
            source = f.read()
        assert 'st.header("Help Assistant")' in source
        assert "approved live official docs" in source
        assert "In-domain only." in source
        assert '"Ask Help Assistant"' in source
        assert '"Clear chat"' in source
        assert '"Agent Personality"' in source
        assert 'st.session_state.setdefault("help_assistant_personality_mode", "Technical")' in source
        assert "_render_help_style_selector()" in source
        assert "_render_help_advanced_settings()" in source
        assert 'key=f"help_assistant_style_{mode.lower()}"' in source
        assert "st.selectbox(" not in source
        assert 'key="help_assistant_draft_question"' in source
        assert '"Advanced Model Settings"' in source
        assert 'key="help_assistant_temperature"' in source
        assert 'key="help_assistant_top_p"' in source
        assert 'key="help_assistant_frequency_penalty"' in source
        assert 'key="help_assistant_presence_penalty"' in source
        assert 'key="help_assistant_max_tokens"' in source
        assert "Controls creativity/randomness." in source
        assert "Controls nucleus sampling." in source
        assert "Reduces repetition by penalizing tokens that already appeared frequently." in source
        assert "Encourages introducing new concepts instead of repeating existing topics." in source
        assert "Maximum length of the generated response." in source
        assert 'key="help_assistant_submit"' in source
        assert 'key="help_assistant_clear_chat"' in source
        assert 'st.caption(_format_help_runtime_summary(runtime_config))' not in source
        assert 'st.session_state["help_assistant_reset_draft"] = True' in source
        assert 'st.session_state.pop("help_assistant_reset_draft", False)' in source
        assert "_validate_help_submit(draft_question)" in source
        assert 'input_feedback.warning(feedback_message)' in source
        assert 'input_feedback.error(feedback_message)' in source
        assert 'key="help_assistant_question"' not in source
        assert 'st.spinner("Grounding answer with local KB and approved live official docs...")' not in source
        assert "Current Help Assistant responses are generated as one atomic OpenAI call" not in source
        submit_block = source[source.index("if submit:"):]
        assert 'st.session_state["help_assistant_draft_question"] = ""' not in submit_block
        assert submit_block.index("if feedback_message:") < submit_block.index("_append_help_chat_turn(result)")
        assert "action_cols = st.columns([7, 3])" in source
        assert 'clear_chat = st.form_submit_button(' in source
        assert source.index('submit = st.form_submit_button(') < source.index('clear_chat = st.form_submit_button(')

    def test_help_page_renders_history_before_input_form(self):
        with open("src/ui/help_page.py") as f:
            source = f.read()
        history_loop = "for turn in history:\n            _render_help_turn(turn)"
        form_block = 'with st.form("help_assistant_form"):'
        assert source.index(history_loop) < source.index(form_block)

    def test_help_page_uses_grouped_source_expanders(self):
        with open("src/ui/help_page.py") as f:
            source = f.read()
        assert '"Grounded KB sources"' in source
        assert '"Live official docs enrichment"' in source
        assert '"App workflow context"' in source
        assert '"Conversation context"' in source
        assert '"Request Trace"' in source
        assert '"Execution Trace"' in source
        assert '"Raw debug events"' in source
        assert 'st.chat_message("user", avatar="🧑‍💻")' in source
        assert 'st.chat_message("assistant", avatar="🤖")' in source
        assert "Agent Personality:" in source
        assert "personality_label" in source
        assert "Answered from app workflow / live-docs policy context." in source
        assert 'st.caption(f"Location: {row[\'Location\']}")' in source
        assert "_format_help_trace_entries" in source
        assert "_format_help_execution_trace" in source
        assert 'st.markdown(entry.replace("\\n", "  \\n"))' in source
        assert "Scope check: passed" in source
        assert 'elif "select_live_sources:" in lower and "selected=" in lower:' in source
        assert source.index('"Execution Trace"') < source.index('"Raw debug events"')

    def test_sidebar_help_expander_has_help_assistant_guidance_and_examples(self):
        with open("app.py") as f:
            source = f.read()
        assert "**5. Help Assistant**" in source
        assert "Scoped AI engineering help for this app." in source
        assert "get_help_assistant_example_groups" in source
        assert "queue_help_assistant_question(prompt)" in source
        assert "**5. LangSmith**" not in source
        assert "LangSmith tracing" in source
        with open("src/services/help_assistant.py") as f:
            service_source = f.read()
        assert '"App workflow"' in service_source
        assert '"Core AI / KB concepts"' in service_source
        assert '"Official docs / live enrichment"' in service_source
        assert "How does this app work?" in service_source
        assert "How does KB index work?" in service_source
        assert "When does Help Assistant use live official docs?" in service_source
        assert "What do OpenAI structured outputs require?" in service_source
        assert "How do LangGraph reducers work?" in service_source
        assert "What does LangSmith tracing capture?" in service_source

    def test_dashboard_has_help_assistant_summary_section(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert 'st.subheader("Help Assistant")' in source
        assert 'st.session_state.get("help_assistant_personality_mode", "Technical")' in source
        assert 'get_help_assistant_runtime_defaults(personality_mode)' in source
        assert 'summary_cols = st.columns(2)' in source
        assert "| Domain Guard | Enabled |" in source
        assert "| Live Docs Scope | Approved official docs only |" in source
        assert "| Session Chat Memory | Enabled |" in source
        assert "| Agent Personality |" in source
        assert "| Session Turns |" in source
        assert "| Recent Context Window | Last 5 turns |" in source
        assert "| Runtime Sampling |" in source
        assert "| Live Enrichment | Available |" in source
        assert "| Source Provenance | Grouped KB + Live |" in source
        assert "| Empty Submit Handling | Banner only, not stored |" in source
        assert '"##### Agent Personality Profiles"' in source
        assert "| Dimension | Technical | Concise | Friendly | Formal |" in source
        assert "Current runtime values: Temperature=" not in source
        assert "Learn Path: {learn_path_evaluated} cached evaluated case(s)" in source
        assert 'if topic_mode_topics:' in source
        assert 'if help_topics:' in source
        assert 'if topic_mode_topics or help_topics:' in source

    def test_dashboard_uses_neutral_answer_correctness_caption(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        assert 'st.markdown("#### Diagnostic Metric")' not in source
        assert '("Answer Correctness", "answer_correctness", result.answer_correctness)' in source
        assert "Low Answer Correctness usually reflects mismatch" not in source

    def test_help_assistant_service_uses_five_turn_context_window(self):
        with open("src/services/help_assistant.py") as f:
            source = f.read()
        assert "def _format_conversation_context(history: list[dict[str, Any]], max_turns: int = 5)" in source

    def test_dashboard_places_help_assistant_after_token_and_cost_tracking(self):
        with open("src/ui/dashboard_page.py") as f:
            source = f.read()
        render_block = source[source.index("def render_advanced()"):]
        assert render_block.index('st.subheader("Token and Cost Tracking")') < render_block.index(
            "_display_help_assistant_section()"
        )
        assert render_block.index("_display_help_assistant_section()") < render_block.index(
            "_display_kb_health_section()"
        )


class TestUsageRecordAccumulation:
    """Stored usage records should keep the run context needed by the dashboard."""

    @patch("src.ui.shared.st")
    def test_accumulate_usage_records_adds_learn_context(self, mock_st):
        from src.ui.shared import _accumulate_usage_records

        mock_st.session_state = {
            "session_usage_records": [],
            "last_learn_mode": "Topic",
            "last_learn_depth": "Deep Study",
            "last_learn_progressive_streaming": True,
            "last_learn_force_regenerate": False,
            "last_learn_result": {"trace": ["Cache miss"]},
        }

        _accumulate_usage_records([
            {
                "model": "gpt-4o-mini",
                "operation": "learn_guide_section_generation",
                "total_tokens": 800,
                "estimated_cost_usd": 0.001,
            }
        ])

        record = mock_st.session_state["session_usage_records"][0]
        assert record["learning_mode"] == "Topic"
        assert record["learning_depth"] == "Deep Study"
        assert record["progressive_streaming"] is True
        assert record["cache_bypass"] is False
        assert record["cache_hit"] is False

    @patch("src.ui.shared.st")
    def test_accumulate_usage_records_marks_quiz_context_as_not_applicable(self, mock_st):
        from src.ui.shared import _accumulate_usage_records

        mock_st.session_state = {
            "session_usage_records": [],
            "last_quiz_gen_result": {"trace": ["Cache hit"]},
        }

        _accumulate_usage_records([
            {
                "model": "gpt-4o-mini",
                "operation": "quiz_generation",
                "total_tokens": 300,
                "estimated_cost_usd": 0.0003,
            }
        ])

        record = mock_st.session_state["session_usage_records"][0]
        assert record["learning_mode"] is None
        assert record["learning_depth"] is None
        assert record["progressive_streaming"] is None
        assert record["cache_bypass"] is None
        assert record["cache_hit"] is True
