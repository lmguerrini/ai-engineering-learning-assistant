"""Tests for Learn UX quality improvements."""

import pytest

from src.schemas import ResponseStyle, DifficultyLevel, Source, StudyGuide
from src.ui.display_helpers import (
    _sanitize_snippet,
    deduplicate_sources,
    downgrade_headings,
    format_source_display,
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
        assert _sanitize_snippet("# Big Title\nContent") == "Big Title\nContent"

    def test_strips_h2(self):
        assert _sanitize_snippet("## Section\nText") == "Section\nText"

    def test_strips_h3(self):
        assert _sanitize_snippet("### Sub\nText") == "Sub\nText"

    def test_strips_multiple_headings(self):
        text = "# Title\n\n## Section\n\nContent"
        result = _sanitize_snippet(text)
        assert "#" not in result

    def test_collapses_blank_lines(self):
        text = "A line\n\n\n\n\nB line"
        result = _sanitize_snippet(text)
        assert "\n\n\n" not in result

    def test_empty_string(self):
        assert _sanitize_snippet("") == ""

    def test_no_headings_unchanged(self):
        text = "Just plain text here."
        assert _sanitize_snippet(text) == text

    def test_removes_single_letter_line(self):
        text = "Some text\nO\nMore text"
        result = _sanitize_snippet(text)
        assert "\nO\n" not in result

    def test_removes_stray_single_letter_mid_text(self):
        text = "word O word"
        result = _sanitize_snippet(text)
        assert " O " not in result

    def test_collapses_multiple_spaces(self):
        text = "word   word"
        result = _sanitize_snippet(text)
        assert "   " not in result


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
        src = Source(title="Test", content_snippet="# Big Heading\nContent")
        info = format_source_display(src)
        assert not info["snippet"].startswith("#")

    def test_format_source_display_shows_source_type(self):
        src = Source(title="Test", metadata={"source_type": "official_docs"})
        info = format_source_display(src)
        types = [k for k, v in info["metadata_items"] if k == "Type"]
        assert len(types) == 1


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


class TestMemoryEmptyState:
    """Memory empty state should provide clear explanation."""

    def test_memory_empty_message(self):
        mem = format_memory_transparency(None)
        assert not mem["loaded"]
        assert "automatically" in mem["message"].lower()
        assert "quiz" in mem["message"].lower()

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
