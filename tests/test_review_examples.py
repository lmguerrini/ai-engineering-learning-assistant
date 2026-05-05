"""Tests for demo/review examples module."""

from src.demo.review_examples import (
    DemoExample,
    DEMO_EXAMPLES,
    get_demo_examples,
    get_demo_titles,
    get_demo_by_title,
)


class TestDemoExamples:
    def test_examples_not_empty(self):
        assert len(DEMO_EXAMPLES) >= 5

    def test_all_examples_have_required_fields(self):
        for ex in DEMO_EXAMPLES:
            assert ex.title, f"Missing title: {ex}"
            assert ex.topic, f"Missing topic: {ex}"
            assert ex.description, f"Missing description: {ex}"
            assert ex.features_exercised, f"Missing features: {ex}"
            assert ex.difficulty in ("beginner", "intermediate", "advanced")
            assert ex.response_style

    def test_titles_are_unique(self):
        titles = [ex.title for ex in DEMO_EXAMPLES]
        assert len(titles) == len(set(titles))

    def test_topics_are_non_trivial(self):
        for ex in DEMO_EXAMPLES:
            assert len(ex.topic) > 10, f"Topic too short: {ex.topic}"

    def test_get_demo_examples_returns_copy(self):
        examples = get_demo_examples()
        assert examples == list(DEMO_EXAMPLES)
        assert examples is not DEMO_EXAMPLES

    def test_get_demo_titles(self):
        titles = get_demo_titles()
        assert len(titles) == len(DEMO_EXAMPLES)
        assert all(isinstance(t, str) for t in titles)
        assert all(len(t) > 0 for t in titles)

    def test_get_demo_by_title_found(self):
        ex = get_demo_by_title("Agentic RAG")
        assert ex is not None
        assert ex.title == "Agentic RAG"
        assert "RAG" in ex.topic

    def test_get_demo_by_title_not_found(self):
        assert get_demo_by_title("Nonexistent Topic") is None

    def test_expected_titles_present(self):
        titles = get_demo_titles()
        assert "LangGraph Conditional Routing" in titles
        assert "Agentic RAG" in titles
        assert "Long-Term Memory and HITL" in titles
        assert "RAG Evaluation" in titles
        assert "Official Docs Fallback" in titles

    def test_features_exercised_are_strings(self):
        for ex in DEMO_EXAMPLES:
            for feat in ex.features_exercised:
                assert isinstance(feat, str)
                assert len(feat) > 0
