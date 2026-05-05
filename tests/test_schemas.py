"""Tests for core Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    DifficultyLevel,
    QuizQuestion,
    QuizResult,
    ResponseStyle,
    Source,
    StudyGuide,
    TopicRequest,
    UserProgress,
)


def test_topic_request_valid():
    req = TopicRequest(topic="LangGraph")
    assert req.topic == "LangGraph"
    assert req.difficulty == DifficultyLevel.INTERMEDIATE
    assert req.style == ResponseStyle.DETAILED


def test_topic_request_empty_topic_rejected():
    with pytest.raises(ValidationError):
        TopicRequest(topic="")


def test_source_defaults():
    src = Source()
    assert src.title == ""
    assert src.relevance_score == 0.0


def test_study_guide_minimal():
    guide = StudyGuide(topic="Agents", difficulty=DifficultyLevel.BEGINNER)
    assert guide.topic == "Agents"
    assert guide.key_concepts == []
    assert guide.sources == []


def test_quiz_question_requires_min_options():
    with pytest.raises(ValidationError):
        QuizQuestion(question="What is an agent?", options=["A"])


def test_quiz_result_defaults():
    result = QuizResult(topic="RAG")
    assert result.score_percent == 0.0
    assert result.weak_areas == []


def test_user_progress_defaults():
    progress = UserProgress()
    assert progress.topics_studied == []
    assert progress.preferred_difficulty == DifficultyLevel.INTERMEDIATE
