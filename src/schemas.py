"""Core Pydantic schemas for the AI Engineering Learning Assistant."""

from enum import Enum

from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    """Difficulty level for learning content and quizzes."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ResponseStyle(str, Enum):
    """Response style preference."""

    CONCISE = "concise"
    DETAILED = "detailed"
    EXAMPLES_HEAVY = "examples_heavy"


class TopicRequest(BaseModel):
    """User request to study a topic."""

    topic: str = Field(..., min_length=1, description="AI Engineering topic to study")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)
    style: ResponseStyle = Field(default=ResponseStyle.DETAILED)


class Source(BaseModel):
    """A single source used in a study guide."""

    title: str = ""
    content_snippet: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


class StudyGuide(BaseModel):
    """Structured study guide output from the Learn workflow."""

    topic: str
    difficulty: DifficultyLevel
    summary: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    detailed_notes: str = ""
    sources: list[Source] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    """A single quiz question."""

    question: str
    options: list[str] = Field(default_factory=list, min_length=2)
    correct_answer: str = ""
    explanation: str = ""
    concept: str = ""


class QuizResult(BaseModel):
    """Result of a quiz attempt."""

    topic: str
    total_questions: int = 0
    correct_count: int = 0
    score_percent: float = 0.0
    weak_areas: list[str] = Field(default_factory=list)
    feedback: str = ""


class UserProgress(BaseModel):
    """Summary of user learning progress."""

    topics_studied: list[str] = Field(default_factory=list)
    quiz_scores: dict[str, float] = Field(default_factory=dict)
    weak_areas: list[str] = Field(default_factory=list)
    preferred_difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    preferred_style: ResponseStyle = ResponseStyle.DETAILED
