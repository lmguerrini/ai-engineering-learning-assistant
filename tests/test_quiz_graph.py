"""Tests for the Quiz LangGraph workflow."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.graphs.quiz_nodes import (
    create_memory_candidate_placeholder,
    evaluate_answers,
    extract_weak_areas,
    generate_quiz,
    load_topic_context,
    load_user_memory_placeholder,
    return_results,
    validate_quiz,
)
from src.graphs.quiz_graph import (
    build_quiz_evaluation_graph,
    build_quiz_generation_graph,
    run_quiz_evaluation,
    run_quiz_generation,
)
from src.schemas import DifficultyLevel, QuizQuestion, QuizResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_questions(n: int = 3) -> list[QuizQuestion]:
    """Create n valid quiz questions for testing."""
    return [
        QuizQuestion(
            question=f"Question {i + 1}?",
            options=[f"A) opt_a_{i}", f"B) opt_b_{i}", f"C) opt_c_{i}", f"D) opt_d_{i}"],
            correct_answer=f"A) opt_a_{i}",
            explanation=f"Explanation for question {i + 1}.",
        )
        for i in range(n)
    ]


# ===================================================================
# load_topic_context
# ===================================================================

class TestLoadTopicContext:
    def test_valid_topic(self):
        state = {"topic": "AI Agents", "trace": []}
        result = load_topic_context(state)
        assert result["error"] is None
        assert result["topic"] == "AI Agents"
        assert "load_topic_context: started" in result["trace"]

    def test_empty_topic(self):
        state = {"topic": "", "trace": []}
        result = load_topic_context(state)
        assert result["error"] is not None
        assert "no topic" in result["error"].lower() or "select a topic" in result["error"].lower()

    def test_missing_topic(self):
        state = {"trace": []}
        result = load_topic_context(state)
        assert result["error"] is not None

    def test_with_study_guide_context(self):
        state = {"topic": "LangGraph", "study_guide_context": "Some guide content", "trace": []}
        result = load_topic_context(state)
        assert result["study_guide_context"] == "Some guide content"
        assert result["error"] is None

    def test_without_study_guide_context(self):
        state = {"topic": "LangGraph", "trace": []}
        result = load_topic_context(state)
        assert "AI Engineering topic: LangGraph" in result["study_guide_context"]

    def test_defaults(self):
        state = {"topic": "AI Agents", "trace": []}
        result = load_topic_context(state)
        assert result["difficulty"] == DifficultyLevel.INTERMEDIATE
        assert result["num_questions"] == 5


# ===================================================================
# load_user_memory_placeholder
# ===================================================================

class TestLoadUserMemoryPlaceholder:
    def test_returns_empty_memory(self):
        state = {"trace": []}
        result = load_user_memory_placeholder(state)
        assert result["user_memory"] == {}
        assert any("placeholder" in t for t in result["trace"])


# ===================================================================
# generate_quiz
# ===================================================================

class TestGenerateQuiz:
    @patch("src.graphs.quiz_nodes.get_settings")
    def test_fallback_when_no_api_key(self, mock_settings):
        mock_settings.return_value = MagicMock(openai_api_key="")
        state = {"topic": "AI Agents", "trace": [], "token_usage": {}}
        result = generate_quiz(state)
        assert len(result["questions"]) >= 1
        assert any("fallback" in t for t in result["trace"])

    @patch("src.graphs.quiz_nodes.get_settings")
    @patch("src.graphs.quiz_nodes.OpenAI")
    def test_successful_generation(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(openai_api_key="sk-test", app_default_model="gpt-4o-mini")
        questions_data = {
            "questions": [
                {
                    "question": "What is an AI agent?",
                    "options": ["A) A program", "B) A rock", "C) A fish", "D) A cloud"],
                    "correct_answer": "A) A program",
                    "explanation": "An AI agent is a program.",
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(questions_data)))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_openai_cls.return_value.chat.completions.create.return_value = mock_response

        state = {"topic": "AI Agents", "num_questions": 1, "trace": [], "token_usage": {}}
        result = generate_quiz(state)
        assert len(result["questions"]) == 1
        assert result["questions"][0].question == "What is an AI agent?"
        assert result["token_usage"]["total_tokens"] == 30

    @patch("src.graphs.quiz_nodes.get_settings")
    @patch("src.graphs.quiz_nodes.OpenAI")
    def test_malformed_json_fallback(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(openai_api_key="sk-test", app_default_model="gpt-4o-mini")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json {{{"))]
        mock_response.usage = None
        mock_openai_cls.return_value.chat.completions.create.return_value = mock_response

        state = {"topic": "AI Agents", "trace": [], "token_usage": {}}
        result = generate_quiz(state)
        assert len(result["questions"]) >= 1
        assert any("malformed" in t.lower() or "fallback" in t.lower() for t in result["trace"])

    @patch("src.graphs.quiz_nodes.get_settings")
    @patch("src.graphs.quiz_nodes.OpenAI")
    def test_llm_exception_fallback(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(openai_api_key="sk-test", app_default_model="gpt-4o-mini")
        mock_openai_cls.return_value.chat.completions.create.side_effect = RuntimeError("API down")

        state = {"topic": "AI Agents", "trace": [], "token_usage": {}}
        result = generate_quiz(state)
        assert len(result["questions"]) >= 1
        assert any("error" in t.lower() or "fallback" in t.lower() for t in result["trace"])


# ===================================================================
# validate_quiz
# ===================================================================

class TestValidateQuiz:
    def test_valid_questions(self):
        questions = _make_questions(3)
        state = {"questions": questions, "num_questions": 3, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is True
        assert result["validation_errors"] == []

    def test_wrong_count(self):
        questions = _make_questions(2)
        state = {"questions": questions, "num_questions": 5, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is False
        assert any("Expected 5" in e for e in result["validation_errors"])

    def test_no_questions(self):
        state = {"questions": [], "num_questions": 3, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is False
        assert any("No questions" in e for e in result["validation_errors"])

    def test_missing_correct_answer(self):
        q = QuizQuestion(
            question="Test?",
            options=["A) a", "B) b", "C) c", "D) d"],
            correct_answer="",
            explanation="Some explanation",
        )
        state = {"questions": [q], "num_questions": 1, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is False
        assert any("missing correct_answer" in e for e in result["validation_errors"])

    def test_correct_answer_not_in_options(self):
        q = QuizQuestion(
            question="Test?",
            options=["A) a", "B) b", "C) c", "D) d"],
            correct_answer="E) not there",
            explanation="Explanation",
        )
        state = {"questions": [q], "num_questions": 1, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is False
        assert any("not in options" in e for e in result["validation_errors"])

    def test_missing_explanation(self):
        q = QuizQuestion(
            question="Test?",
            options=["A) a", "B) b", "C) c", "D) d"],
            correct_answer="A) a",
            explanation="",
        )
        state = {"questions": [q], "num_questions": 1, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is False
        assert any("missing explanation" in e for e in result["validation_errors"])

    def test_too_few_options(self):
        q = QuizQuestion(
            question="Test?",
            options=["A) a", "B) b"],
            correct_answer="A) a",
            explanation="Explanation",
        )
        state = {"questions": [q], "num_questions": 1, "trace": []}
        result = validate_quiz(state)
        assert result["validation_passed"] is False
        assert any("options" in e.lower() for e in result["validation_errors"])


# ===================================================================
# evaluate_answers
# ===================================================================

class TestEvaluateAnswers:
    def test_all_correct(self):
        questions = _make_questions(3)
        answers = [q.correct_answer for q in questions]
        state = {"topic": "AI", "questions": questions, "user_answers": answers, "trace": []}
        result = evaluate_answers(state)
        assert result["score"] == 100.0
        assert all(result["per_question_correct"])
        assert result["quiz_result"].correct_count == 3

    def test_all_wrong(self):
        questions = _make_questions(3)
        answers = ["wrong"] * 3
        state = {"topic": "AI", "questions": questions, "user_answers": answers, "trace": []}
        result = evaluate_answers(state)
        assert result["score"] == 0.0
        assert not any(result["per_question_correct"])
        assert result["quiz_result"].correct_count == 0

    def test_partial(self):
        questions = _make_questions(4)
        answers = [questions[0].correct_answer, "wrong", questions[2].correct_answer, "wrong"]
        state = {"topic": "AI", "questions": questions, "user_answers": answers, "trace": []}
        result = evaluate_answers(state)
        assert result["score"] == 50.0
        assert result["per_question_correct"] == [True, False, True, False]

    def test_incomplete_answers_padded(self):
        questions = _make_questions(3)
        answers = [questions[0].correct_answer]  # only 1 answer for 3 questions
        state = {"topic": "AI", "questions": questions, "user_answers": answers, "trace": []}
        result = evaluate_answers(state)
        assert len(result["per_question_correct"]) == 3
        assert result["per_question_correct"][0] is True
        assert result["per_question_correct"][1] is False
        assert result["per_question_correct"][2] is False

    def test_no_questions(self):
        state = {"topic": "AI", "questions": [], "user_answers": [], "trace": []}
        result = evaluate_answers(state)
        assert result["score"] == 0.0
        assert result["error"] is not None

    def test_explanations_present(self):
        questions = _make_questions(2)
        answers = [questions[0].correct_answer, "wrong"]
        state = {"topic": "AI", "questions": questions, "user_answers": answers, "trace": []}
        result = evaluate_answers(state)
        assert len(result["explanations"]) == 2
        assert "Correct" in result["explanations"][0]
        assert "Incorrect" in result["explanations"][1]


# ===================================================================
# extract_weak_areas
# ===================================================================

class TestExtractWeakAreas:
    def test_identifies_wrong_questions(self):
        questions = _make_questions(3)
        state = {
            "questions": questions,
            "per_question_correct": [True, False, False],
            "score": 33.3,
            "quiz_result": QuizResult(topic="AI", total_questions=3, correct_count=1, score_percent=33.3),
            "trace": [],
        }
        result = extract_weak_areas(state)
        assert len(result["weak_areas"]) == 2
        assert result["quiz_result"].weak_areas == result["weak_areas"]

    def test_no_weak_areas_when_all_correct(self):
        questions = _make_questions(2)
        state = {
            "questions": questions,
            "per_question_correct": [True, True],
            "score": 100.0,
            "quiz_result": QuizResult(topic="AI", total_questions=2, correct_count=2, score_percent=100.0),
            "trace": [],
        }
        result = extract_weak_areas(state)
        assert result["weak_areas"] == []

    def test_high_score_next_steps(self):
        state = {
            "questions": _make_questions(1),
            "per_question_correct": [True],
            "score": 100.0,
            "quiz_result": QuizResult(topic="AI", total_questions=1, correct_count=1, score_percent=100.0),
            "trace": [],
        }
        result = extract_weak_areas(state)
        assert any("harder" in s.lower() or "advancing" in s.lower() for s in result["suggested_next_steps"])

    def test_low_score_next_steps(self):
        state = {
            "questions": _make_questions(3),
            "per_question_correct": [False, False, False],
            "score": 0.0,
            "quiz_result": QuizResult(topic="AI", total_questions=3, correct_count=0, score_percent=0.0),
            "trace": [],
        }
        result = extract_weak_areas(state)
        assert any("study" in s.lower() for s in result["suggested_next_steps"])

    def test_medium_score_next_steps(self):
        state = {
            "questions": _make_questions(2),
            "per_question_correct": [True, False],
            "score": 50.0,
            "quiz_result": QuizResult(topic="AI", total_questions=2, correct_count=1, score_percent=50.0),
            "trace": [],
        }
        result = extract_weak_areas(state)
        assert any("review" in s.lower() or "weak" in s.lower() for s in result["suggested_next_steps"])


# ===================================================================
# Placeholder nodes
# ===================================================================

class TestPlaceholderNodes:
    def test_create_memory_candidate_placeholder(self):
        state = {"trace": []}
        result = create_memory_candidate_placeholder(state)
        assert any("placeholder" in t for t in result["trace"])

    def test_return_results(self):
        state = {"trace": []}
        result = return_results(state)
        assert any("done" in t for t in result["trace"])


# ===================================================================
# Graph compilation
# ===================================================================

class TestGraphCompilation:
    def test_generation_graph_compiles(self):
        graph = build_quiz_generation_graph()
        app = graph.compile()
        assert app is not None

    def test_evaluation_graph_compiles(self):
        graph = build_quiz_evaluation_graph()
        app = graph.compile()
        assert app is not None


# ===================================================================
# Graph routing
# ===================================================================

class TestGraphRouting:
    @patch("src.graphs.quiz_nodes.get_settings")
    def test_empty_topic_returns_error(self, mock_settings):
        """Empty topic should route through load_topic_context → return_results with error."""
        mock_settings.return_value = MagicMock(openai_api_key="")
        result = run_quiz_generation(topic="", difficulty=DifficultyLevel.BEGINNER)
        assert result.get("error") is not None
        assert "topic" in result["error"].lower()

    @patch("src.graphs.quiz_nodes.get_settings")
    def test_generation_produces_questions(self, mock_settings):
        """Valid topic with no API key should produce fallback questions."""
        mock_settings.return_value = MagicMock(openai_api_key="")
        result = run_quiz_generation(topic="AI Agents", num_questions=5)
        questions = result.get("questions", [])
        assert len(questions) >= 1
        trace = result.get("trace", [])
        assert any("fallback" in t for t in trace)

    def test_evaluation_graph_scores_correctly(self):
        """Evaluation graph should compute correct score for given answers."""
        questions = _make_questions(3)
        answers = [questions[0].correct_answer, "wrong", questions[2].correct_answer]
        result = run_quiz_evaluation(topic="AI", questions=questions, user_answers=answers)
        assert result.get("score") == pytest.approx(66.7, abs=0.1)
        assert len(result.get("weak_areas", [])) == 1


# ===================================================================
# Fallback behavior
# ===================================================================

class TestFallbackBehavior:
    @patch("src.graphs.quiz_nodes.get_settings")
    def test_no_api_key_still_returns_quiz(self, mock_settings):
        mock_settings.return_value = MagicMock(openai_api_key="")
        result = run_quiz_generation(topic="LangGraph")
        assert result.get("questions") is not None
        assert len(result["questions"]) >= 1

    def test_evaluation_with_empty_answers(self):
        questions = _make_questions(3)
        result = run_quiz_evaluation(topic="AI", questions=questions, user_answers=[])
        assert result.get("score") == 0.0
        assert len(result.get("per_question_correct", [])) == 3
        assert all(c is False for c in result["per_question_correct"])

    def test_evaluation_with_no_questions(self):
        result = run_quiz_evaluation(topic="AI", questions=[], user_answers=[])
        assert result.get("error") is not None
