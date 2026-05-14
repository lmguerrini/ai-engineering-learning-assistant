"""LangGraph Quiz workflow — compiles and exposes the quiz graph.

The graph implements a quiz generation and evaluation pipeline:
  load_topic_context → load_user_memory → generate_quiz → validate_quiz
    → (if validation fails → return with error)
    → return validated quiz (generation phase)

  evaluate_answers → extract_weak_areas → create_memory_candidate → return_results
    (evaluation phase, triggered after user submits answers)
"""

from loguru import logger
from langgraph.graph import END, START, StateGraph

from src.graphs.quiz_nodes import (
    create_memory_candidate,
    evaluate_answers,
    extract_weak_areas,
    generate_quiz,
    load_topic_context,
    load_user_memory,
    return_results,
    validate_quiz,
)
from src.graphs.quiz_state import QuizState
from src.schemas import DifficultyLevel, QuizQuestion


def _route_after_topic_context(state: QuizState) -> str:
    """Route to END if topic loading failed, otherwise continue."""
    if state.get("error"):
        return "return_results"
    return "load_user_memory"


# ---------------------------------------------------------------------------
# Generation graph: topic → memory → generate → validate → return
# ---------------------------------------------------------------------------

def build_quiz_generation_graph() -> StateGraph:
    """Build the quiz generation StateGraph (uncompiled)."""
    graph = StateGraph(QuizState)

    graph.add_node("load_topic_context", load_topic_context)
    graph.add_node("load_user_memory", load_user_memory)
    graph.add_node("generate_quiz", generate_quiz)
    graph.add_node("validate_quiz", validate_quiz)
    graph.add_node("return_results", return_results)

    graph.add_edge(START, "load_topic_context")
    graph.add_conditional_edges("load_topic_context", _route_after_topic_context)
    graph.add_edge("load_user_memory", "generate_quiz")
    graph.add_edge("generate_quiz", "validate_quiz")
    graph.add_edge("validate_quiz", "return_results")
    graph.add_edge("return_results", END)

    return graph


# ---------------------------------------------------------------------------
# Evaluation graph: evaluate → extract_weak_areas → memory → return
# ---------------------------------------------------------------------------

def build_quiz_evaluation_graph() -> StateGraph:
    """Build the quiz evaluation StateGraph (uncompiled)."""
    graph = StateGraph(QuizState)

    graph.add_node("evaluate_answers", evaluate_answers)
    graph.add_node("extract_weak_areas", extract_weak_areas)
    graph.add_node("create_memory_candidate", create_memory_candidate)
    graph.add_node("return_results", return_results)

    graph.add_edge(START, "evaluate_answers")
    graph.add_edge("evaluate_answers", "extract_weak_areas")
    graph.add_edge("extract_weak_areas", "create_memory_candidate")
    graph.add_edge("create_memory_candidate", "return_results")
    graph.add_edge("return_results", END)

    return graph


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_quiz_generation(
    topic: str,
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
    num_questions: int = 5,
    study_guide_context: str = "",
    force_regenerate: bool = False,
) -> QuizState:
    """Generate a quiz and return the state with questions.

    This is the main entry point for quiz generation.
    """
    try:
        logger.info("[QuizGenGraph] Starting run — topic='{}', difficulty={}, questions={}", topic, difficulty.value, num_questions)
        app = build_quiz_generation_graph().compile()
        initial_state: QuizState = {
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "study_guide_context": study_guide_context,
            "force_regenerate": force_regenerate,
            "trace": [],
            "token_usage": {},
        }
        result = app.invoke(initial_state)
        q_count = len(result.get("questions") or [])
        logger.info("[QuizGenGraph] Run complete — topic='{}', questions_generated={}", topic, q_count)
        return result
    except Exception as e:
        logger.error("[QuizGenGraph] Run failed — topic='{}', error={}", topic, e)
        return {
            "topic": topic,
            "difficulty": difficulty,
            "questions": [],
            "error": f"Quiz generation failed: {e}",
            "trace": [f"run_quiz_generation: exception — {e}"],
            "token_usage": {},
        }


def run_quiz_evaluation(
    topic: str,
    questions: list[QuizQuestion],
    user_answers: list[str],
) -> QuizState:
    """Evaluate user answers and return the state with results.

    This is the main entry point for answer evaluation.
    """
    try:
        logger.info("[QuizEvalGraph] Starting run — topic='{}', answers={}", topic, len(user_answers))
        app = build_quiz_evaluation_graph().compile()
        initial_state: QuizState = {
            "topic": topic,
            "questions": questions,
            "user_answers": user_answers,
            "trace": [],
            "token_usage": {},
        }
        result = app.invoke(initial_state)
        score = result.get("score", 0.0)
        logger.info("[QuizEvalGraph] Run complete — topic='{}', score={}", topic, score)
        return result
    except Exception as e:
        logger.error("[QuizEvalGraph] Run failed — topic='{}', error={}", topic, e)
        return {
            "topic": topic,
            "questions": questions,
            "user_answers": user_answers,
            "score": 0.0,
            "error": f"Quiz evaluation failed: {e}",
            "trace": [f"run_quiz_evaluation: exception — {e}"],
            "token_usage": {},
        }
