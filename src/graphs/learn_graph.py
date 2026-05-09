"""LangGraph Learn workflow — compiles and exposes the learn graph.

The graph implements Agentic RAG:
  validate → load_memory → retrieve → assess quality
    → (if insufficient) refine & re-retrieve
    → generate study guide → quality check → persist → return
"""

from collections.abc import Callable

from loguru import logger
from langgraph.graph import END, START, StateGraph

from src.graphs.learn_nodes import (
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
from src.graphs.learn_state import LearningState
from src.schemas import DifficultyLevel, ResponseStyle, StudyGuide


def _route_after_validation(state: LearningState) -> str:
    """Route to END if validation failed, otherwise continue."""
    if state.get("error"):
        return "error_path"
    return "success_path"


def _route_after_assessment(state: LearningState) -> str:
    """Route based on source quality: refine or proceed to generation."""
    if state.get("source_quality_ok"):
        return "generate_path"
    if state.get("attempts", 0) >= 2:
        return "generate_path"
    return "refine_path"


def build_learn_graph() -> StateGraph:
    """Build and return the (uncompiled) Learn StateGraph."""
    graph = StateGraph(LearningState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("load_user_memory", load_user_memory)
    graph.add_node("retrieve_sources", retrieve_sources)
    graph.add_node("assess_source_quality", assess_source_quality)
    graph.add_node("refine_query_if_needed", refine_query_if_needed)
    graph.add_node("generate_study_guide", generate_study_guide)
    graph.add_node("quality_check", quality_check)
    graph.add_node("persist_learning_event_placeholder", persist_learning_event_placeholder)
    graph.add_node("return_output", return_output)

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input",
        _route_after_validation,
        {"error_path": "return_output", "success_path": "load_user_memory"},
    )
    graph.add_edge("load_user_memory", "retrieve_sources")
    graph.add_edge("retrieve_sources", "assess_source_quality")
    graph.add_conditional_edges(
        "assess_source_quality",
        _route_after_assessment,
        {"generate_path": "generate_study_guide", "refine_path": "refine_query_if_needed"},
    )
    graph.add_edge("refine_query_if_needed", "retrieve_sources")
    graph.add_edge("generate_study_guide", "quality_check")
    graph.add_edge("quality_check", "persist_learning_event_placeholder")
    graph.add_edge("persist_learning_event_placeholder", "return_output")
    graph.add_edge("return_output", END)

    return graph


def compile_learn_graph():
    """Compile the Learn graph into a runnable."""
    return build_learn_graph().compile()


def run_learn_workflow(
    topic: str,
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
    style: ResponseStyle = ResponseStyle.DETAILED,
    force_regenerate: bool = False,
    progressive_streaming: bool = True,
    progress_callback: Callable[[StudyGuide], None] | None = None,
) -> LearningState:
    """Run the full Learn workflow and return the final state.

    This is the main entry point for the Learn feature.
    Handles all exceptions gracefully and returns a state with error info.
    """
    try:
        logger.info("[LearnGraph] Starting run — topic='{}', difficulty={}, style={}", topic, difficulty.value, style.value)
        app = compile_learn_graph()
        initial_state: LearningState = {
            "topic": topic,
            "difficulty": difficulty,
            "style": style,
            "force_regenerate": force_regenerate,
            "progressive_streaming": progressive_streaming,
            "trace": [],
            "token_usage": {},
        }
        if progress_callback is not None:
            initial_state["progress_callback"] = progress_callback
        result = app.invoke(
            initial_state,
            config={"run_name": f"learn_workflow:{topic}"},
        )
        sources_count = len(result.get("retrieved_docs") or [])
        fallback_used = not result.get("source_quality_ok", True)
        logger.info(
            "[LearnGraph] Run complete — topic='{}', sources={}, fallback={}",
            topic, sources_count, fallback_used,
        )
        return result
    except Exception as e:
        logger.error("[LearnGraph] Run failed — topic='{}', error={}", topic, e)
        return {
            "topic": topic,
            "difficulty": difficulty,
            "style": style,
            "error": f"Learn workflow failed: {e}",
            "trace": [f"run_learn_workflow: exception — {e}"],
            "study_guide": None,
            "token_usage": {},
        }
