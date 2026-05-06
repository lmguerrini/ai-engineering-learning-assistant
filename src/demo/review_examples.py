"""Curated demo topics and review examples for the AI Engineering Learning Assistant.

Provides a set of ready-to-use demo queries that exercise key app features:
Learn workflow, Quiz workflow, Agentic RAG, KB retrieval, and official docs fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoExample:
    """A single demo topic/query for review or demonstration."""

    title: str
    topic: str
    description: str
    features_exercised: list[str]
    difficulty: str = "intermediate"
    response_style: str = "detailed"
    learning_mode: str = "Topic"


DEMO_EXAMPLES: list[DemoExample] = [
    DemoExample(
        title="LangGraph Conditional Routing",
        topic="LangGraph conditional routing and state-based graph orchestration",
        description=(
            "Demonstrates how the Learn workflow retrieves and explains "
            "LangGraph's conditional edges, state management, and routing patterns."
        ),
        features_exercised=[
            "Learn workflow",
            "Curated KB retrieval",
            "Agentic RAG source assessment",
        ],
        difficulty="intermediate",
        learning_mode="Topic",
    ),
    DemoExample(
        title="Agentic RAG",
        topic="Agentic RAG with source quality assessment and query refinement",
        description=(
            "Showcases the app's core Agentic RAG behavior: retrieve sources, "
            "assess quality, optionally refine the query, and retrieve again."
        ),
        features_exercised=[
            "Learn workflow",
            "Source quality assessment",
            "Query refinement loop",
            "Multi-attempt retrieval",
        ],
        difficulty="advanced",
        learning_mode="Topic",
    ),
    DemoExample(
        title="Long-Term Memory and HITL",
        topic="Long-term memory for AI agents and human-in-the-loop patterns",
        description=(
            "Tests the Learn workflow on memory/HITL topics and demonstrates "
            "how quiz results can be saved or skipped via HITL approval."
        ),
        features_exercised=[
            "Learn workflow",
            "Quiz workflow",
            "Memory personalization",
            "HITL save/skip",
        ],
        difficulty="intermediate",
        learning_mode="Topic",
    ),
    DemoExample(
        title="RAG Evaluation",
        topic="RAG evaluation metrics and retrieval quality assessment",
        description=(
            "Exercises retrieval of RAG evaluation concepts from both "
            "curated KB and official docs (RAGAs reference)."
        ),
        features_exercised=[
            "Learn workflow",
            "Curated KB retrieval",
            "Official docs fallback",
        ],
        difficulty="advanced",
        learning_mode="Topic",
    ),
    DemoExample(
        title="Official Docs Fallback",
        topic="Pydantic validation and settings configuration",
        description=(
            "Tests a topic where curated KB may have limited coverage, "
            "triggering the official docs fallback enrichment from Pydantic docs."
        ),
        features_exercised=[
            "Learn workflow",
            "Official docs fallback",
            "Source metadata transparency",
        ],
        difficulty="beginner",
        learning_mode="Topic",
    ),
]


def get_demo_examples() -> list[DemoExample]:
    """Return all curated demo examples."""
    return list(DEMO_EXAMPLES)


def get_demo_titles() -> list[str]:
    """Return titles of all demo examples for UI selectors."""
    return [ex.title for ex in DEMO_EXAMPLES]


def get_demo_by_title(title: str) -> DemoExample | None:
    """Look up a demo example by its title.

    Args:
        title: Exact title to search for.

    Returns:
        Matching DemoExample or None.
    """
    for ex in DEMO_EXAMPLES:
        if ex.title == title:
            return ex
    return None
