"""Reusable display helpers for observability, transparency, and error states."""

from __future__ import annotations

from typing import Any


def format_source_display(source: Any) -> dict:
    """Format a Source object into a display-friendly dictionary.

    Returns a dict with keys: title, snippet, relevance, metadata_items.
    """
    title = getattr(source, "title", "") or "Untitled"
    snippet = getattr(source, "content_snippet", "") or ""
    relevance = getattr(source, "relevance_score", 0.0)
    metadata = getattr(source, "metadata", {}) or {}

    meta_items = []
    if metadata.get("topic"):
        meta_items.append(("Topic", metadata["topic"]))
    if metadata.get("filename"):
        meta_items.append(("File", metadata["filename"]))
    if metadata.get("source"):
        meta_items.append(("Source", metadata["source"]))

    return {
        "title": title,
        "snippet": snippet if snippet else "_No preview available._",
        "relevance": relevance,
        "relevance_label": f"{relevance:.1f}" if relevance > 0 else "N/A",
        "metadata_items": meta_items,
    }


def format_sources_summary(sources: list) -> str:
    """Return a short summary string for a list of sources."""
    count = len(sources) if sources else 0
    if count == 0:
        return "No sources used."
    return f"{count} source(s) used."


def format_trace_entry(entry: str) -> str:
    """Format a single trace entry for readable display."""
    return entry


def format_graph_state_summary(result: dict) -> list[dict]:
    """Extract key state fields from a workflow result for debug display.

    Returns a list of {label, value} dicts.
    """
    fields = []

    if result.get("topic"):
        fields.append({"label": "Topic", "value": result["topic"]})

    difficulty = result.get("difficulty")
    if difficulty:
        val = difficulty.value if hasattr(difficulty, "value") else str(difficulty)
        fields.append({"label": "Level / Difficulty", "value": val})

    style = result.get("style")
    if style:
        val = style.value if hasattr(style, "value") else str(style)
        fields.append({"label": "Response Style", "value": val})

    docs = result.get("retrieved_docs", [])
    fields.append({"label": "Sources Retrieved", "value": str(len(docs))})

    attempts = result.get("attempts")
    if attempts is not None:
        fields.append({"label": "Retrieval Attempts", "value": str(attempts)})

    if result.get("query_refined"):
        fields.append({"label": "Query Refined", "value": "Yes"})

    if result.get("memory_profile"):
        fields.append({"label": "Memory Profile", "value": "Loaded"})
    else:
        fields.append({"label": "Memory Profile", "value": "Not available"})

    token_usage = result.get("token_usage", {})
    if token_usage and token_usage.get("total_tokens"):
        fields.append({"label": "Total Tokens", "value": f"{token_usage['total_tokens']:,}"})

    return fields


def format_memory_transparency(memory_profile: dict | None) -> dict:
    """Format memory profile for transparency display.

    Returns a dict with structured fields for UI rendering.
    """
    if not memory_profile:
        return {
            "loaded": False,
            "message": "No memory profile available. Complete quizzes and save results to build your profile.",
        }

    return {
        "loaded": True,
        "recent_topics": memory_profile.get("recent_topics", []),
        "weak_areas": memory_profile.get("recurring_weak_areas", []),
        "average_score": memory_profile.get("average_score"),
        "suggested_focus": memory_profile.get("suggested_focus_topics", []),
        "preferred_style": memory_profile.get("preferred_style"),
    }


def format_error_message(error_type: str) -> dict:
    """Return a user-friendly error message and suggestion for common error types."""
    messages = {
        "no_api_key": {
            "icon": "🔑",
            "title": "OpenAI API Key Missing",
            "message": "No OpenAI API key is configured. The app will use fallback content.",
            "suggestion": "Add your API key to the .env file as OPENAI_API_KEY.",
        },
        "retrieval_failure": {
            "icon": "🔍",
            "title": "Retrieval Failed",
            "message": "Could not retrieve relevant documents from the knowledge base.",
            "suggestion": "Try a different topic or check that documents exist in data/raw/.",
        },
        "no_sources": {
            "icon": "📭",
            "title": "No Sources Found",
            "message": "No relevant sources were found for this topic.",
            "suggestion": "Try a broader topic or add more documents to the knowledge base.",
        },
        "quiz_generation_failure": {
            "icon": "❌",
            "title": "Quiz Generation Failed",
            "message": "Could not generate quiz questions.",
            "suggestion": "Try again or select a different topic.",
        },
        "incomplete_answers": {
            "icon": "⚠️",
            "title": "Incomplete Answers",
            "message": "Some questions were left unanswered. They will be marked as incorrect.",
            "suggestion": "Answer all questions before submitting for best results.",
        },
        "memory_save_failure": {
            "icon": "💾",
            "title": "Memory Save Failed",
            "message": "Could not save the result to learning memory.",
            "suggestion": "Try again. If the issue persists, check disk permissions.",
        },
        "empty_progress": {
            "icon": "📊",
            "title": "No Progress Data",
            "message": "No learning sessions recorded yet.",
            "suggestion": "Complete a quiz and save your result to start tracking progress.",
        },
    }
    return messages.get(error_type, {
        "icon": "⚠️",
        "title": "Something Went Wrong",
        "message": "An unexpected error occurred.",
        "suggestion": "Try again or check the Advanced/Debug section for details.",
    })
