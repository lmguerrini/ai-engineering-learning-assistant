"""Reusable display helpers for observability, transparency, and error states."""

from __future__ import annotations

import re
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
    for key, label in [
        ("topic", "Topic"),
        ("sprint", "Sprint"),
        ("part", "Part"),
        ("tags", "Tags"),
        ("filename", "File"),
        ("source_type", "Type"),
        ("source", "Source"),
    ]:
        val = metadata.get(key)
        if val:
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            meta_items.append((label, str(val)))

    clean = _sanitize_snippet(snippet) if snippet else ""

    # Detect placeholder/static relevance (hardcoded 0.5 everywhere)
    is_real_relevance = relevance > 0 and relevance != 0.5

    return {
        "title": title,
        "snippet": clean if clean else "_No clean preview available._",
        "relevance": relevance if is_real_relevance else 0.0,
        "relevance_label": f"{relevance:.1f}" if is_real_relevance else "",
        "metadata_items": meta_items,
    }


def _sanitize_snippet(text: str) -> str:
    """Strip markdown headings, short broken fragments, and collapse whitespace.

    Aggressively removes chunk artifacts (lone letters, partial words,
    markdown noise) so that only readable, professional text remains.
    Returns an empty string when nothing useful survives cleanup.
    """
    # Remove markdown heading markers (# ## ### etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown link/image reference artifacts like [](url) or ![]()
    text = re.sub(r"!?\[\]\([^)]*\)", "", text)
    # Remove stray single-letter lines (e.g. lone "O", "N")
    text = re.sub(r"^[A-Z]\n", "", text, flags=re.MULTILINE)
    # Remove short (1-3 char) uppercase-only lines — chunk artifacts like "ND", "IO"
    text = re.sub(r"^[A-Z]{1,3}\s*$", "", text, flags=re.MULTILINE)
    # Remove stray single letters surrounded by whitespace mid-text
    text = re.sub(r"(?<=\s)[A-Z](?=\s)", "", text)
    # Strip leading broken fragment (line that doesn't start with a capital
    # letter or bullet — likely a truncated tail from chunking)
    text = re.sub(r"^[a-z][^\n]{0,40}\n", "", text, count=1)
    # --- Fix mid-sentence starts ---
    # If text starts with a lowercase word or partial sentence, skip to the
    # first sentence boundary (period/exclamation/question followed by space
    # and an uppercase letter) to avoid truncated-looking previews.
    text = _skip_to_sentence_start(text)
    # Strip trailing broken fragment (incomplete sentence not ending in punctuation)
    stripped = text.rstrip()
    if stripped and stripped[-1] not in '.!?:;)"\'' and len(stripped) > 20:
        # Cut back to the last sentence-ending punctuation
        last_stop = max(stripped.rfind('. '), stripped.rfind('? '),
                        stripped.rfind('! '), stripped.rfind('.\n'),
                        stripped.rfind('.'))
        if last_stop > len(stripped) // 4:
            text = stripped[: last_stop + 1]
    # Remove lines that are only punctuation / special chars
    text = re.sub(r"^[^\w\s]{1,5}\s*$", "", text, flags=re.MULTILINE)
    # Remove bare URLs on their own line
    text = re.sub(r"^https?://\S+\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    text = text.strip()
    # If only very short junk remains, treat as empty
    if len(text) < 8 or not re.search(r"[a-zA-Z]{2,}", text):
        return ""
    return text


def _skip_to_sentence_start(text: str) -> str:
    """If *text* begins mid-sentence, advance to the next clean sentence start.

    A mid-sentence start is detected when the first non-whitespace character
    is lowercase.  We then look for the first sentence boundary (`.` / `!` / `?`
    followed by whitespace and an uppercase letter) and return from there.
    If no boundary is found within the first 200 chars we return the original
    text unchanged — it may still be usable.
    """
    stripped = text.lstrip()
    if not stripped or stripped[0].isupper() or stripped[0] in '-•*–—0123456789':
        return text  # already starts cleanly
    # Look for sentence boundary in the first 200 chars
    m = re.search(r'[.!?]\s+([A-Z])', stripped[:200])
    if m:
        return stripped[m.start(1):]
    # No boundary found — return original
    return text


def deduplicate_sources(sources: list) -> list:
    """Deduplicate sources by filename, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list = []
    for src in sources:
        meta = getattr(src, "metadata", {}) or {}
        key = meta.get("filename", "") or getattr(src, "title", "")
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def format_sources_summary(sources: list) -> str:
    """Return a short summary string for a list of sources."""
    count = len(sources) if sources else 0
    if count == 0:
        return "No source files used."
    if count == 1:
        return "1 unique source file displayed."
    return f"{count} unique source files displayed."


def downgrade_headings(text: str) -> str:
    """Downgrade Markdown heading levels by one step in generated content.

    # -> ##, ## -> ###, etc.  Prevents oversized headings in rendered Learn output.
    """
    # Process from h5 down to h1 so replacements don't collide.
    for level in range(5, 0, -1):
        pattern = re.compile(r"^" + "#" * level + r"\s", re.MULTILINE)
        replacement = "#" * (level + 1) + " "
        text = pattern.sub(replacement, text)
    return text


def format_trace_entry(entry: str) -> str:
    """Format a single trace entry for readable display."""
    return entry


def _format_learning_depth_value(style) -> str:
    """Map internal response-style values to user-facing dashboard labels."""
    val = style.value if hasattr(style, "value") else str(style)
    if val == "concise":
        return "Summary"
    if val == "detailed":
        return "Deep Study"
    return val


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
        fields.append({"label": "Learn Path", "value": val})

    style = result.get("style")
    if style:
        fields.append({"label": "Learning Depth", "value": _format_learning_depth_value(style)})

    docs = result.get("retrieved_docs", [])
    fields.append({"label": "Passages Retrieved", "value": str(len(docs))})

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
            "message": "No learning memory available yet. Complete quizzes and learning sessions to build personalized learning memory.",
        }

    result = {
        "loaded": True,
        "recent_topics": memory_profile.get("recent_topics", []),
        "weak_areas": memory_profile.get("recurring_weak_areas", []),
        "average_score": memory_profile.get("average_score"),
        "suggested_focus": memory_profile.get("suggested_focus_topics", []),
        "preferred_style": memory_profile.get("preferred_style"),
    }

    # If profile dict exists but contains no meaningful data, mark as not loaded
    has_data = (
        bool(result["recent_topics"])
        or bool(result["weak_areas"])
        or result["average_score"] is not None
        or bool(result["suggested_focus"])
        or bool(result["preferred_style"])
    )
    if not has_data:
        result["loaded"] = False

    return result


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
        "suggestion": "Try again or check the Dashboard section for details.",
    })
