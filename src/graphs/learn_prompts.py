"""Prompt-building helpers for the Learn LangGraph workflow.

These pure functions compose the LLM prompt and memory context
for study guide generation.  They are used by learn_nodes.py.
"""

from src.graphs.learn_state import LearningState
from src.kb.loader import Document
from src.schemas import DifficultyLevel, ResponseStyle


def _build_memory_context(state: LearningState) -> str:
    """Build personalization context from memory profile if available."""
    profile = state.get("memory_profile", {})
    if not profile or not profile.get("recent_topics"):
        return ""

    parts: list[str] = []
    weak = profile.get("recurring_weak_areas", [])
    if weak:
        parts.append(f"The learner has recurring weak areas in: {', '.join(weak[:5])}.")
        parts.append("Emphasize these concepts if they relate to the current topic.")

    recent = profile.get("recent_topics", [])
    if recent:
        parts.append(f"Recently studied topics: {', '.join(recent[:5])}.")
        parts.append("Mention useful connections to these topics where appropriate.")

    avg = profile.get("average_score")
    if avg is not None:
        if avg < 50:
            parts.append("The learner's average score is low — use simpler language and more foundational explanations.")
        elif avg >= 80:
            parts.append("The learner's average score is high — include more advanced nuances.")

    fb_suggestion = profile.get("feedback_suggestion")
    if fb_suggestion == "simplify":
        parts.append("User feedback indicates content should be simpler and clearer.")
    elif fb_suggestion == "increase_difficulty":
        parts.append("User feedback indicates content could be more challenging.")

    return "\n".join(parts)


def _build_prompt(state: LearningState) -> str:
    """Build the LLM prompt for study guide generation."""
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    style = state.get("style", ResponseStyle.DETAILED)
    docs: list[Document] = state.get("retrieved_docs", [])

    sources_text = ""
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("topic", doc.metadata.get("filename", f"source_{i}"))
        sources_text += f"\n--- Source {i}: {title} ---\n{doc.content}\n"

    is_deep = style in (ResponseStyle.DETAILED, ResponseStyle.EXAMPLES_HEAVY)

    if is_deep:
        style_instruction = (
            "Produce a comprehensive, academic-quality Learn Path.\n"
            "Structure the output with clearly separated Markdown sections:\n"
            "1. Overview\n"
            "2. Conceptual Explanation\n"
            "3. Architecture / Implementation Details\n"
            "4. Practical Examples (include code in ```python blocks when relevant)\n"
            "5. Common Mistakes\n"
            "6. When to Use / When Not to Use\n"
            "7. Review Checklist (5-8 verification items)\n"
            "8. Summary Table (Markdown table comparing key aspects)\n\n"
            "Each section must be substantial. Do not write shallow one-liners.\n"
            "Include concrete code examples where the topic involves implementation.\n"
            "Use proper Markdown formatting throughout."
        )
    else:
        style_instruction = (
            "Be concise and schematic. Provide a short summary-level Learn Path.\n"
            "Use bullet points for key points.\n"
            "Optionally include a compact Markdown table comparing key aspects.\n"
            "Do not include a long Table of Contents or large examples.\n"
            "Each key concept should be one sentence max."
        )

    difficulty_instruction = ""
    if difficulty == DifficultyLevel.ADVANCED:
        difficulty_instruction = (
            "\nAdvanced-level requirements:\n"
            "- Discuss architecture tradeoffs and design decisions.\n"
            "- Cover implementation concerns and edge cases.\n"
            "- Include production considerations (scaling, error handling, monitoring).\n"
            "- Address observability and testing where relevant.\n"
            "- Mention security or reliability notes when applicable.\n"
            "- Assume the reader already knows the basics.\n"
        )
    elif difficulty == DifficultyLevel.BEGINNER:
        difficulty_instruction = (
            "\nBeginner-level requirements:\n"
            "- Explain every concept from first principles.\n"
            "- Avoid jargon without defining it first.\n"
            "- Use simple analogies where helpful.\n"
        )

    # Learn Path mode produces broader multi-topic curriculum;
    # Topic mode is focused on a single subject.
    is_learn_path = ":" in topic and len(topic) > 60
    if is_learn_path:
        mode_instruction = (
            "This is a LEARN PATH — a guided multi-topic curriculum.\n"
            "Cover ALL listed sub-topics with dedicated sections for each.\n"
            "Structure the output like a course module, not a single-topic answer.\n"
            "Provide broader coverage and more material than a single-topic study.\n"
        )
    else:
        mode_instruction = (
            "This is a TOPIC study — a focused deep-dive into one subject.\n"
            "Stay focused on this single topic and cover it thoroughly.\n"
        )

    memory_context = _build_memory_context(state)
    personalization = ""
    if memory_context:
        personalization = f"\nPersonalization context:\n{memory_context}\n"

    return (
        f"You are an AI Engineering tutor. Generate a structured Learn Path "
        f"on the topic '{topic}' at {difficulty.value} level.\n\n"
        f"{mode_instruction}\n"
        f"Style: {style_instruction}\n"
        f"{difficulty_instruction}\n"
        f"{personalization}\n"
        f"Use ONLY the following sources to build the Learn Path. "
        f"Do not invent information beyond what is in the sources.\n"
        f"{sources_text}\n\n"
        f"For key_concepts, provide each concept with a one-sentence explanation "
        f"separated by a colon, e.g. 'Concept Name: Brief explanation of the concept.'\n\n"
        f"Respond with valid JSON matching this schema:\n"
        f'{{\n'
        f'  "topic": "{topic}",\n'
        f'  "difficulty": "{difficulty.value}",\n'
        f'  "summary": "2-4 sentence overview",\n'
        f'  "key_concepts": ["Concept: explanation", ...],\n'
        f'  "detailed_notes": "multi-paragraph markdown notes with subsections",\n'
        f'  "sources": [{{"title": "...", "content_snippet": "...", "relevance_score": 0.9}}]\n'
        f'}}\n\n'
        f"Return ONLY the JSON object, no extra text."
    )
