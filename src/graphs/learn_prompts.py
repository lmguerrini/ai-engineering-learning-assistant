"""Prompt-building helpers for the Learn LangGraph workflow.
These pure functions compose the LLM prompt and memory context
for study guide generation.  They are used by learn_nodes.py.
"""
from src.graphs.learn_state import LearningState
from src.kb.loader import Document
from src.schemas import DifficultyLevel, ResponseStyle
from src.ui.shared import _LEARN_PATH_STABLE_TOPICS


def is_deep_study_learn_path(state: LearningState) -> bool:
    """Return True when the request is Deep Study + Learn Path.

    This combination uses a dedicated markdown-only generation flow
    (no JSON) to avoid truncation / malformed-JSON issues.
    """
    style = state.get("style", ResponseStyle.CONCISE)
    topic = state.get("topic", "")
    is_deep = style in (ResponseStyle.DETAILED, ResponseStyle.EXAMPLES_HEAVY)
    is_learn_path = ":" in topic and len(topic) > 60
    return is_deep and is_learn_path


def is_deep_study_topic(state: LearningState) -> bool:
    """Return True when the request is Deep Study + single Topic.

    Uses a dedicated markdown-only generation flow (no JSON) to avoid
    the truncated-JSON / malformed-JSON problem on long outputs.
    """
    style = state.get("style", ResponseStyle.CONCISE)
    topic = state.get("topic", "")
    is_deep = style in (ResponseStyle.DETAILED, ResponseStyle.EXAMPLES_HEAVY)
    is_learn_path = ":" in topic and len(topic) > 60
    return is_deep and not is_learn_path


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
    is_learn_path = ":" in topic and len(topic) > 60

    # --- Deep Study + Learn Path: long-form handbook ---
    if is_deep and is_learn_path:
        style_instruction = (
            "You are writing a PROFESSIONAL CURRICULUM HANDBOOK.\n"
            "This is an intensive Deep Study mode. The output must be very long and thorough.\n\n"
            "STRUCTURE REQUIREMENTS:\n"
            "Identify every distinct sub-topic listed in the Learn Path topic string.\n"
            "For EACH sub-topic, produce a MAJOR standalone section with ALL of the following:\n"
            "  1. Theory — core concepts, definitions, and mental models\n"
            "  2. Architecture — how the component is structured internally\n"
            "  3. When to Use — concrete scenarios and decision criteria\n"
            "  4. Implementation Details — step-by-step approach with code in ```python blocks\n"
            "  5. Practical Example — a realistic, runnable code example\n"
            "  6. Common Mistakes — pitfalls and how to avoid them\n"
            "  7. Review Checklist — 5-8 verification items for self-assessment\n\n"
            "Each major section must be substantial (multiple paragraphs, not one-liners).\n"
            "Use ## for each major topic heading, ### for sub-sections within.\n"
            "Include concrete code examples wherever implementation is relevant.\n"
            "Use proper Markdown formatting throughout.\n"
            "The total output should read like a chapter from a technical handbook.\n"
            "Do NOT compress multiple topics into a single shallow overview.\n"
            "Do NOT skip any sub-topic."
        )
    # --- Deep Study + single Topic ---
    elif is_deep:
        style_instruction = (
            "You are writing an EXHAUSTIVE, production-grade engineering reference "
            "on a single topic.\n"
            "Use ALL retrieved context fully — extract every useful detail, do not "
            "summarise sources superficially.\n"
            "The output must be long, deeply technical, and self-contained.\n\n"
            "REQUIRED SECTIONS (use ## headings, in this exact order):\n"
            "1. **Overview** — scope, purpose, relevance, and where this topic fits "
            "in the broader AI/ML engineering stack (3-5 paragraphs).\n"
            "2. **Conceptual Foundations** — core theory, formal definitions, mental "
            "models, and underlying principles explained with precision. Include "
            "mathematical intuition or pseudocode where it aids understanding.\n"
            "3. **Architecture / Internal Design** — how the component is structured "
            "internally; data flow, key abstractions, extension points. Use ASCII "
            "diagrams or structured lists to illustrate architecture.\n"
            "4. **Implementation Details** — step-by-step approach with code in "
            "```python blocks; cover configuration, parameters, integration points, "
            "and environment setup. Show how pieces connect in a real codebase.\n"
            "5. **Practical Examples** — at least two realistic, runnable code examples "
            "with line-by-line commentary. One should demonstrate basic usage, the "
            "other an advanced or production scenario.\n"
            "6. **Production Considerations** — scaling, performance tuning, error "
            "handling, monitoring, observability hooks, and deployment patterns.\n"
            "7. **Common Mistakes & Anti-Patterns** — concrete pitfalls with root-cause "
            "explanations and corrective patterns. Include before/after code where useful.\n"
            "8. **When to Use / When Not to Use** — decision matrix, alternatives, "
            "trade-offs, and migration paths from/to competing approaches.\n"
            "9. **Comparison Table** — a Markdown table comparing this approach with "
            "2-3 alternatives across dimensions like complexity, performance, "
            "ecosystem support, and learning curve.\n"
            "10. **Review Checklist** — 8-10 verification items an engineer should "
            "confirm before shipping code that uses this topic.\n\n"
            "Each section must be substantial (multiple paragraphs, not one-liners).\n"
            "Prioritize concrete, actionable engineering content over abstract prose.\n"
            "Do NOT merge sections or skip any.\n"
            "Do NOT include a 'Topics' list — this is a single-topic deep study.\n"
            "Do NOT pad with filler — every sentence must add information.\n"
            "Use proper Markdown formatting throughout."
        )
    # --- Summary + Learn Path: compact curriculum overview ---
    elif is_learn_path:
        level_label = difficulty.value.capitalize()
        stable_topics = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])
        topic_list_str = "\n".join(f"  {i}. {name}" for i, name in enumerate(stable_topics, 1))

        style_instruction = (
            "Produce a COMPACT CURRICULUM OVERVIEW for this Learn Path.\n"
            "Structure it as a concise but substantive syllabus-style document.\n\n"
            "Use this EXACT section order:\n\n"
            "1. **Overview** — 3-5 sentences explaining the purpose and scope of this path.\n\n"
            "2. **Topics** — a compact Markdown table with these columns:\n"
            "   | # | Topic | Key Focus | Prerequisites | Estimated Effort |\n"
            "   Use EXACTLY these topic names, one row per topic:\n"
            f"{topic_list_str}\n"
            "   Do NOT rename, reorder, or skip any topic.\n\n"
            "3. **Recommended Study Order** — a brief note on the suggested sequence.\n\n"
            "4. **Learning Outcomes** — 5-7 bullet points describing what the learner will achieve.\n\n"
            "5. **Learn Path Overview** — for each topic listed above, provide:\n"
            "   • A 2-3 sentence description of what the learner will study and why it matters.\n"
            "   • 3-4 key concepts as bullet points with brief explanations.\n"
            "   • One sentence on prerequisites or connections to other topics.\n\n"
            "The overview should be rich enough to serve as a standalone syllabus.\n"
            "Do NOT include long code examples or deep explanations.\n"
            "Use proper Markdown formatting."
        )
    # --- Summary + single Topic ---
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
        f"Use the following sources as reference material. "
        f"Synthesize and explain concepts IN YOUR OWN WORDS. "
        f"Do NOT copy or paste source text verbatim into the output. "
        f"Do NOT include raw source snippets in detailed_notes. "
        f"Keep all source attribution in the 'sources' JSON field only.\n"
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


def _build_deep_study_markdown_prompt(state: LearningState) -> str:
    """Build a prompt for Deep Study + Learn Path that requests raw Markdown.

    Unlike ``_build_prompt`` this does NOT ask for JSON output.  The LLM
    returns a long-form Markdown handbook directly, which avoids the
    truncated-JSON / malformed-JSON problem entirely.
    """
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    docs: list[Document] = state.get("retrieved_docs", [])

    sources_text = ""
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("topic", doc.metadata.get("filename", f"source_{i}"))
        sources_text += f"\n--- Source {i}: {title} ---\n{doc.content}\n"

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

    memory_context = _build_memory_context(state)
    personalization = ""
    if memory_context:
        personalization = f"\nPersonalization context:\n{memory_context}\n"

    level_label = difficulty.value.capitalize()
    stable_topics = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])
    numbered_topics = "\n".join(
        f"  ## {i}. {name}" for i, name in enumerate(stable_topics, 1)
    )

    return (
        f"You are a senior AI/ML engineer writing an EXHAUSTIVE, production-grade "
        f"engineering curriculum.\n"
        f"Topic: '{topic}'\n"
        f"Level: {difficulty.value}\n\n"
        f"This is an intensive Deep Study mode — the DEEPEST content tier.\n"
        f"Each topic section must read like a standalone technical reference, "
        f"NOT a brief overview. Shallow sections with many headings and little "
        f"explanation are unacceptable.\n\n"
        f"STRUCTURE REQUIREMENTS:\n"
        f"Use EXACTLY these major sections as ## headings, in this order:\n"
        f"{numbered_topics}\n\n"
        f"Do NOT rename, reorder, merge, or skip any section.\n\n"
        f"DEPTH REQUIREMENTS — for EACH section above, include ALL of these "
        f"subsections (### headings):\n"
        f"  1. **Theory & Context** — core concepts, formal definitions, mental "
        f"models, and where this fits in the broader AI/ML stack. Explain the "
        f"WHY, not just the WHAT. (3+ paragraphs minimum)\n"
        f"  2. **Architecture / Internal Design** — how the component is "
        f"structured internally; data flow, key abstractions, extension points. "
        f"Use ASCII diagrams or structured lists to illustrate.\n"
        f"  3. **Implementation Details** — step-by-step approach with code in "
        f"```python blocks; cover configuration, parameters, integration points, "
        f"and environment setup. Show how pieces connect in a real codebase.\n"
        f"  4. **Practical Examples** — at least TWO realistic code examples: "
        f"one basic usage, one advanced/production scenario. Include line-by-line "
        f"commentary explaining key decisions.\n"
        f"  5. **Common Mistakes & Anti-Patterns** — concrete pitfalls with "
        f"root-cause explanations and corrective patterns. Include before/after "
        f"code pairs that contain ACTUAL runnable function/class structures, not "
        f"placeholder-only comments like '# bad approach'. Even illustrative code "
        f"must show meaningful logic.\n"
        f"  6. **Review Checklist** — 5-8 verification items for self-assessment "
        f"before shipping code that uses this topic.\n\n"
        f"QUALITY RULES:\n"
        f"- Each subsection must be substantial (multiple paragraphs, not one-liners).\n"
        f"- Prefer fewer but RICHER subsections over many shallow ones.\n"
        f"- Every sentence must add information — do NOT pad with filler.\n"
        f"- Include concrete code examples wherever implementation is relevant.\n"
        f"- STRICT CODE BLOCK RULE: Every ```python or ``` code block MUST contain "
        f"executable or structurally meaningful code — functions, classes, conditionals, "
        f"loops, configuration dicts, test skeletons, or concrete CLI commands. "
        f"NEVER produce a code block whose only content is comments like "
        f"'# Set up a schedule for regular evaluations' or '# Complex chain with "
        f"multiple responsibilities'. If the idea is conceptual, express it as "
        f"prose or bullet text OUTSIDE a code block instead.\n"
        f"- Before/after code pairs must show real function/class bodies with logic, "
        f"not comment-only placeholders.\n"
        f"- Use ## for each major topic heading, ### for sub-sections within.\n"
        f"- Use proper Markdown formatting throughout.\n"
        f"- The total output should be VERY LONG — comparable to a chapter from "
        f"a professional technical book.\n"
        f"- Do NOT compress multiple topics into a single shallow overview.\n"
        f"- Do NOT skip any sub-topic or subsection.\n"
        f"{difficulty_instruction}\n"
        f"{personalization}\n"
        f"Use the following sources as reference material.\n"
        f"Synthesize IN YOUR OWN WORDS — do NOT copy source text verbatim.\n"
        f"Use ALL retrieved context fully — extract every useful detail.\n"
        f"{sources_text}\n\n"
        f"OUTPUT FORMAT: Return ONLY Markdown text. Do NOT wrap in JSON.\n"
        f"Do NOT include any JSON structure. Just write the curriculum in Markdown."
    )


def _build_deep_study_topic_markdown_prompt(state: LearningState) -> str:
    """Build a prompt for Deep Study + single Topic that requests raw Markdown.

    Mirrors ``_build_deep_study_markdown_prompt`` but tailored for a single
    topic deep-dive.  Returns Markdown directly — no JSON wrapper.
    """
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    docs: list[Document] = state.get("retrieved_docs", [])

    sources_text = ""
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("topic", doc.metadata.get("filename", f"source_{i}"))
        sources_text += f"\n--- Source {i}: {title} ---\n{doc.content}\n"

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

    memory_context = _build_memory_context(state)
    personalization = ""
    if memory_context:
        personalization = f"\nPersonalization context:\n{memory_context}\n"

    return (
        f"You are a senior AI/ML engineer writing an EXHAUSTIVE, production-grade "
        f"engineering reference on a single topic.\n"
        f"Topic: '{topic}'\n"
        f"Level: {difficulty.value}\n\n"
        f"This is an intensive Deep Study mode — a focused deep-dive into ONE subject.\n"
        f"The output must be very long, deeply technical, and self-contained.\n"
        f"Use ALL retrieved context fully — extract every useful detail.\n\n"
        f"REQUIRED SECTIONS (use ## headings, in this exact order):\n"
        f"1. **Overview** — scope, purpose, relevance, and where this topic fits "
        f"in the broader AI/ML engineering stack (3-5 paragraphs).\n"
        f"2. **Conceptual Foundations** — core theory, formal definitions, mental "
        f"models, and underlying principles explained with precision. Include "
        f"mathematical intuition or pseudocode where it aids understanding.\n"
        f"3. **Architecture / Internal Design** — how the component is structured "
        f"internally; data flow, key abstractions, extension points. Use ASCII "
        f"diagrams or structured lists to illustrate architecture.\n"
        f"4. **Implementation Details** — step-by-step approach with code in "
        f"```python blocks; cover configuration, parameters, integration points, "
        f"and environment setup. Show how pieces connect in a real codebase.\n"
        f"5. **Practical Examples** — at least two realistic, runnable code examples "
        f"with line-by-line commentary. One should demonstrate basic usage, the "
        f"other an advanced or production scenario.\n"
        f"6. **Production Considerations** — scaling, performance tuning, error "
        f"handling, monitoring, observability hooks, and deployment patterns.\n"
        f"7. **Common Mistakes & Anti-Patterns** — concrete pitfalls with root-cause "
        f"explanations and corrective patterns. Before/after code pairs must contain "
        f"ACTUAL runnable function/class structures, not placeholder-only comments "
        f"like '# bad approach'. Even illustrative code must show meaningful logic.\n"
        f"8. **When to Use / When Not to Use** — decision matrix, alternatives, "
        f"trade-offs, and migration paths from/to competing approaches.\n"
        f"9. **Comparison Table** — a Markdown table comparing this approach with "
        f"2-3 alternatives across dimensions like complexity, performance, "
        f"ecosystem support, and learning curve.\n"
        f"10. **Review Checklist** — 8-10 verification items an engineer should "
        f"confirm before shipping code that uses this topic.\n\n"
        f"QUALITY RULES:\n"
        f"- Each section must be substantial (multiple paragraphs, not one-liners).\n"
        f"- Prefer fewer but RICHER subsections over many shallow ones. Avoid many "
        f"headings with short explanations — minor points should be bullets or "
        f"smaller subsections, not top-level headings.\n"
        f"- Each major section should contain substantial academic/professional "
        f"explanation comparable to a textbook chapter section.\n"
        f"- Prioritize concrete, actionable engineering content over abstract prose.\n"
        f"- All code examples must contain actual function/class/logic structure — "
        f"never use comment-only placeholders. Even illustrative code must be "
        f"semi-runnable with meaningful structure.\n"
        f"- Do NOT merge sections or skip any.\n"
        f"- Do NOT include a 'Topics' list — this is a single-topic deep study.\n"
        f"- Do NOT pad with filler — every sentence must add information.\n"
        f"- Use proper Markdown formatting throughout.\n"
        f"{difficulty_instruction}\n"
        f"{personalization}\n"
        f"Use the following sources as reference material.\n"
        f"Synthesize IN YOUR OWN WORDS — do NOT copy source text verbatim.\n"
        f"{sources_text}\n\n"
        f"OUTPUT FORMAT: Return ONLY Markdown text. Do NOT wrap in JSON.\n"
        f"Do NOT include any JSON structure. Just write the reference in Markdown."
    )
