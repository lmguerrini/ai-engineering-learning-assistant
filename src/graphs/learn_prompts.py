"""Prompt-building helpers for the Learn LangGraph workflow.
These pure functions compose the LLM prompt and memory context
for study guide generation.  They are used by learn_nodes.py.
"""
from src.graphs.learn_state import LearningState
from src.kb.loader import Document
from src.schemas import DifficultyLevel, ResponseStyle
from src.ui.shared import _LEARN_PATH_STABLE_TOPICS

_DEEP_STUDY_CODE_RULES = (
    "CODE BLOCK QUALITY RULES:\n"
    "- Code blocks must contain real, concise, meaningful, implementation-oriented examples.\n"
    "- Prefer short runnable examples or compact realistic snippets over long pseudo-implementations.\n"
    "- Do NOT produce comment-only code blocks, placeholder-only implementations, or fake scaffolding.\n"
    "- Do NOT write placeholder comments such as '# Logic to ...', '# Placeholder ...', or '... additional logic ...'.\n"
    "- If a concrete example would require too much invented code, explain the concept outside the code block instead of fabricating code.\n"
)

_LEARN_PATH_DEEP_STUDY_ENGINEERING_RULES = (
    "LEARN PATH SYSTEM CONNECTION RULES:\n"
    "- Write like an experienced AI engineering mentor: practical, educational, and accessible without becoming generic courseware.\n"
    "- Explain each topic in isolation, then add short 'System Connection' notes showing how it depends on, constrains, or reinforces other topics in this path.\n"
    "- Use System Connection notes for dependencies, tradeoffs, and failure propagation between concepts such as LangGraph, RAG, memory, HITL, observability, checkpointers, streaming, and deployment.\n"
    "- Prefer clear operational reasoning over repeated definitions; explain engineering tradeoffs, architecture decisions, and when to use / when not to use the approach.\n"
    "- Include production failure modes and incident-style examples where relevant: scaling bottlenecks, latency/token tradeoffs, state explosion, retrieval precision/recall failures, vector DB fragmentation, and hallucination containment gaps.\n"
    "- Cover operational patterns where relevant: async orchestration, concurrency, streaming orchestration, retry/backoff strategies, MCP/tool safety, observability, and eval methodology.\n"
    "- Examples should be minimal but real: prefer focused snippets with LangGraph nodes/checkpointers, LangChain retrievers, OpenAI client calls, vector DB operations, tracing hooks, retry/error boundaries, or deployment/runtime config.\n"
    "- Avoid toy scaffolding and fake production code: do not use placeholder helpers like call_tool() or log_error(e), fake domains like api.example.com, or tutorial filler that pretends to be a complete app.\n"
    "- Use comparison tables or decision matrices when they clarify implementation choices.\n"
    "- Make review checklists operational: include measurable criteria for latency, cost, reliability, eval coverage, rollback, and monitoring.\n"
)


def _build_sources_text(docs: list[Document]) -> str:
    """Render retrieved docs into a prompt-ready source block."""
    sources_text = ""
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("topic", doc.metadata.get("filename", f"source_{i}"))
        sources_text += f"\n--- Source {i}: {title} ---\n{doc.content}\n"
    return sources_text


def _build_difficulty_instruction(difficulty: DifficultyLevel) -> str:
    """Return prompt guidance tuned to the requested difficulty level."""
    if difficulty == DifficultyLevel.ADVANCED:
        return (
            "\nAdvanced-level requirements:\n"
            "- Discuss architecture tradeoffs and design decisions.\n"
            "- Cover implementation concerns and edge cases.\n"
            "- Include production considerations (scaling, error handling, monitoring).\n"
            "- Address observability and testing where relevant.\n"
            "- Mention security or reliability notes when applicable.\n"
            "- Assume the reader already knows the basics.\n"
        )
    if difficulty == DifficultyLevel.BEGINNER:
        return (
            "\nBeginner-level requirements:\n"
            "- Explain every concept from first principles.\n"
            "- Avoid jargon without defining it first.\n"
            "- Use simple analogies where helpful.\n"
        )
    return ""


def _build_personalization_context(state: LearningState) -> str:
    """Return the personalization section for prompts when memory exists."""
    memory_context = _build_memory_context(state)
    if not memory_context:
        return ""
    return f"\nPersonalization context:\n{memory_context}\n"


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

    sources_text = _build_sources_text(docs)

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
        topic_heading_str = "\n".join(f"   - ## {name}" for name in stable_topics)

        style_instruction = (
            "Produce a COMPACT CURRICULUM OVERVIEW for this Learn Path.\n"
            "Structure it as a concise but substantive syllabus-style document.\n\n"
            "Use the JSON `summary` field for the short path overview.\n"
            "In `detailed_notes`, do NOT repeat that overview and do NOT use generic numbered "
            "template headings. Keep the structure tied to the actual path topics.\n\n"
            "Use this exact structure in `detailed_notes`:\n\n"
            "1. A `## Study Sequence` section with a short note on the recommended order.\n"
            "2. A `## Learning Outcomes` section with 4-6 concise bullet points.\n"
            "3. One `## <Topic Name>` section for EACH Learn Path topic, using these exact headings:\n"
            f"{topic_heading_str}\n"
            "   For each topic section, include:\n"
            "   - a 2-3 sentence description of what the learner will study and why it matters\n"
            "   - 2-4 short bullet points covering key concepts or skills\n"
            "   - one brief prerequisite or connection note when relevant\n\n"
            "When you reference the path topics, use EXACTLY these topic names:\n"
            f"{topic_list_str}\n"
            "   Do NOT rename, reorder, or skip any topic.\n\n"
            "The result should read like a clean syllabus tied to the actual path topics.\n"
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

    difficulty_instruction = _build_difficulty_instruction(difficulty)

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

    personalization = _build_personalization_context(state)

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

    sources_text = _build_sources_text(docs)

    difficulty_instruction = _build_difficulty_instruction(difficulty)

    personalization = _build_personalization_context(state)

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
        f"explanation are unacceptable. The goal is practical engineering mentorship "
        f"that connects concepts across the path, not generic LLM courseware.\n\n"
        f"STRUCTURE REQUIREMENTS:\n"
        f"Use EXACTLY these major sections as ## headings, in this order:\n"
        f"{numbered_topics}\n\n"
        f"Do NOT rename, reorder, merge, or skip any section.\n\n"
        f"DEPTH REQUIREMENTS — for EACH section above, include ALL of these "
        f"subsections (### headings):\n"
        f"  1. **Theory & Context** — core concepts, formal definitions, mental "
        f"models, and where this fits in the broader AI/ML stack. Keep definitions "
        f"tight and emphasize why the concept matters in production. (3+ paragraphs minimum)\n"
        f"  2. **Architecture / Internal Design** — how the component is "
        f"structured internally; data flow, key abstractions, extension points, "
        f"state boundaries, orchestration model, and architecture tradeoffs. "
        f"Use ASCII diagrams, structured lists, comparison tables, or decision matrices where useful.\n"
        f"  3. **Implementation Details** — step-by-step approach with code in "
        f"```python blocks; cover configuration, parameters, integration points, "
        f"environment setup, retry/backoff, observability hooks, and safety boundaries. "
        f"Show how pieces connect in a real codebase.\n"
        f"  4. **Practical Examples** — at least TWO realistic code examples: "
        f"one basic usage, one advanced/production scenario. Keep examples short, "
        f"realistic, implementation-oriented, and tied to operational decisions or incident-style scenarios.\n"
        f"  5. **Common Mistakes & Anti-Patterns** — concrete pitfalls with "
        f"root-cause explanations, production failure modes, and corrective patterns. Include before/after "
        f"code pairs that contain ACTUAL runnable function/class structures, not "
        f"placeholder-only comments like '# bad approach'. Even illustrative code "
        f"must show meaningful logic.\n"
        f"  6. **Review Checklist** — 5-8 verification items for self-assessment "
        f"before shipping code that uses this topic; include operational criteria, "
        f"not just conceptual recall.\n\n"
        f"After each major topic section, include 2-4 short **System Connection** "
        f"notes that explain how this topic interacts with earlier or later path "
        f"topics through dependencies, tradeoffs, or failure propagation.\n\n"
        f"QUALITY RULES:\n"
        f"- Each subsection must be substantial (multiple paragraphs, not one-liners).\n"
        f"- Prefer fewer but RICHER subsections over many shallow ones.\n"
        f"- Every sentence must add information — do NOT pad with filler.\n"
        f"- Include concrete code examples wherever implementation is relevant.\n"
        f"{_LEARN_PATH_DEEP_STUDY_ENGINEERING_RULES}"
        f"- STRICT CODE BLOCK RULE: Every ```python or ``` code block MUST contain "
        f"executable or structurally meaningful code — functions, classes, conditionals, "
        f"loops, configuration dicts, test skeletons, or concrete CLI commands. "
        f"NEVER produce a code block whose only content is comments like "
        f"'# Set up a schedule for regular evaluations' or '# Complex chain with "
        f"multiple responsibilities'. If the idea is conceptual, express it as "
        f"prose or bullet text OUTSIDE a code block instead.\n"
        f"{_DEEP_STUDY_CODE_RULES}"
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
        f"with concise explanation outside the block when needed. One should "
        f"demonstrate basic usage, the other an advanced or production scenario.\n"
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
        f"{_DEEP_STUDY_CODE_RULES}"
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


def _build_progressive_summary_prompt(state: LearningState) -> str:
    """Build a small JSON prompt for the early Learn overview stage."""
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    docs: list[Document] = state.get("retrieved_docs", [])
    is_learn_path = ":" in topic and len(topic) > 60
    sources_text = _build_sources_text(docs)
    difficulty_instruction = _build_difficulty_instruction(difficulty)
    personalization = _build_personalization_context(state)

    if is_learn_path:
        level_label = difficulty.value.capitalize()
        stable_topics = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])
        topic_list = "\n".join(f"- {name}" for name in stable_topics)
        mode_instruction = (
            "This is a LEARN PATH request.\n"
            "Write a concise curriculum overview that introduces the path as a whole.\n"
            "For key_concepts, return ONLY these exact topic names in the same order:\n"
            f"{topic_list}\n"
            "Do not rename, merge, or annotate them."
        )
    else:
        mode_instruction = (
            "This is a single-topic Deep Study request.\n"
            "Write a concise overview and extract 5-7 key concepts.\n"
            "Each key concept may include a short explanation after a colon."
        )

    return (
        f"You are an AI Engineering tutor preparing the EARLY OVERVIEW stage of a "
        f"study guide.\n"
        f"Topic: '{topic}'\n"
        f"Level: {difficulty.value}\n\n"
        f"{mode_instruction}\n"
        f"{difficulty_instruction}\n"
        f"{personalization}\n"
        f"Use the following sources as reference material.\n"
        f"Synthesize in your own words. Do not quote source text verbatim.\n"
        f"{sources_text}\n\n"
        f"Return ONLY valid JSON in this schema:\n"
        f'{{\n'
        f'  "summary": "2-4 sentence overview",\n'
        f'  "key_concepts": ["Concept", "Concept: short explanation"]\n'
        f'}}'
    )


def _build_progressive_topic_section_prompt(
    state: LearningState,
    summary: str,
    key_concepts: list[str],
    section_names: list[str],
) -> str:
    """Build a prompt for one bundle of Topic-mode detailed sections."""
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    docs: list[Document] = state.get("retrieved_docs", [])
    sources_text = _build_sources_text(docs)
    difficulty_instruction = _build_difficulty_instruction(difficulty)
    personalization = _build_personalization_context(state)
    sections_text = "\n".join(f"- {name}" for name in section_names)
    concepts_text = "\n".join(f"- {concept}" for concept in key_concepts)

    return (
        f"You are writing ONE PART of a production-grade engineering reference.\n"
        f"Topic: '{topic}'\n"
        f"Level: {difficulty.value}\n\n"
        f"Existing overview:\n{summary}\n\n"
        f"Key concepts already identified:\n{concepts_text}\n\n"
        f"Write ONLY these sections, in this exact order, using ## headings:\n"
        f"{sections_text}\n\n"
        f"Rules:\n"
        f"- Return ONLY Markdown.\n"
        f"- Do NOT include an Overview section.\n"
        f"- Do NOT include Sources, citations, or JSON.\n"
        f"- Use substantial, technical explanations.\n"
        f"- Keep content self-contained and non-repetitive.\n"
        f"- Every ```python or ``` code block must contain meaningful code, not comments only.\n"
        f"- If a section requests a comparison table or checklist, render it directly in Markdown.\n"
        f"{difficulty_instruction}\n"
        f"{personalization}\n"
        f"Use these sources as grounding context:\n"
        f"{sources_text}\n"
    )


def _build_progressive_learn_path_section_prompt(
    state: LearningState,
    summary: str,
    topic_name: str,
    section_number: int,
) -> str:
    """Build a prompt for one Learn Path topic section in Deep Study mode."""
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    docs: list[Document] = state.get("retrieved_docs", [])
    sources_text = _build_sources_text(docs)
    difficulty_instruction = _build_difficulty_instruction(difficulty)
    personalization = _build_personalization_context(state)

    return (
        f"You are writing ONE major section of a technical curriculum handbook "
        f"as an experienced AI engineering mentor.\n"
        f"Curriculum summary:\n{summary}\n\n"
        f"Topic section: {section_number}. {topic_name}\n"
        f"Level: {difficulty.value}\n\n"
        f"Return ONLY Markdown with this exact top-level heading:\n"
        f"## {section_number}. {topic_name}\n\n"
        f"Inside that section, include these exact ### subsections in order:\n"
        f"### Theory & Context\n"
        f"### Architecture / Internal Design\n"
        f"### Implementation Details\n"
        f"### Practical Examples\n"
        f"### Common Mistakes & Anti-Patterns\n"
        f"### Review Checklist\n\n"
        f"After the checklist, add a short `### System Connection` subsection with "
        f"2-4 bullets showing dependencies, tradeoffs, or failure propagation between "
        f"this topic and other path topics such as LangGraph, RAG, memory, HITL, "
        f"observability, checkpointers, streaming, and deployment.\n\n"
        f"Rules:\n"
        f"- Do not add any other ## headings.\n"
        f"- Keep the section substantial and technically precise.\n"
        f"- Keep the tone practical, educational, and accessible rather than overly niche or encyclopedia-like.\n"
        f"- Use concrete code and configuration where relevant.\n"
        f"{_LEARN_PATH_DEEP_STUDY_ENGINEERING_RULES}"
        f"- Every ```python or ``` code block must contain meaningful code, not comments only.\n"
        f"{_DEEP_STUDY_CODE_RULES}"
        f"- Do not include sources, citations, or JSON.\n"
        f"{difficulty_instruction}\n"
        f"{personalization}\n"
        f"Use these sources as grounding context:\n"
        f"{sources_text}\n"
    )
