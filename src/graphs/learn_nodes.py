"""Node functions for the Learn LangGraph workflow.

Each function takes a LearningState dict and returns a partial state update.
"""

import json
from typing import Any

from loguru import logger
from openai import OpenAI

from src.config import get_settings
from src.graphs.learn_state import LearningState
from src.kb.loader import Document
from src.kb.retrieval import retrieve_documents
from src.schemas import DifficultyLevel, ResponseStyle, Source, StudyGuide
from src.services.cost_tracker import build_usage_record
from src.services.retry import with_retry

# Minimum number of retrieved chunks to consider sources sufficient
_MIN_SOURCES = 2
# Minimum total characters across chunks to consider sources sufficient
_MIN_CONTENT_CHARS = 200


# ---------------------------------------------------------------------------
# Node: validate_input
# ---------------------------------------------------------------------------

def validate_input(state: LearningState) -> dict:
    """Validate that required input fields are present and well-formed."""
    trace = list(state.get("trace", []))
    trace.append("validate_input: started")

    topic = state.get("topic", "").strip()
    if not topic:
        trace.append("validate_input: failed — empty topic")
        return {"error": "Please enter a topic to study.", "trace": trace}

    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    style = state.get("style", ResponseStyle.DETAILED)
    query = topic

    trace.append(f"validate_input: ok — topic='{topic}', difficulty={difficulty}, style={style}")
    return {
        "topic": topic,
        "difficulty": difficulty,
        "style": style,
        "query": query,
        "error": None,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: load_memory_placeholder
# ---------------------------------------------------------------------------

def load_memory_placeholder(state: LearningState) -> dict:
    """Placeholder for loading user memory (implemented in Phase 5)."""
    trace = list(state.get("trace", []))
    trace.append("load_memory_placeholder: no memory loaded (placeholder)")
    return {"user_memory": {}, "trace": trace}


# ---------------------------------------------------------------------------
# Node: retrieve_sources
# ---------------------------------------------------------------------------

def retrieve_sources(state: LearningState) -> dict:
    """Retrieve relevant documents from the knowledge base."""
    trace = list(state.get("trace", []))
    attempts = state.get("attempts", 0) + 1
    query = state.get("query", state.get("topic", ""))
    trace.append(f"retrieve_sources: query='{query}' (attempt {attempts})")

    docs = retrieve_documents(query=query, top_k=6)
    trace.append(f"retrieve_sources: got {len(docs)} chunks")
    return {"retrieved_docs": docs, "attempts": attempts, "trace": trace}


# ---------------------------------------------------------------------------
# Node: assess_source_quality
# ---------------------------------------------------------------------------

def assess_source_quality(state: LearningState) -> dict:
    """Evaluate whether retrieved sources are sufficient for guide generation."""
    trace = list(state.get("trace", []))
    docs: list[Document] = state.get("retrieved_docs", [])
    total_chars = sum(len(d.content) for d in docs)

    ok = len(docs) >= _MIN_SOURCES and total_chars >= _MIN_CONTENT_CHARS
    trace.append(
        f"assess_source_quality: {len(docs)} docs, {total_chars} chars → "
        f"{'sufficient' if ok else 'insufficient'}"
    )
    return {"source_quality_ok": ok, "trace": trace}


# ---------------------------------------------------------------------------
# Node: refine_query_if_needed
# ---------------------------------------------------------------------------

def refine_query_if_needed(state: LearningState) -> dict:
    """Broaden the query when sources were insufficient.

    Only updates the query and sets query_refined flag.
    The next step (retrieve_sources) will perform the actual retrieval.
    """
    trace = list(state.get("trace", []))
    topic = state.get("topic", "")
    old_query = state.get("query", topic)

    refined = f"{old_query} overview concepts examples"
    trace.append(f"refine_query_if_needed: refined '{old_query}' → '{refined}'")

    return {
        "query": refined,
        "query_refined": True,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: generate_study_guide
# ---------------------------------------------------------------------------

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

    style_instruction = {
        ResponseStyle.CONCISE: "Be concise and to the point.",
        ResponseStyle.DETAILED: "Be thorough and detailed.",
        ResponseStyle.EXAMPLES_HEAVY: "Use many practical examples.",
    }.get(style, "Be thorough and detailed.")

    return (
        f"You are an AI Engineering tutor. Generate a structured study guide "
        f"on the topic '{topic}' at {difficulty.value} level.\n\n"
        f"Style: {style_instruction}\n\n"
        f"Use ONLY the following sources to build the guide. "
        f"Do not invent information beyond what is in the sources.\n"
        f"{sources_text}\n\n"
        f"Respond with valid JSON matching this schema:\n"
        f'{{\n'
        f'  "topic": "{topic}",\n'
        f'  "difficulty": "{difficulty.value}",\n'
        f'  "summary": "2-4 sentence overview",\n'
        f'  "key_concepts": ["concept1", "concept2", ...],\n'
        f'  "detailed_notes": "multi-paragraph markdown notes",\n'
        f'  "sources": [{{"title": "...", "content_snippet": "...", "relevance_score": 0.9}}]\n'
        f'}}\n\n'
        f"Return ONLY the JSON object, no extra text."
    )


def _parse_study_guide(raw: str, state: LearningState) -> StudyGuide:
    """Parse the LLM output into a StudyGuide, with fallback."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    data = json.loads(text)
    return StudyGuide(**data)


def _build_fallback_guide(state: LearningState) -> StudyGuide:
    """Build a minimal guide from retrieved sources when LLM fails."""
    docs: list[Document] = state.get("retrieved_docs", [])
    topic = state.get("topic", "Unknown")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)

    sources = []
    notes_parts = []
    for doc in docs[:5]:
        title = doc.metadata.get("topic", doc.metadata.get("filename", "source"))
        sources.append(Source(title=title, content_snippet=doc.content[:200], relevance_score=0.5))
        notes_parts.append(doc.content[:500])

    return StudyGuide(
        topic=topic,
        difficulty=difficulty,
        summary=f"Study guide for {topic} (generated from retrieved sources without LLM).",
        key_concepts=[topic],
        detailed_notes="\n\n".join(notes_parts) if notes_parts else "No content available.",
        sources=sources,
    )


def generate_study_guide(state: LearningState) -> dict:
    """Generate a structured study guide using the LLM."""
    trace = list(state.get("trace", []))
    trace.append("generate_study_guide: started")
    token_usage = dict(state.get("token_usage", {}))

    settings = get_settings()
    if not settings.openai_api_key:
        trace.append("generate_study_guide: no API key — using fallback guide")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage}

    prompt = _build_prompt(state)

    model = settings.app_default_model
    usage_records = list(state.get("usage_records", []))

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        def _llm_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

        response = with_retry(
            callable=_llm_call,
            max_attempts=2,
            base_delay=1.0,
            handled_exceptions=(Exception,),
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + (usage.prompt_tokens or 0)
            token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + (usage.completion_tokens or 0)
            token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + (usage.total_tokens or 0)

        usage_dict = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        usage_records.append(build_usage_record(model, "learn_guide_generation", usage_dict))

        trace.append(f"generate_study_guide: LLM returned {len(raw)} chars")

        guide = _parse_study_guide(raw, state)
        trace.append("generate_study_guide: parsed successfully")
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    except json.JSONDecodeError as e:
        logger.warning("Malformed LLM JSON output: {}", e)
        trace.append(f"generate_study_guide: malformed JSON — {e}, using fallback")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    except Exception as e:
        logger.error("LLM call failed: {}", e)
        trace.append(f"generate_study_guide: LLM error — {e}, using fallback")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}


# ---------------------------------------------------------------------------
# Node: quality_check
# ---------------------------------------------------------------------------

def quality_check(state: LearningState) -> dict:
    """Check the generated study guide meets minimum quality."""
    trace = list(state.get("trace", []))
    guide = state.get("study_guide")

    if guide is None:
        trace.append("quality_check: no guide produced")
        return {"quality_passed": False, "trace": trace}

    passed = bool(guide.summary and guide.key_concepts and guide.detailed_notes)
    trace.append(f"quality_check: {'passed' if passed else 'failed'}")
    return {"quality_passed": passed, "trace": trace}


# ---------------------------------------------------------------------------
# Node: persist_learning_event_placeholder
# ---------------------------------------------------------------------------

def persist_learning_event_placeholder(state: LearningState) -> dict:
    """Placeholder for persisting the learning event (Phase 5)."""
    trace = list(state.get("trace", []))
    trace.append("persist_learning_event_placeholder: skipped (placeholder)")
    return {"trace": trace}


# ---------------------------------------------------------------------------
# Node: return_output
# ---------------------------------------------------------------------------

def return_output(state: LearningState) -> dict:
    """Final node — ensures output fields are set."""
    trace = list(state.get("trace", []))
    trace.append("return_output: done")
    return {"trace": trace}
