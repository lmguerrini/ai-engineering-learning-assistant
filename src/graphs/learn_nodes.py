"""Node functions for the Learn LangGraph workflow.

Each function takes a LearningState dict and returns a partial state update.
"""

import json
from typing import Any

from loguru import logger
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from src.config import get_settings
from src.graphs.learn_prompts import (
    _build_memory_context,
    _build_prompt,
    _build_progressive_learn_path_section_prompt,
    _build_progressive_summary_prompt,
    _build_progressive_topic_section_prompt,
    _build_deep_study_markdown_prompt,
    _build_deep_study_topic_markdown_prompt,
    is_deep_study_learn_path,
    is_deep_study_topic,
)
from src.graphs.learn_state import LearningState
from src.kb.loader import Document
from src.kb.official_docs import retrieve_with_fallback
from src.kb.retrieval import retrieve_documents
from src.schemas import DifficultyLevel, ResponseStyle, Source, StudyGuide
from src.services.cache import build_cache_key, get_cached_value, set_cached_value
from src.services.cost_tracker import build_usage_record
from src.services.retry import with_retry
from src.ui.shared import _LEARN_PATH_STABLE_TOPICS

# Minimum number of retrieved chunks to consider sources sufficient
_MIN_SOURCES = 2
# Minimum total characters across chunks to consider sources sufficient
_MIN_CONTENT_CHARS = 200
# Prompt version — bump to invalidate stale cached outputs
_PROMPT_VERSION = "v16"
_TOPIC_DEEP_STUDY_BUNDLES = [
    [
        "Conceptual Foundations",
        "Architecture / Internal Design",
        "Implementation Details",
    ],
    [
        "Practical Examples",
        "Production Considerations",
        "Common Mistakes & Anti-Patterns",
    ],
    [
        "When to Use / When Not to Use",
        "Comparison Table",
        "Review Checklist",
    ],
]


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
# Node: load_user_memory
# ---------------------------------------------------------------------------

def load_user_memory(state: LearningState) -> dict:
    """Load user memory profile and feedback summary."""
    trace = list(state.get("trace", []))
    trace.append("load_user_memory: started")

    try:
        from src.memory.memory_service import get_user_profile_summary as _get_profile

        profile = _get_profile()
    except Exception as e:
        logger.warning("Failed to load memory profile: {}", e)
        profile = {
            "recent_topics": [],
            "recurring_weak_areas": [],
            "average_score": None,
            "preferred_style": None,
            "suggested_focus_topics": [],
        }

    # Attach feedback suggestion to profile for downstream use
    try:
        from src.memory.feedback_service import get_feedback_summary

        fb = get_feedback_summary()
        if fb.get("suggestion"):
            profile["feedback_suggestion"] = fb["suggestion"]
    except Exception:
        pass

    has_data = bool(profile.get("recent_topics"))
    trace.append(
        f"load_user_memory: {'profile loaded' if has_data else 'no memory data'}"
    )
    return {"user_memory": profile, "memory_profile": profile, "trace": trace}


# ---------------------------------------------------------------------------
# Node: retrieve_sources
# ---------------------------------------------------------------------------

def _get_learn_path_topics(state: LearningState) -> list[str] | None:
    """Return per-topic query list if state represents a Learn Path, else None."""
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    level_label = difficulty.value.capitalize()
    topics = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])
    # Learn Path topics contain ":" in the topic string (e.g. "Foundations of …: LLM basics, …")
    if ":" in topic and topics:
        return topics
    return None


def retrieve_sources(state: LearningState) -> dict:
    """Retrieve relevant documents from the knowledge base.

    First retrieves from the curated KB, then uses official docs
    as fallback/enrichment.  For Deep Study (DETAILED style), official
    docs are always merged to enrich technical precision.

    For Learn Path Deep Study, retrieval is topic-aware: chunks are
    retrieved per individual Learn Path topic, then merged and
    deduplicated to ensure source diversity across all topics.
    """
    trace = list(state.get("trace", []))
    attempts = state.get("attempts", 0) + 1
    query = state.get("query", state.get("topic", ""))
    style = state.get("style", ResponseStyle.CONCISE)
    is_deep = style in (ResponseStyle.DETAILED, ResponseStyle.EXAMPLES_HEAVY)
    trace.append(f"retrieve_sources: query='{query}' (attempt {attempts})")

    # --- Topic-aware retrieval for Learn Path Deep Study ---
    path_topics = _get_learn_path_topics(state) if is_deep else None

    if path_topics:
        # Per-topic retrieval: retrieve a few chunks per topic, merge, dedup
        per_topic_k = max(4, 10 // len(path_topics) + 1)
        all_curated: list[Document] = []
        seen_content: set[str] = set()
        for sub_topic in path_topics:
            topic_docs = retrieve_documents(query=sub_topic, top_k=per_topic_k)
            for d in topic_docs:
                key = d.content[:100]
                if key not in seen_content:
                    all_curated.append(d)
                    seen_content.add(key)
        # Also retrieve with the full query to catch cross-topic chunks
        full_docs = retrieve_documents(query=query, top_k=6)
        for d in full_docs:
            key = d.content[:100]
            if key not in seen_content:
                all_curated.append(d)
                seen_content.add(key)
        trace.append(
            f"retrieve_sources: topic-aware retrieval — "
            f"{len(path_topics)} topics, {per_topic_k} chunks/topic, "
            f"{len(all_curated)} unique curated chunks"
        )
        curated_docs = all_curated
    else:
        curated_top_k = 10 if is_deep else 6
        curated_docs = retrieve_documents(query=query, top_k=curated_top_k)
        trace.append(f"retrieve_sources: got {len(curated_docs)} curated chunks")

    if is_deep:
        # Deep Study: always enrich with official docs regardless of sufficiency
        from src.kb.official_docs import retrieve_official_docs

        official_query = query
        official_docs = retrieve_official_docs(query=official_query, top_k=6)
        curated_count = len(curated_docs)
        # Merge curated (primary) + official (enrichment), dedup by content
        seen_content = {d.content[:100] for d in curated_docs}
        official_added = 0
        for od in official_docs:
            if od.content[:100] not in seen_content:
                curated_docs.append(od)
                seen_content.add(od.content[:100])
                official_added += 1
        trace.append(
            f"retrieve_sources: Deep Study — curated={curated_count}, "
            f"official_retrieved={len(official_docs)}, "
            f"official_added={official_added}, "
            f"final={len(curated_docs)}"
        )
        docs = curated_docs
    else:
        # Summary: use official docs only as fallback when curated is insufficient
        docs = retrieve_with_fallback(
            query=query,
            curated_docs=curated_docs,
            min_sources=_MIN_SOURCES,
            min_content_chars=_MIN_CONTENT_CHARS,
        )
        official_count = len(docs) - len(curated_docs)
        if official_count > 0:
            trace.append(f"retrieve_sources: added {official_count} official doc chunks as fallback")

    trace.append(f"retrieve_sources: total {len(docs)} chunks")
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

# _build_memory_context and _build_prompt live in learn_prompts.py
# and are imported at the top of this module for backward compatibility.


def _parse_study_guide(raw: str, state: LearningState) -> StudyGuide:
    """Parse the LLM output into a StudyGuide, with fallback.

    Always populates sources from retrieved_docs so the guide carries
    real provenance information regardless of what the LLM returned.
    Attempts to recover from truncated JSON when the LLM output was
    cut short by the max_tokens limit.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to recover truncated JSON — close any open string and braces
        repaired = text.rstrip()
        # If truncated inside a string value, close it
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        # Close any open arrays/objects
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)
        data = json.loads(repaired)  # raises if still invalid

    # Build sources from retrieved docs (authoritative) instead of relying
    # on whatever the LLM chose to include in its JSON output.
    docs: list[Document] = state.get("retrieved_docs", [])
    if docs:
        sources = []
        for doc in docs[:5]:
            title = doc.metadata.get("topic", doc.metadata.get("filename", "source"))
            meta = {k: str(v) for k, v in doc.metadata.items() if k in ("filename", "source", "topic", "source_type")}
            sources.append(Source(title=title, content_snippet=doc.content[:200], relevance_score=0.5, metadata=meta))
        data["sources"] = [s.model_dump() for s in sources]

    return StudyGuide(**data)


def _build_fallback_guide(state: LearningState) -> StudyGuide:
    """Build a clean minimal guide when LLM generation fails.

    Does NOT dump raw retrieved chunks into the body.  Instead,
    produces a short synthesized placeholder with topic names so
    the user sees a presentable result while sources remain in the
    Sources section only.
    """
    docs: list[Document] = state.get("retrieved_docs", [])
    topic = state.get("topic", "Unknown")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)

    # Collect source metadata for the Sources section only
    sources = []
    seen_titles: set[str] = set()
    for doc in docs[:5]:
        title = doc.metadata.get("topic", doc.metadata.get("filename", "source"))
        meta = {k: str(v) for k, v in doc.metadata.items() if k in ("filename", "source", "topic")}
        sources.append(Source(title=title, content_snippet=doc.content[:200], relevance_score=0.5, metadata=meta))
        seen_titles.add(title)

    # Build a clean topic list from source titles
    topic_list = ", ".join(sorted(seen_titles)) if seen_titles else topic

    return StudyGuide(
        topic=topic,
        difficulty=difficulty,
        summary=(
            f"An overview of {topic}. This guide covers the key concepts "
            f"and provides a starting point for further study."
        ),
        key_concepts=[f"{t}: See the sources section for details." for t in sorted(seen_titles)[:8]] or [topic],
        detailed_notes=(
            f"This study guide could not be fully generated at this time. "
            f"The following topics are covered by the available sources: "
            f"{topic_list}.\n\n"
            f"Please try again or select a different topic for a complete guide."
        ),
        sources=sources,
    )


def _build_learn_cache_key(state: LearningState) -> str:
    """Build a cache key for the study guide based on inputs + memory hash."""
    profile = state.get("memory_profile", {})
    payload = {
        "topic": state.get("topic", ""),
        "difficulty": str(state.get("difficulty", "")),
        "style": str(state.get("style", "")),
        "memory_hash": str(sorted(profile.items())) if profile else "",
        "prompt_version": _PROMPT_VERSION,
    }
    return build_cache_key("learn_guide", payload)


def _build_sources_list(state: LearningState) -> list[Source]:
    """Build a list of Source objects from retrieved documents."""
    docs: list[Document] = state.get("retrieved_docs", [])
    sources: list[Source] = []
    for doc in docs[:5]:
        title = doc.metadata.get("topic", doc.metadata.get("filename", "source"))
        meta = {
            k: str(v)
            for k, v in doc.metadata.items()
            if k in ("filename", "source", "topic", "source_type")
        }
        sources.append(
            Source(
                title=title,
                content_snippet=doc.content[:200],
                relevance_score=0.5,
                metadata=meta,
            )
        )
    return sources


def _extract_summary_from_markdown(markdown: str) -> str:
    """Extract the first meaningful paragraph as summary from handbook markdown."""
    lines = markdown.strip().splitlines()
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip headings and empty lines at the start
        if not paragraph and (not stripped or stripped.startswith("#")):
            continue
        if stripped:
            paragraph.append(stripped)
        elif paragraph:
            break  # end of first paragraph
    return " ".join(paragraph[:4]) if paragraph else "Deep Study curriculum reference."


def _strip_code_fences(text: str) -> str:
    """Remove outer fenced code blocks when present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        return stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return stripped


def _parse_json_payload(raw: str) -> dict[str, Any]:
    """Parse a small JSON payload returned by the model."""
    return json.loads(_strip_code_fences(raw))


def _append_usage(
    response: Any,
    *,
    model: str,
    operation: str,
    token_usage: dict[str, int],
    usage_records: list[dict[str, Any]],
) -> None:
    """Accumulate token usage and append a usage record for one LLM call."""
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
    usage_records.append(build_usage_record(model, operation, usage_dict))


def _call_llm_text(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    trace: list[str],
    trace_label: str,
    token_usage: dict[str, int],
    usage_records: list[dict[str, Any]],
    usage_operation: str,
) -> str:
    """Execute a single OpenAI completion call and accumulate usage."""
    def _llm_call() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    response = with_retry(
        callable=_llm_call,
        max_attempts=2,
        base_delay=1.0,
        handled_exceptions=(Exception,),
    )
    raw = response.choices[0].message.content or ""
    _append_usage(
        response,
        model=model,
        operation=usage_operation,
        token_usage=token_usage,
        usage_records=usage_records,
    )
    trace.append(f"{trace_label}: LLM returned {len(raw)} chars")
    return raw


def _build_progressive_guide(
    *,
    topic: str,
    difficulty: DifficultyLevel,
    summary: str,
    key_concepts: list[str],
    detailed_notes: str,
    sources: list[Source],
) -> StudyGuide:
    """Construct a StudyGuide object for partial-progress UI updates."""
    return StudyGuide(
        topic=topic,
        difficulty=difficulty,
        summary=summary,
        key_concepts=key_concepts,
        detailed_notes=detailed_notes,
        sources=sources,
    )


def _emit_progress_update(state: LearningState, guide: StudyGuide, trace: list[str], stage: str) -> None:
    """Invoke the optional UI progress callback with the latest partial guide."""
    callback = state.get("progress_callback")
    if not callback:
        return
    try:
        callback(guide)
        trace.append(f"generate_study_guide: progress emitted ({stage})")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Learn progress callback failed at stage '{}': {}", stage, exc)


def _coerce_key_concepts(value: Any, fallback: list[str]) -> list[str]:
    """Normalize model-returned key concept items into a clean list."""
    if not isinstance(value, list):
        return fallback
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or fallback


def _generate_deep_study_learn_path(state: LearningState) -> dict:
    """Dedicated generation flow for Deep Study + Learn Path.

    Generates the handbook as raw Markdown (no JSON) and constructs
    the StudyGuide object manually.  This avoids the truncated-JSON
    problem that occurs when a huge markdown handbook is wrapped in JSON.
    """
    trace = list(state.get("trace", []))
    trace.append("generate_study_guide: deep_study_learn_path flow")
    token_usage = dict(state.get("token_usage", {}))
    usage_records = list(state.get("usage_records", []))

    # --- Check cache ---
    cache_key = _build_learn_cache_key(state)
    if not state.get("force_regenerate"):
        cached = get_cached_value(cache_key)
        if cached is not None:
            try:
                guide = StudyGuide(**cached)
                trace.append("generate_study_guide: cache hit")
                return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}
            except Exception:
                pass
    else:
        trace.append("generate_study_guide: cache bypassed (force_regenerate)")

    settings = get_settings()
    if not settings.openai_api_key:
        trace.append("generate_study_guide: no API key — using fallback guide")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    prompt = _build_deep_study_markdown_prompt(state)
    model = settings.app_default_model

    try:
        client = wrap_openai(OpenAI(api_key=settings.openai_api_key))

        def _llm_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=16384,
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

        trace.append(f"generate_study_guide: LLM returned {len(raw)} chars (markdown)")

        # --- Build StudyGuide manually from markdown ---
        topic = state.get("topic", "Deep Study")
        difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)

        # Use stable, professionally-capitalised topic list
        topic_parts = topic.split(":", 1)
        curriculum_title = topic_parts[0].strip()
        level_label = difficulty.value.capitalize()
        key_concepts = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])

        summary = _extract_summary_from_markdown(raw)
        sources = _build_sources_list(state)

        guide = StudyGuide(
            topic=curriculum_title,
            difficulty=difficulty,
            summary=summary,
            key_concepts=key_concepts,
            detailed_notes=raw.strip(),
            sources=sources,
        )
        trace.append("generate_study_guide: StudyGuide built from markdown")

        # Cache the result
        try:
            set_cached_value(cache_key, guide.model_dump(), ttl_seconds=3600)
            trace.append("generate_study_guide: cached")
        except Exception:
            pass

        return {
            "study_guide": guide,
            "trace": trace,
            "token_usage": token_usage,
            "usage_records": usage_records,
        }

    except Exception as e:
        logger.error("Deep Study LLM call failed: {}", e)
        trace.append(f"generate_study_guide: LLM error — {e}, using fallback")
        guide = _build_fallback_guide(state)
        return {
            "study_guide": guide,
            "trace": trace,
            "token_usage": token_usage,
            "usage_records": usage_records,
        }


def _generate_deep_study_topic(state: LearningState) -> dict:
    """Dedicated generation flow for Deep Study + single Topic.

    Generates the content as raw Markdown (no JSON) and constructs
    the StudyGuide object manually.  This avoids the truncated-JSON
    / malformed-JSON problem that occurs with large outputs.
    """
    trace = list(state.get("trace", []))
    trace.append("generate_study_guide: deep_study_topic flow")
    token_usage = dict(state.get("token_usage", {}))
    usage_records = list(state.get("usage_records", []))

    # --- Check cache ---
    cache_key = _build_learn_cache_key(state)
    if not state.get("force_regenerate"):
        cached = get_cached_value(cache_key)
        if cached is not None:
            try:
                guide = StudyGuide(**cached)
                trace.append("generate_study_guide: cache hit")
                return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}
            except Exception:
                pass
    else:
        trace.append("generate_study_guide: cache bypassed (force_regenerate)")

    settings = get_settings()
    if not settings.openai_api_key:
        trace.append("generate_study_guide: no API key — using fallback guide")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    prompt = _build_deep_study_topic_markdown_prompt(state)
    model = settings.app_default_model

    try:
        client = wrap_openai(OpenAI(api_key=settings.openai_api_key))

        def _llm_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=16384,
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

        trace.append(f"generate_study_guide: LLM returned {len(raw)} chars (markdown)")

        # --- Build StudyGuide manually from markdown ---
        topic = state.get("topic", "Deep Study")
        difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
        summary = _extract_summary_from_markdown(raw)
        sources = _build_sources_list(state)

        guide = StudyGuide(
            topic=topic,
            difficulty=difficulty,
            summary=summary,
            key_concepts=[topic],
            detailed_notes=raw.strip(),
            sources=sources,
        )
        trace.append("generate_study_guide: StudyGuide built from markdown")

        # Cache the result
        try:
            set_cached_value(cache_key, guide.model_dump(), ttl_seconds=3600)
            trace.append("generate_study_guide: cached")
        except Exception:
            pass

        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    except Exception as e:
        logger.error("LLM call failed (deep_study_topic): {}", e)
        trace.append(f"generate_study_guide: LLM error — {e}, using fallback")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records, "generation_failed": True}


def generate_study_guide(state: LearningState) -> dict:
    """Generate a structured study guide using the LLM.

    For Deep Study modes, delegates to dedicated markdown-only flows
    that avoid JSON parsing of huge content.
    """
    # --- Dedicated flow for Deep Study + Learn Path ---
    if is_deep_study_learn_path(state):
        return _generate_deep_study_learn_path(state)

    # --- Dedicated flow for Deep Study + single Topic ---
    if is_deep_study_topic(state):
        return _generate_deep_study_topic(state)

    trace = list(state.get("trace", []))
    trace.append("generate_study_guide: started")
    token_usage = dict(state.get("token_usage", {}))
    usage_records = list(state.get("usage_records", []))

    # --- Check cache ---
    cache_key = _build_learn_cache_key(state)
    if not state.get("force_regenerate"):
        cached = get_cached_value(cache_key)
        if cached is not None:
            try:
                guide = StudyGuide(**cached)
                trace.append("generate_study_guide: cache hit")
                return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}
            except Exception:
                pass  # invalid cache entry — continue to LLM
    else:
        trace.append("generate_study_guide: cache bypassed (force_regenerate)")

    settings = get_settings()
    if not settings.openai_api_key:
        trace.append("generate_study_guide: no API key — using fallback guide")
        guide = _build_fallback_guide(state)
        return {"study_guide": guide, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    prompt = _build_prompt(state)

    model = settings.app_default_model

    try:
        client = wrap_openai(OpenAI(api_key=settings.openai_api_key))

        style = state.get("style", ResponseStyle.CONCISE)
        is_deep = style in (ResponseStyle.DETAILED, ResponseStyle.EXAMPLES_HEAVY)
        max_tokens = 16384 if is_deep else 2048

        def _llm_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
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

        # Cache the result
        try:
            set_cached_value(cache_key, guide.model_dump(), ttl_seconds=3600)
            trace.append("generate_study_guide: cached")
        except Exception:
            pass

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
