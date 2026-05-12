"""Scoped Help Assistant service with approved live-doc enrichment.

The assistant is intentionally narrow:
- only AI engineering / app-domain questions
- only approved official live sources
- no autonomous browsing or crawling
"""

from __future__ import annotations

import ast
import json
import re
from difflib import get_close_matches
from typing import Any
from urllib.parse import urlparse

from langsmith.wrappers import wrap_openai
from loguru import logger
from openai import OpenAI

from src.config import get_settings
from src.kb.loader import Document
from src.kb.official_docs import detect_query_domains, infer_domain, retrieve_official_docs
from src.kb.retrieval import retrieve_documents
from src.services.cost_tracker import build_usage_record
from src.services.external_docs_updater import (
    fetch_external_doc_url,
    get_external_doc_source_registry,
    preview_external_doc_content,
)
from src.services.retry import with_retry

OUT_OF_DOMAIN_MESSAGE = (
    "Out of domain — I can only help with AI engineering, RAG systems, agents, "
    "evaluation, LangChain ecosystem, and related topics."
)

HELP_ASSISTANT_PERSONALITY_MODES = (
    "Technical",
    "Concise",
    "Friendly",
    "Formal",
)

HELP_ASSISTANT_PERSONALITY_PROFILES: dict[str, dict[str, str]] = {
    "Technical": {
        "tone": "Pragmatic senior AI engineer",
        "verbosity": "Dense",
        "best_use_case": "Implementation, architecture, and tradeoff questions",
        "output_style": "Adaptive technical mode: compact system overview for workflow questions, runtime architecture discussion for implementation and debugging questions",
        "prompt_contract": (
            "Response style: Technical.\n"
            "- Adopt the voice of a senior AI engineer explaining the system to another engineer.\n"
            "- Assume the reader already sees the UI and wants to understand the system underneath it.\n"
            "- Answer like an implementation review, architecture discussion, or engineer handoff rather than a tutorial.\n"
            "- Keep the explanation compact, concrete, and technically literate.\n"
            "- Let the question type control the framing: compact system overview for workflow questions, runtime reasoning for debugging and architecture questions.\n"
            "- Keep Formal as the documentation/manual personality; Technical should read like engineer-to-engineer explanation."
        ),
    },
    "Concise": {
        "tone": "Direct and compressed",
        "verbosity": "Low",
        "best_use_case": "Quick answers and fast reviewer checks",
        "output_style": "Short answer first, then minimal supporting bullets only if needed",
        "prompt_contract": (
            "Response style: Concise.\n"
            "- Keep the answer visibly short.\n"
            "- Start with the direct answer in the first sentence.\n"
            "- Use minimal background and only the smallest helpful bullet list when needed.\n"
            "- Avoid long transitions, caveats, or extra context unless the question requires them.\n"
            "- Make the answer clearly shorter than the other personalities would."
        ),
    },
    "Friendly": {
        "tone": "Warm, approachable, onboarding-oriented",
        "verbosity": "Moderate",
        "best_use_case": "Explanations for newer users or softer onboarding",
        "output_style": "Guided teammate-style explanation with natural pacing, softer transitions, and light structure",
        "prompt_contract": (
            "Response style: Friendly.\n"
            "- Use approachable language, softer transitions, and natural conversational pacing.\n"
            "- Explain concepts as if helping a capable teammate who is new to this part of the stack.\n"
            "- Prefer simpler wording first, then introduce jargon only when useful and explain it briefly.\n"
            "- Prefer short paragraphs over bullets; use bullets only when they genuinely make the explanation easier to follow.\n"
            "- Keep the tone warm, guided, and professional without becoming casual, chatty, or fluffy.\n"
            "- Make the explanation feel like a helpful walkthrough, not compressed engineering notes."
        ),
    },
    "Formal": {
        "tone": "Neutral, professional, documentation-like",
        "verbosity": "Moderate",
        "best_use_case": "Reviewer summaries, enterprise documentation tone",
        "output_style": "Polished documentation-style answer with explicit sectioning and restrained wording",
        "prompt_contract": (
            "Response style: Formal.\n"
            "- Use polished, neutral, professional wording.\n"
            "- Keep the structure orderly, sectioned, and documentation-like.\n"
            "- Prefer explicit headings, complete sentences, restrained transitions, and orderly sequencing.\n"
            "- Avoid conversational phrasing or overly dense jargon.\n"
            "- Present the answer as if it were written for internal technical documentation."
        ),
    },
}

HELP_ASSISTANT_RUNTIME_DEFAULTS = {
    "Concise": {
        "temperature": 0.2,
        "top_p": 0.8,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.0,
        "max_tokens": 1100,
    },
    "Friendly": {
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.3,
        "max_tokens": 1100,
    },
    "Formal": {
        "temperature": 0.3,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.0,
        "max_tokens": 1100,
    },
    "Technical": {
        "temperature": 0.15,
        "top_p": 0.85,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 1100,
    },
}

HELP_ASSISTANT_EXAMPLES = [
    "Explain LangGraph state management",
    "Difference between RAG and Agentic RAG",
    "How does tool calling work in LangChain?",
    "How should I evaluate a RAG pipeline with RAGAs?",
]

HELP_ASSISTANT_EXAMPLE_GROUPS = {
    "App workflow": [
        "How does this app work?",
        "What does the dashboard show?",
        "How do Learn and Quiz work?",
        "What is Official Docs Sync?",
        "How does the app use official docs alongside curated notes?",
        "When does Help Assistant use live official docs?",
        "What is the difference between official snapshots and live docs enrichment?",
    ],
    "Core AI / KB concepts": [
        "Difference between RAG and Agentic RAG",
        "Explain LangGraph state management",
        "How does KB index work?",
        "How should I evaluate a RAG pipeline with RAGAs?",
    ],
    "Official docs / live enrichment": [
        "How does tool calling work in LangChain?",
        "What do OpenAI structured outputs require?",
        "How do LangGraph reducers work?",
        "What does LangSmith tracing capture?",
    ],
}

_GENERAL_DOMAIN_KEYWORDS = [
    "ai engineering",
    "llm",
    "prompt",
    "embedding",
    "retrieval",
    "rag",
    "agentic rag",
    "agent",
    "tool calling",
    "function calling",
    "evaluation",
    "observability",
    "vector store",
    "vector database",
    "memory",
    "human in the loop",
    "checkpoint",
    "langchain",
    "langgraph",
    "langsmith",
    "streamlit",
    "chroma",
    "pydantic",
    "ragas",
    "openai",
    "study guide",
    "learn path",
    "knowledge base",
    "kb index",
    "official docs",
    "official snapshot",
    "official snapshots",
    "live docs",
    "live docs enrichment",
    "live official docs",
    "curated notes",
]

_APP_HELP_PATTERNS = [
    "this app",
    "use this app",
    "how does this app work",
    "how do i use this app",
    "what does the dashboard show",
    "dashboard show",
    "official docs sync",
    "how does kb index work",
    "kb index",
    "how do learn and quiz work",
    "learn and quiz",
    "learn path",
    "progressive streaming",
    "help assistant",
    "session cost",
    "session tokens",
    "official docs alongside curated notes",
    "official snapshots",
    "live docs enrichment",
    "live official docs enrichment",
    "difference between official snapshots and live docs enrichment",
    "use official docs alongside curated notes",
]

_FOLLOW_UP_PATTERNS = [
    "what was my last question",
    "what did you just say",
    "what was the last answer",
    "can you explain that more simply",
    "explain that more simply",
    "can you expand on that",
    "expand on that",
    "what did you say",
    "that question",
    "that answer",
    "that source",
    "those sources",
    "live official docs enrichment",
    "live docs enrichment",
    "official snapshots and live docs enrichment",
]

_APP_WORKFLOW_CONTEXT = (
    "App Workflow Context:\n"
    "- Learn: generates topic study guides and Learn Paths using local KB retrieval/RAG.\n"
    "- Quiz: generates and evaluates quizzes grounded in retrieved materials.\n"
    "- Progress: shows saved quiz attempts, weak areas, and learning history.\n"
    "- Dashboard: reviewer-facing page with Observability, Agent Capabilities / Tool Registry, Token and Cost Tracking, Help Assistant summary, Knowledge Base Health, External Docs / API Updater, Evaluation Readiness (RAGAs), Learning Signals, and Workflow Readiness.\n"
    "- Runtime Info: sidebar panel showing OpenAI/LangSmith readiness, KB Index, Official Docs Sync, RAGAs Evaluation, Agent Personality, model, session tokens, and session cost.\n"
    "- Official Docs Sync: manually refreshes configured official-doc snapshots; a KB rebuild is needed afterward.\n"
    "- RAGAs Evaluation: dashboard section for cached benchmark results and manual evaluation reruns.\n"
    "- KB Index: knowledge-base health state shown in the sidebar and Dashboard, with manual rebuild controls.\n"
    "- Help Assistant: answers in-domain app and AI-engineering questions using curated KB, official snapshots, and approved live official docs when relevant.\n"
    "- Live enrichment: used only when the question matches configured official source domains or explicitly asks about official docs, live docs, or current docs. It does not crawl arbitrary sites."
)

_SOURCE_HINTS: dict[str, dict[str, Any]] = {
    "agent_harness_production.md": {
        "domain": "langgraph",
        "keywords": ["deployment", "production", "architecture", "agent harness"],
    },
    "ai_agents_react_pattern.md": {
        "domain": "agents",
        "keywords": ["agent", "react", "reasoning", "tool use"],
    },
    "chroma_vector_store.md": {
        "domain": "chroma",
        "keywords": ["chroma", "chromadb", "vector store", "vector database"],
    },
    "langchain_core_tools.md": {
        "domain": "langchain",
        "keywords": ["langchain", "tools", "lcel", "runnable", "chain"],
    },
    "langgraph_state_orchestration.md": {
        "domain": "langgraph",
        "keywords": ["langgraph", "state", "orchestration", "graph"],
    },
    "langsmith_observability.md": {
        "domain": "langsmith",
        "keywords": ["langsmith", "tracing", "observability"],
    },
    "loguru_logging.md": {
        "domain": "loguru",
        "keywords": ["loguru", "logging", "logger"],
    },
    "memory_human_in_the_loop.md": {
        "domain": "langgraph",
        "keywords": ["memory", "human in the loop", "approval", "review"],
    },
    "openai_api_structured_outputs.md": {
        "domain": "openai",
        "keywords": ["openai", "structured outputs", "function calling", "responses api"],
    },
    "pydantic_validation_settings.md": {
        "domain": "pydantic",
        "keywords": ["pydantic", "validation", "settings", "schema"],
    },
    "ragas_evaluation.md": {
        "domain": "ragas",
        "keywords": ["ragas", "evaluation", "faithfulness", "answer relevancy"],
    },
    "state_management_agentic_rag.md": {
        "domain": "rag",
        "keywords": ["agentic rag", "rag", "state management", "retrieval"],
    },
    "streamlit_app_patterns.md": {
        "domain": "streamlit",
        "keywords": ["streamlit", "ui", "sidebar", "app patterns"],
    },
    "tool_calling_function_calling.md": {
        "domain": "tools",
        "keywords": ["tool calling", "function calling", "tools", "structured tool-call"],
    },
}

_SOURCE_KIND_LABELS = {
    "curated_kb": "Curated KB",
    "official_snapshot": "Official Snapshot",
    "live_official_docs": "Live Official Docs",
}


def _humanize_structured_key(key: Any) -> str:
    """Convert a dict key into a readable markdown heading label."""
    text = str(key).strip().replace("_", " ")
    if not text:
        return "Section"
    return text[:1].upper() + text[1:]


def _format_structured_value_markdown(value: Any, *, depth: int = 3) -> str:
    """Convert structured dict/list content into readable markdown prose."""
    if isinstance(value, dict):
        parts: list[str] = []
        heading = "#" * min(depth, 6)
        for key, nested in value.items():
            label = _humanize_structured_key(key)
            rendered = _format_structured_value_markdown(nested, depth=depth + 1)
            if rendered.startswith("- ") or rendered.startswith(f"{'#' * min(depth + 1, 6)} "):
                parts.append(f"{heading} {label}\n{rendered}")
            else:
                parts.append(f"{heading} {label}\n{rendered}")
        return "\n\n".join(part for part in parts if part.strip())

    if isinstance(value, (list, tuple)):
        lines: list[str] = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                rendered = _format_structured_value_markdown(item, depth=depth + 1)
                indented = "\n".join(
                    f"  {line}" if line else line
                    for line in rendered.splitlines()
                )
                lines.append(f"-\n{indented}")
            else:
                lines.append(f"- {str(item).strip()}")
        return "\n".join(line for line in lines if line.strip())

    if value is None:
        return "Not available."
    if isinstance(value, bool):
        return "Yes." if value else "No."
    return str(value).strip()


def _coerce_help_answer_to_markdown(answer_markdown: str) -> str:
    """Convert accidental structured payloads into readable markdown prose."""
    cleaned = _strip_generated_sources_section(answer_markdown).strip()
    if not cleaned:
        return ""

    stripped = cleaned.strip()
    if not stripped.startswith(("{", "[")):
        return cleaned

    parsed: Any | None = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            parsed = None

    if isinstance(parsed, (dict, list, tuple)):
        rendered = _format_structured_value_markdown(parsed).strip()
        return rendered or cleaned
    if isinstance(parsed, str):
        return parsed.strip() or cleaned
    return cleaned


def get_help_assistant_examples() -> list[str]:
    """Return example prompts for the Help Assistant page."""
    return list(HELP_ASSISTANT_EXAMPLES)


def get_help_assistant_example_groups() -> dict[str, list[str]]:
    """Return categorized example prompts for sidebar guidance."""
    return {label: list(prompts) for label, prompts in HELP_ASSISTANT_EXAMPLE_GROUPS.items()}


def get_help_assistant_personality_modes() -> list[str]:
    """Return supported Help Assistant response styles for the UI."""
    return list(HELP_ASSISTANT_PERSONALITY_MODES)


def get_help_assistant_personality_profiles() -> dict[str, dict[str, str]]:
    """Return reviewer-facing personality profile metadata."""
    return {mode: dict(profile) for mode, profile in HELP_ASSISTANT_PERSONALITY_PROFILES.items()}


def get_help_assistant_runtime_defaults(mode: str | None) -> dict[str, float | int]:
    """Return the default sampling/runtime config for one Help Assistant style."""
    normalized = normalize_help_assistant_personality_mode(mode)
    return dict(HELP_ASSISTANT_RUNTIME_DEFAULTS[normalized])


def get_help_assistant_source_registry() -> list[dict[str, Any]]:
    """Return approved live sources enriched with domain/keyword metadata."""
    registry: list[dict[str, Any]] = []
    for source in get_external_doc_source_registry():
        filename = str(source["filename"])
        hint = _SOURCE_HINTS.get(filename, {})
        registry.append(
            {
                "name": str(source["name"]),
                "filename": filename,
                "urls": list(source["urls"]),
                "domain": str(hint.get("domain") or infer_domain(filename)),
                "keywords": list(hint.get("keywords", [])),
            }
        )
    return registry


def get_help_assistant_scope() -> dict[str, Any]:
    """Return supported topics and approved source configuration for the UI/tests."""
    registry = get_help_assistant_source_registry()
    return {
        "supported_topics": sorted(set(_GENERAL_DOMAIN_KEYWORDS)),
        "approved_domains": sorted({row["domain"] for row in registry}),
        "example_prompts": get_help_assistant_examples(),
        "out_of_domain_message": OUT_OF_DOMAIN_MESSAGE,
    }


def get_help_assistant_app_workflow_context() -> str:
    """Return the built-in app workflow grounding block for app-help answers."""
    return _APP_WORKFLOW_CONTEXT


def classify_technical_help_question(question: str) -> str:
    """Classify a Technical-mode question as product overview or runtime architecture."""
    lower = _normalize_query(question)
    if not lower:
        return "runtime_architecture"

    product_overview_patterns = (
        "how does this app work",
        "explain the app workflow",
        "what does the dashboard do",
        "what does the dashboard show",
        "how do learn and quiz work",
        "how does the app use official docs alongside curated notes",
        "when does help assistant use live official docs",
        "what is official docs sync",
    )
    if any(pattern in lower for pattern in product_overview_patterns):
        return "product_overview"

    runtime_patterns = (
        "why does",
        "why is",
        "how would you debug",
        "debug retrieval drift",
        "retrieval drift",
        "skip retrieval",
        "separate local retrieval",
        "local retrieval from live enrichment",
        "kb rebuild",
        "state management",
        "state flow",
        "control flow",
        "routing",
        "orchestration",
        "tool calling",
        "function calling",
        "langsmith tracing",
        "observability",
        "failure behavior",
        "failure mode",
        "live enrichment restricted",
    )
    if any(pattern in lower for pattern in runtime_patterns):
        return "runtime_architecture"

    if is_app_help_query(question):
        return "product_overview"
    return "runtime_architecture"


def normalize_help_assistant_personality_mode(mode: str | None) -> str:
    """Return a supported Help Assistant response style."""
    if mode in HELP_ASSISTANT_PERSONALITY_MODES:
        return str(mode)
    return "Technical"


def is_app_help_query(query: str) -> bool:
    """Return whether the query is about this app's workflow or reviewer UX."""
    lower = _normalize_query(query)
    return bool(lower) and any(pattern in lower for pattern in _APP_HELP_PATTERNS)


def is_help_query_in_domain(
    query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> bool:
    """Return whether the question is in scope for the Help Assistant."""
    lower = _normalize_query(query)
    if not lower:
        return False

    if conversation_history and _is_follow_up_query(lower):
        return True

    if is_app_help_query(query):
        return True

    if detect_query_domains(query):
        return True

    return any(keyword in lower for keyword in _GENERAL_DOMAIN_KEYWORDS)


def select_live_help_sources(query: str, max_sources: int = 3) -> list[dict[str, Any]]:
    """Select the most relevant approved official live sources for a query."""
    lower = _normalize_query(query)
    registry = get_help_assistant_source_registry()
    matched_domains = set(detect_query_domains(query))
    is_app_help = is_app_help_query(query)

    scored: list[tuple[float, dict[str, Any]]] = []
    for source in registry:
        score = 0.0
        source_domain = str(source["domain"])
        if source_domain in matched_domains:
            score += 4.0
        if source_domain != "general" and source_domain in lower:
            score += 2.0
        for keyword in source.get("keywords", []):
            if str(keyword).lower() in lower:
                score += 1.0
        if score > 0:
            scored.append((score, source))

    if not scored and is_app_help:
        return []

    if not scored and is_help_query_in_domain(query):
        fallback_domains = ("langchain", "langgraph", "openai", "rag", "agents")
        for source in registry:
            if source["domain"] in fallback_domains:
                scored.append((0.25, source))

    scored.sort(key=lambda item: (-item[0], str(item[1]["name"])))
    selected: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for _score, source in scored:
        filename = str(source["filename"])
        if filename in seen_files:
            continue
        seen_files.add(filename)
        selected.append(source)
        if len(selected) >= max_sources:
            break

    return selected


def answer_help_question(
    question: str,
    *,
    model: str | None = None,
    live_fetcher=None,
    conversation_history: list[dict[str, Any]] | None = None,
    personality_mode: str | None = None,
    runtime_config: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Answer one in-domain question with local KB + approved live-doc enrichment."""
    query = question.strip()
    history = conversation_history or []
    trace: list[str] = []
    style_mode = normalize_help_assistant_personality_mode(personality_mode)
    sampling = _normalize_help_runtime_config(runtime_config, personality_mode=style_mode)
    personality_metadata = _build_help_personality_metadata(style_mode, sampling)

    if not query:
        return {
            "status": "error",
            "question": "",
            "answer_markdown": "",
            "message": "Please enter a question for the Help Assistant.",
            "sources": [],
            "trace": ["validate_input: empty question"],
            "live_enrichment_used": False,
            "usage_records": [],
            **personality_metadata,
        }

    trace.append("validate_scope: started")
    if not is_help_query_in_domain(query, conversation_history=history):
        trace.append("validate_scope: refused — out of domain")
        return {
            "status": "refused",
            "question": query,
            "answer_markdown": "",
            "message": OUT_OF_DOMAIN_MESSAGE,
            "sources": [],
            "trace": trace,
            "live_enrichment_used": False,
            "usage_records": [],
            **personality_metadata,
        }

    history_only_answer = _answer_from_history_if_possible(
        query,
        history,
        trace,
        personality_metadata=personality_metadata,
    )
    if history_only_answer is not None:
        return history_only_answer

    settings = get_settings()
    model_name = model or settings.app_default_model
    context_query = _resolve_help_context_query(query, history)
    if context_query != query:
        trace.append(f"follow_up_context: using prior question — {context_query}")
    app_help = is_app_help_query(query) or is_app_help_query(context_query)

    if app_help:
        trace.append("retrieve_local_context: skipped — app workflow context preferred")
        curated_docs = []
        official_docs = []
    else:
        trace.append("retrieve_local_context: started")
        curated_docs = retrieve_documents(query=context_query, top_k=4)
        official_docs = retrieve_official_docs(query=context_query, top_k=4)
        trace.append(
            f"retrieve_local_context: curated={len(curated_docs)}, official_snapshot={len(official_docs)}"
        )

    if app_help:
        trace.append("select_live_sources: skipped — app workflow question")
        selected_live_sources = []
    else:
        trace.append("select_live_sources: started")
        selected_live_sources = select_live_help_sources(context_query)
        trace.append(
            f"select_live_sources: selected={len(selected_live_sources)} from approved registry"
        )

    live_fetch = live_fetcher or fetch_external_doc_url
    live_sources, live_failures = _fetch_live_context(selected_live_sources, live_fetch, trace)

    source_rows = _build_source_rows(curated_docs, official_docs, live_sources)
    prompt = _build_help_assistant_prompt(
        question=query,
        context_question=context_query,
        curated_docs=curated_docs,
        official_docs=official_docs,
        live_sources=live_sources,
        conversation_history=history,
        personality_mode=style_mode,
    )

    if not settings.openai_api_key:
        trace.append("answer_generation: failed — missing OpenAI API key")
        return {
            "status": "error",
            "question": query,
            "answer_markdown": "",
            "message": "OpenAI API key is required to answer Help Assistant questions.",
            "sources": source_rows,
            "trace": trace,
            "live_enrichment_used": bool(live_sources),
            "usage_records": [],
            **personality_metadata,
        }

    trace.append(f"answer_generation: started — model={model_name}")
    try:
        client = wrap_openai(OpenAI(api_key=settings.openai_api_key))
        response = _call_help_llm(
            client=client,
            model=model_name,
            prompt=prompt,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            frequency_penalty=float(sampling["frequency_penalty"]),
            presence_penalty=float(sampling["presence_penalty"]),
            max_tokens=int(sampling["max_tokens"]),
        )
    except Exception as exc:
        logger.error("Help Assistant answer generation failed: {}", exc)
        trace.append(f"answer_generation: failed — {exc}")
        return {
            "status": "error",
            "question": query,
            "answer_markdown": "",
            "message": "The Help Assistant could not generate an answer right now. Please try again.",
            "sources": source_rows,
            "trace": trace,
            "live_enrichment_used": bool(live_sources),
            "usage_records": [],
            **personality_metadata,
        }

    raw = response.choices[0].message.content or ""
    answer_markdown = _parse_help_answer(raw)
    usage_record = _build_help_usage_record(response=response, model=model_name)
    trace.append(f"answer_generation: completed — {len(answer_markdown)} chars")
    if live_failures:
        trace.append(f"live_enrichment: {len(live_failures)} fetch failure(s) preserved as non-fatal")

    return {
        "status": "answered",
        "question": query,
        "answer_markdown": answer_markdown,
        "message": "",
        "sources": source_rows,
        "trace": trace,
        "live_enrichment_used": bool(live_sources),
        "usage_records": [usage_record],
        **personality_metadata,
    }


def _normalize_query(query: str) -> str:
    """Normalize a user query for keyword matching."""
    return re.sub(r"\s+", " ", query.strip().lower())


def _tokenize_query(query: str) -> list[str]:
    """Split a normalized query into alpha tokens for tolerant intent matching."""
    return re.findall(r"[a-z]+", _normalize_query(query))


def _contains_close_token(tokens: list[str], target: str, cutoff: float = 0.8) -> bool:
    """Return whether a token list contains a close-enough match for a target token."""
    if target in tokens:
        return True
    return bool(get_close_matches(target, tokens, n=1, cutoff=cutoff))


def _is_follow_up_query(query: str) -> bool:
    """Return whether the query is a chat follow-up that depends on prior turns."""
    lower = _normalize_query(query)
    if not lower:
        return False
    return (
        any(pattern in lower for pattern in _FOLLOW_UP_PATTERNS)
        or _is_last_question_query(lower)
        or _is_last_answer_query(lower)
        or _is_live_enrichment_follow_up_query(lower)
    )


def _is_last_question_query(query: str) -> bool:
    """Return whether the follow-up explicitly asks for the prior user question."""
    lower = _normalize_query(query)
    if "last question" in lower:
        return True

    tokens = _tokenize_query(lower)
    return _contains_close_token(tokens, "question") and _contains_close_token(tokens, "last")


def _is_last_answer_query(query: str) -> bool:
    """Return whether the follow-up explicitly asks for the prior assistant answer."""
    lower = _normalize_query(query)
    return "what did you just say" in lower or "what did you say" in lower or "last answer" in lower


def _is_live_enrichment_follow_up_query(query: str) -> bool:
    """Return whether the query asks about prior live-doc usage or source provenance."""
    lower = _normalize_query(query)
    if not lower:
        return False

    mentions_live_docs = any(
        phrase in lower
        for phrase in (
            "live official docs",
            "live docs enrichment",
            "official snapshots and live docs enrichment",
        )
    )
    mentions_prior_answer = any(
        phrase in lower
        for phrase in (
            "that question",
            "that answer",
            "to answer that",
            "for that answer",
            "did you use",
        )
    )
    return mentions_live_docs and mentions_prior_answer


def _resolve_help_context_query(query: str, history: list[dict[str, Any]]) -> str:
    """Use the last substantive user question to ground ambiguous follow-ups."""
    if not history or not _is_follow_up_query(query):
        return query
    last_question = _get_last_history_question(history)
    return last_question or query


def _get_last_history_question(history: list[dict[str, Any]]) -> str:
    """Return the most recent prior user question from Help Assistant history."""
    for turn in reversed(history):
        question = str(turn.get("question", "")).strip()
        if question:
            return question
    return ""


def _get_last_history_answer(history: list[dict[str, Any]]) -> str:
    """Return the most recent prior assistant answer or refusal text."""
    for turn in reversed(history):
        answer = str(turn.get("answer_markdown") or turn.get("message") or "").strip()
        if answer:
            return answer
    return ""


def _get_last_history_turn(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest Help Assistant turn when available."""
    return history[-1] if history else None


def _answer_from_history_if_possible(
    query: str,
    history: list[dict[str, Any]],
    trace: list[str],
    *,
    personality_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Answer simple meta follow-ups directly from session chat history when possible."""
    if not history:
        return None

    if _is_last_question_query(query):
        last_question = _get_last_history_question(history)
        if not last_question:
            return None
        trace.append("follow_up_answer: answered from chat history — last question")
        return {
            "status": "answered",
            "question": query,
            "answer_markdown": f"Your last Help Assistant question was: **{last_question}**",
            "message": "",
            "sources": [],
            "trace": trace,
            "live_enrichment_used": False,
            "usage_records": [],
            **personality_metadata,
        }

    if _is_last_answer_query(query):
        last_answer = _compact_text_excerpt(_get_last_history_answer(history), max_chars=500)
        if not last_answer:
            return None
        trace.append("follow_up_answer: answered from chat history — last answer")
        return {
            "status": "answered",
            "question": query,
            "answer_markdown": f"My last answer was:\n\n{last_answer}",
            "message": "",
            "sources": [],
            "trace": trace,
            "live_enrichment_used": False,
            "usage_records": [],
            **personality_metadata,
        }

    if _is_live_enrichment_follow_up_query(query):
        last_turn = _get_last_history_turn(history)
        if not last_turn:
            return None
        used_live_docs = bool(last_turn.get("live_enrichment_used"))
        source_rows = list(last_turn.get("sources", []))
        if used_live_docs:
            answer = (
                "Yes. My last answer used approved live official docs enrichment in addition to "
                "grounded KB or official-snapshot context."
            )
        else:
            answer = (
                "No. My last answer did not use live official docs enrichment. "
                "It stayed within grounded KB or official-snapshot context."
            )
        trace.append("follow_up_answer: answered from chat history — live enrichment usage")
        return {
            "status": "answered",
            "question": query,
            "answer_markdown": answer,
            "message": "",
            "sources": source_rows,
            "trace": trace,
            "live_enrichment_used": used_live_docs,
            "usage_records": [],
            **personality_metadata,
        }

    return None


def _fetch_live_context(
    selected_sources: list[dict[str, Any]],
    fetcher,
    trace: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    """Fetch live context from the approved source registry."""
    live_sources: list[dict[str, str]] = []
    failures: list[str] = []

    for source in selected_sources:
        source_name = str(source["name"])
        source_domain = str(source["domain"])
        fetched = False
        for url in list(source["urls"])[:1]:
            fetch_result = _normalize_live_fetch(fetcher(url), fallback_url=url)
            if not fetch_result["ok"] or not str(fetch_result["content"]).strip():
                failures.append(_format_live_failure(fetch_result))
                continue

            snippet = _clean_live_preview_text(
                preview_external_doc_content(str(fetch_result["content"]), max_chars=700)
            )
            if not snippet.strip():
                failures.append(f"{urlparse(url).netloc or url} (empty preview)")
                continue

            live_sources.append(
                {
                    "title": source_name,
                    "kind": "live_official_docs",
                    "domain": source_domain,
                    "url": str(fetch_result.get("final_url") or url),
                    "snippet": snippet,
                }
            )
            fetched = True
            break

        if fetched:
            trace.append(f"live_fetch: {source_name} fetched")
        else:
            trace.append(f"live_fetch: {source_name} failed")

    return live_sources, failures


def _normalize_live_fetch(result, *, fallback_url: str) -> dict[str, Any]:
    """Normalize one live fetch response into a consistent structured payload."""
    if isinstance(result, dict):
        payload = dict(result)
        payload.setdefault("ok", bool(payload.get("content")))
        payload.setdefault("content", None)
        payload.setdefault("error", None if payload.get("ok") else "no response")
        payload.setdefault("final_url", fallback_url)
        payload.setdefault("source_url", fallback_url)
        return payload
    if isinstance(result, str):
        return {
            "ok": True,
            "content": result,
            "error": None,
            "final_url": fallback_url,
            "source_url": fallback_url,
        }
    return {
        "ok": False,
        "content": None,
        "error": "no response",
        "final_url": fallback_url,
        "source_url": fallback_url,
    }


def _format_live_failure(fetch_result: dict[str, Any]) -> str:
    """Return one concise failure reason for a live source fetch."""
    url = str(fetch_result.get("source_url") or fetch_result.get("final_url") or "URL")
    domain = urlparse(url).netloc or url
    error = str(fetch_result.get("error") or "unknown error").strip()
    return f"{domain} ({error})"


def _clean_live_preview_text(text: str) -> str:
    """Return a cleaner live-doc preview when fetched HTML reduces to navigation noise."""
    snippet = _compact_text_excerpt(text, max_chars=700)
    if not snippet:
        return "Live source fetched; readable preview unavailable."

    lower = snippet.lower()
    if lower in {"redirecting", "redirecting...", "redirecting…"}:
        return "Live source fetched; readable preview unavailable."

    noisy_markers = (
        "redirecting",
        "skip to content",
        "table of contents",
        "toggle navigation",
        "main navigation",
        "breadcrumb",
    )
    marker_hits = sum(1 for marker in noisy_markers if marker in lower)
    if marker_hits >= 2:
        return "Live source fetched; readable preview unavailable."
    return snippet


def _normalize_help_runtime_config(
    runtime_config: dict[str, float | int] | None,
    *,
    personality_mode: str,
) -> dict[str, float | int]:
    """Merge a runtime override with the selected Help Assistant preset defaults."""
    normalized = get_help_assistant_runtime_defaults(personality_mode)
    if not runtime_config:
        return normalized

    for key in normalized:
        if key in runtime_config and runtime_config[key] is not None:
            normalized[key] = runtime_config[key]
    return normalized


def _is_custom_help_runtime_config(
    personality_mode: str,
    runtime_config: dict[str, float | int],
) -> bool:
    """Return whether the runtime config differs from the selected preset defaults."""
    defaults = get_help_assistant_runtime_defaults(personality_mode)
    return any(runtime_config[key] != defaults[key] for key in defaults)


def _build_help_personality_metadata(
    personality_mode: str,
    runtime_config: dict[str, float | int],
) -> dict[str, Any]:
    """Build stored personality metadata for one Help Assistant turn."""
    is_custom = _is_custom_help_runtime_config(personality_mode, runtime_config)
    return {
        "personality_mode": personality_mode,
        "personality_label": "Custom" if is_custom else personality_mode,
        "runtime_is_custom": is_custom,
        "runtime_config": dict(runtime_config),
    }


def _build_source_rows(
    curated_docs: list[Document],
    official_docs: list[Document],
    live_sources: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build display-ready source rows for the Help Assistant page."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for doc in curated_docs[:4]:
        row = _document_to_source_row(doc, kind="curated_kb")
        key = (row["Title"], row["Kind"], row["Location"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    for doc in official_docs[:4]:
        row = _document_to_source_row(doc, kind="official_snapshot")
        key = (row["Title"], row["Kind"], row["Location"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    for source in live_sources:
        row = {
            "Title": source["title"],
            "Kind": _SOURCE_KIND_LABELS["live_official_docs"],
            "Domain": source["domain"],
            "Location": source["url"],
            "Snippet": source["snippet"],
        }
        key = (row["Title"], row["Kind"], row["Location"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows


def _document_to_source_row(doc: Document, *, kind: str) -> dict[str, str]:
    """Convert a retrieved local document into a display-ready source row."""
    metadata = doc.metadata or {}
    title = (
        metadata.get("topic")
        or metadata.get("filename")
        or metadata.get("source")
        or "Source"
    )
    domain = str(metadata.get("domain") or infer_domain(str(metadata.get("filename", ""))))
    location = str(metadata.get("filename") or metadata.get("source") or "local")
    snippet = _compact_text_excerpt(doc.content, max_chars=500)
    return {
        "Title": str(title),
        "Kind": _SOURCE_KIND_LABELS[kind],
        "Domain": domain,
        "Location": location,
        "Snippet": snippet,
    }


def _compact_text_excerpt(text: str, max_chars: int = 500) -> str:
    """Compact markdown or plain text into a short readable prompt/source excerpt."""
    if not text:
        return ""
    compact = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    compact = re.sub(r"```.*?```", " ", compact, flags=re.DOTALL)
    compact = re.sub(r"`([^`]*)`", r"\1", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) > max_chars:
        return compact[: max_chars - 3].rstrip() + "..."
    return compact


def _build_help_assistant_prompt(
    *,
    question: str,
    context_question: str | None = None,
    curated_docs: list[Document],
    official_docs: list[Document],
    live_sources: list[dict[str, str]],
    conversation_history: list[dict[str, Any]],
    personality_mode: str,
) -> str:
    """Build the Help Assistant grounding prompt."""
    effective_question = context_question or question
    normalized_mode = normalize_help_assistant_personality_mode(personality_mode)
    curated_block = _format_prompt_documents(curated_docs, kind_label="Curated KB")
    official_block = _format_prompt_documents(official_docs, kind_label="Official Snapshot")
    live_block = _format_prompt_live_sources(live_sources)
    follow_up_block = _format_conversation_context(conversation_history)
    style_block = _format_personality_mode_instruction(normalized_mode)
    technical_question_instruction = ""
    if normalized_mode == "Technical":
        if classify_technical_help_question(effective_question) == "product_overview":
            technical_question_instruction = (
                "Technical question category: product_overview.\n"
                "- Give a compact technical overview of how the system is composed.\n"
                "- Start from the architectural split, runtime boundary, or concrete mechanism directly.\n"
                "- Avoid generic definitional openings.\n"
                "- Mention major flows or surfaces only when they help explain the architecture.\n"
                "- Keep the answer concise and engineering-oriented, not like onboarding docs.\n"
            )
        else:
            technical_question_instruction = (
                "Technical question category: runtime_architecture.\n"
                "- Start from the concrete mechanism, execution boundary, or orchestration behavior directly.\n"
                "- Avoid generic definitional openings.\n"
                "- Lead with routing, orchestration, state flow, retrieval boundaries, observability, or failure behavior.\n"
                "- Explain why the runtime chooses that path and what to inspect when it goes wrong.\n"
            )
    app_question_instruction = (
        (
            "This is an app-workflow question. Answer primarily from the App Workflow Context below. "
            "For Technical mode, follow the routed question category below and explain either the compact system overview "
            "or the underlying runtime behavior as appropriate. "
            "Only reference generic AI-engineering concepts when they directly explain this app. "
            "The answer body must be natural Markdown prose, not a JSON object.\n"
        )
        if is_app_help_query(effective_question) and normalized_mode == "Technical"
        else (
            "This is an app-workflow question. Answer primarily from the App Workflow Context below. "
            "Name the relevant sections directly: Learn, Quiz, Progress, Dashboard, Official Docs Sync, "
            "Help Assistant, Runtime Info, RAGAs Evaluation, and KB Index. "
            "Only reference generic AI-engineering concepts when they directly explain this app. "
            "The answer body must be natural Markdown prose, not a JSON object.\n"
            if is_app_help_query(effective_question)
            else ""
        )
    )
    follow_up_instruction = (
        "This is a follow-up question. Use the recent conversation first, then the grounded context below.\n"
        if conversation_history and _is_follow_up_query(question)
        else ""
    )

    return (
        "You are the Help Assistant for the AI Engineering Learning App.\n\n"
        "Your scope is narrow:\n"
        "- Answer only AI engineering, RAG, agents, evaluation, LangChain/LangGraph, "
        "OpenAI API, Chroma, Streamlit, Pydantic, LangSmith, observability, "
        "tool-calling, prompt-engineering, and closely related app-architecture questions.\n"
        "- Use ONLY the grounded context provided below.\n"
        "- If the context is incomplete, say so plainly instead of inventing details.\n"
        "- Prefer practical, production-oriented explanations over generic chatbot phrasing.\n"
        "- Keep the answer concise but substantive.\n"
        "- Do NOT include a `Sources`, `Sources used`, or bibliography section in the answer.\n"
        "- Inside `answer_markdown`, NEVER answer as JSON, a dict, a schema, or a key/value object.\n"
        "- Inside `answer_markdown`, ALWAYS answer as natural markdown prose with normal sentences, bullets, and sections.\n"
        "- The UI renders source provenance separately.\n\n"
        f"{style_block}\n\n"
        f"{technical_question_instruction}"
        f"Question:\n{question}\n\n"
        f"{follow_up_instruction}"
        f"{app_question_instruction}"
        f"{_APP_WORKFLOW_CONTEXT}\n\n"
        f"{follow_up_block}\n\n"
        f"{curated_block}\n\n"
        f"{official_block}\n\n"
        f"{live_block}\n\n"
        "Return valid JSON with this shape:\n"
        '{"answer_markdown": "..."}\n'
        "Return ONLY the JSON object."
    )


def _format_prompt_documents(docs: list[Document], *, kind_label: str) -> str:
    """Format local retrieved documents into a bounded prompt block."""
    if not docs:
        return f"{kind_label}:\n- None"

    lines = [f"{kind_label}:"]
    for idx, doc in enumerate(docs[:4], 1):
        meta = doc.metadata or {}
        title = meta.get("topic") or meta.get("filename") or f"{kind_label} {idx}"
        snippet = _compact_text_excerpt(doc.content, max_chars=700)
        lines.append(f"- {title}: {snippet}")
    return "\n".join(lines)


def _format_personality_mode_instruction(mode: str) -> str:
    """Format a short style instruction block for the selected personality mode."""
    normalized = normalize_help_assistant_personality_mode(mode)
    return HELP_ASSISTANT_PERSONALITY_PROFILES[normalized]["prompt_contract"]


def _format_prompt_live_sources(live_sources: list[dict[str, str]]) -> str:
    """Format live official-doc snippets into a bounded prompt block."""
    if not live_sources:
        return "Live Official Docs:\n- None"

    lines = ["Live Official Docs:"]
    for source in live_sources[:3]:
        lines.append(
            f"- {source['title']} ({source['url']}): {source['snippet']}"
        )
    return "\n".join(lines)


def _call_help_llm(
    *,
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
) -> Any:
    """Execute the Help Assistant LLM call with retry."""

    def _llm_call() -> Any:
        # TODO: True token streaming would require the OpenAI streaming API plus
        # incremental Streamlit UI updates instead of this single atomic response call.
        return client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded AI engineering support assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )

    return with_retry(
        callable=_llm_call,
        max_attempts=2,
        base_delay=1.0,
        handled_exceptions=(Exception,),
    )


def _parse_help_answer(raw: str) -> str:
    """Parse the model response, accepting JSON or plain markdown fallback."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _coerce_help_answer_to_markdown(text)
    answer = str(payload.get("answer_markdown", "")).strip()
    return _coerce_help_answer_to_markdown(answer or text)


def _build_help_usage_record(*, response: Any, model: str) -> dict[str, Any]:
    """Build a usage record for one Help Assistant model call."""
    usage = getattr(response, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    return build_usage_record(model, "help_assistant_answer", usage_dict)


def _format_conversation_context(history: list[dict[str, Any]], max_turns: int = 5) -> str:
    """Format lightweight follow-up context from recent Help Assistant turns."""
    if not history:
        return "Recent conversation:\n- None"

    lines = ["Recent conversation:"]
    for turn in history[-max_turns:]:
        question = str(turn.get("question", "")).strip()
        answer = str(turn.get("answer_markdown") or turn.get("message") or "").strip()
        answer = _compact_text_excerpt(answer, max_chars=280)
        if question:
            lines.append(f"- User: {question}")
        if answer:
            lines.append(f"  Assistant: {answer}")
    return "\n".join(lines)


def _strip_generated_sources_section(answer_markdown: str) -> str:
    """Remove model-added source sections; the UI renders provenance separately."""
    if not answer_markdown:
        return ""

    patterns = [
        r"\n#{1,6}\s+Sources(?: used)?\s*\n.*$",
        r"\n\*\*Sources(?: used)?\*\*\s*\n.*$",
        r"\nSources(?: used)?\s*\n(?:[-*].*\n?)+$",
    ]
    cleaned = answer_markdown
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()
