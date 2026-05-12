"""Help Assistant page renderer."""

from __future__ import annotations

import re

import streamlit as st

from src.services.help_assistant import (
    answer_help_question,
    get_help_assistant_personality_modes,
    get_help_assistant_runtime_defaults,
    get_help_assistant_scope,
)
from src.ui.shared import _accumulate_usage_records


def queue_help_assistant_question(prompt: str) -> None:
    """Prefill the Help Assistant input and switch to that page."""
    st.session_state["help_assistant_question"] = prompt
    st.session_state["active_section"] = "Help Assistant"
    st.rerun()


def _sync_help_assistant_draft() -> None:
    """Apply any queued prefill/reset value before the text widget is created."""
    if st.session_state.pop("help_assistant_reset_draft", False):
        st.session_state["help_assistant_draft_question"] = ""
    queued_value = st.session_state.pop("help_assistant_question", None)
    if "help_assistant_draft_question" not in st.session_state:
        st.session_state["help_assistant_draft_question"] = ""
    if queued_value is not None:
        st.session_state["help_assistant_draft_question"] = queued_value


def _get_help_chat_history() -> list[dict]:
    """Return the current session Help Assistant chat history."""
    return list(st.session_state.get("help_assistant_chat_history", []))


def _append_help_chat_turn(turn: dict) -> None:
    """Append one Help Assistant turn to session state."""
    history = _get_help_chat_history()
    history.append(turn)
    st.session_state["help_assistant_chat_history"] = history


def _clear_help_chat() -> None:
    """Clear Help Assistant chat state for the current session."""
    st.session_state["help_assistant_chat_history"] = []
    st.session_state["help_assistant_question"] = ""
    st.session_state["help_assistant_reset_draft"] = True


def _ensure_help_runtime_settings() -> None:
    """Initialize Help Assistant runtime settings from the current personality preset."""
    mode = st.session_state.get("help_assistant_personality_mode", "Technical")
    defaults = get_help_assistant_runtime_defaults(mode)
    st.session_state.setdefault("help_assistant_temperature", float(defaults["temperature"]))
    st.session_state.setdefault("help_assistant_top_p", float(defaults["top_p"]))
    st.session_state.setdefault("help_assistant_frequency_penalty", float(defaults["frequency_penalty"]))
    st.session_state.setdefault("help_assistant_presence_penalty", float(defaults["presence_penalty"]))
    st.session_state.setdefault("help_assistant_max_tokens", int(defaults["max_tokens"]))


def _apply_help_style_preset(mode: str) -> None:
    """Apply one Help Assistant personality preset and its runtime defaults."""
    st.session_state["help_assistant_personality_mode"] = mode
    defaults = get_help_assistant_runtime_defaults(mode)
    st.session_state["help_assistant_temperature"] = float(defaults["temperature"])
    st.session_state["help_assistant_top_p"] = float(defaults["top_p"])
    st.session_state["help_assistant_frequency_penalty"] = float(defaults["frequency_penalty"])
    st.session_state["help_assistant_presence_penalty"] = float(defaults["presence_penalty"])
    st.session_state["help_assistant_max_tokens"] = int(defaults["max_tokens"])


def _get_help_runtime_config() -> dict[str, float | int]:
    """Return the current Help Assistant runtime config from session state."""
    _ensure_help_runtime_settings()
    return {
        "temperature": float(st.session_state["help_assistant_temperature"]),
        "top_p": float(st.session_state["help_assistant_top_p"]),
        "frequency_penalty": float(st.session_state["help_assistant_frequency_penalty"]),
        "presence_penalty": float(st.session_state["help_assistant_presence_penalty"]),
        "max_tokens": int(st.session_state["help_assistant_max_tokens"]),
    }


def _help_runtime_is_modified() -> bool:
    """Return whether the current runtime settings differ from the selected preset defaults."""
    mode = st.session_state.get("help_assistant_personality_mode", "Technical")
    defaults = get_help_assistant_runtime_defaults(mode)
    current = _get_help_runtime_config()
    return any(current[key] != defaults[key] for key in defaults)


def _format_help_runtime_summary(runtime_config: dict[str, float | int]) -> str:
    """Return a compact runtime sampling summary for reviewer-facing UI."""
    return (
        f"Temperature={runtime_config['temperature']} | "
        f"Top-p={runtime_config['top_p']} | "
        f"Frequency penalty={runtime_config['frequency_penalty']} | "
        f"Presence penalty={runtime_config['presence_penalty']} | "
        f"Max tokens={runtime_config['max_tokens']}"
    )


def _split_help_sources(source_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split Help Assistant sources into grounded KB and live enrichment groups."""
    grounded = [
        row for row in source_rows
        if row.get("Kind") in {"Curated KB", "Official Snapshot"}
    ]
    live = [
        row for row in source_rows
        if row.get("Kind") == "Live Official Docs"
    ]
    return grounded, live


def _render_source_group(rows: list[dict[str, str]], *, title: str, caption: str) -> None:
    """Render one grouped source expander."""
    if not rows:
        return

    with st.expander(f"{title} ({len(rows)})"):
        st.caption(caption)
        table_rows = [
            {
                "Title": row["Title"],
                "Kind": row["Kind"],
                "Domain": row["Domain"],
                "Location": row["Location"],
            }
            for row in rows
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        st.markdown("**Source previews**")
        for row in rows:
            st.markdown(
                f"- **{row['Title']}** ({row['Kind']}) — "
                f"{_compact_help_preview(row['Snippet']) or '_No preview available._'}"
            )
            if row.get("Location"):
                st.caption(f"Location: {row['Location']}")


def _render_help_sources(source_rows: list[dict[str, str]]) -> None:
    """Render grouped Help Assistant provenance."""
    grounded_rows, live_rows = _split_help_sources(source_rows)

    if not grounded_rows and not live_rows:
        return

    _render_source_group(
        grounded_rows,
        title="Grounded KB sources",
        caption="Curated KB and official-doc snapshot sources used for this answer.",
    )
    if live_rows:
        _render_source_group(
            live_rows,
            title="Live official docs enrichment",
            caption="Approved live official-doc snippets fetched for extra freshness.",
        )


def _get_help_context_panel(turn: dict) -> tuple[str, str, list[str]] | None:
    """Return a fallback provenance panel when no retrieved source rows exist."""
    if turn.get("sources"):
        return None

    trace = [str(entry).lower() for entry in turn.get("trace", [])]
    if any("chat history" in entry for entry in trace):
        return (
            "Conversation context",
            "This answer was resolved from the recent Help Assistant session context.",
            [
                "Used the last five Help Assistant turns kept in session memory.",
                "No KB retrieval or live official-doc fetch was needed for this follow-up.",
            ],
        )

    return (
        "App workflow context",
        "This answer was grounded in the app's built-in workflow and reviewer-facing system context.",
        [
            "Uses the app workflow context for Learn, Quiz, Progress, Dashboard, Runtime Info, Official Docs Sync, RAGAs Evaluation, KB Index, and Help Assistant.",
            "No KB retrieval or live official-doc fetch was needed for this answer.",
        ],
    )


def _render_help_context_panel(turn: dict) -> None:
    """Render fallback provenance for successful answers without retrieved source rows."""
    panel = _get_help_context_panel(turn)
    if not panel:
        return

    title, caption, bullets = panel
    with st.expander(title):
        st.caption(caption)
        for bullet in bullets:
            st.markdown(f"- {bullet}")


def _render_help_style_selector() -> None:
    """Render horizontal style-selection buttons for Help Assistant."""
    current_mode = st.session_state.get("help_assistant_personality_mode", "Technical")
    modes = get_help_assistant_personality_modes()
    mode_cols = st.columns(len(modes))
    for idx, mode in enumerate(modes):
        button_type = "primary" if mode == current_mode else "secondary"
        if mode_cols[idx].button(
            mode,
            key=f"help_assistant_style_{mode.lower()}",
            type=button_type,
            use_container_width=True,
        ):
            if current_mode != mode:
                _apply_help_style_preset(mode)
                st.rerun()


def _render_help_advanced_settings() -> None:
    """Render collapsible runtime controls for Help Assistant model sampling."""
    runtime_config = _get_help_runtime_config()
    with st.expander("Advanced Model Settings"):
        if _help_runtime_is_modified():
            st.info("Preset modified manually.")

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(runtime_config["temperature"]),
            step=0.05,
            key="help_assistant_temperature",
            help="Controls creativity/randomness. Lower values are more deterministic; higher values are more creative.",
        )
        st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=float(runtime_config["top_p"]),
            step=0.05,
            key="help_assistant_top_p",
            help="Controls nucleus sampling. Lower values restrict token selection to higher-probability tokens.",
        )
        st.slider(
            "Frequency penalty",
            min_value=0.0,
            max_value=2.0,
            value=float(runtime_config["frequency_penalty"]),
            step=0.1,
            key="help_assistant_frequency_penalty",
            help="Reduces repetition by penalizing tokens that already appeared frequently.",
        )
        st.slider(
            "Presence penalty",
            min_value=0.0,
            max_value=2.0,
            value=float(runtime_config["presence_penalty"]),
            step=0.1,
            key="help_assistant_presence_penalty",
            help="Encourages introducing new concepts instead of repeating existing topics.",
        )
        st.slider(
            "Max tokens",
            min_value=400,
            max_value=1800,
            value=int(runtime_config["max_tokens"]),
            step=50,
            key="help_assistant_max_tokens",
            help="Maximum length of the generated response.",
        )


def _get_help_answer_caption(turn: dict) -> str:
    """Return the right reviewer-facing caption for one Help Assistant answer."""
    personality_label = str(
        turn.get("personality_label")
        or turn.get("personality_mode")
        or "Technical"
    )
    question = str(turn.get("question", "")).lower()

    if turn.get("live_enrichment_used"):
        base = "Live official docs enrichment was used for this answer."
        return f"{base} | Agent Personality: {personality_label}"

    source_rows = turn.get("sources", [])
    if source_rows:
        base = "Answered from local curated and official-doc snapshot context."
        return f"{base} | Agent Personality: {personality_label}"

    trace = [str(entry).lower() for entry in turn.get("trace", [])]
    if any("chat history" in entry for entry in trace):
        base = "Answered from session chat history."
        return f"{base} | Agent Personality: {personality_label}"

    if "when does help assistant use live official docs" in question:
        base = "Answered from app workflow / live-docs policy context."
    else:
        base = "Answered from app workflow context."
    return f"{base} | Agent Personality: {personality_label}"


def _format_help_trace_entries(trace: list[str]) -> list[str]:
    """Return reviewer-friendly Help Assistant trace labels."""
    friendly: list[str] = []
    seen: set[str] = set()

    def _append_once(key: str, message: str) -> None:
        if key in seen:
            return
        seen.add(key)
        friendly.append(message)

    for entry in trace:
        text = str(entry)
        lower = text.lower()

        if "validate_scope: refused" in lower:
            _append_once("scope", "Scope check: refused as out of domain")
        elif "validate_scope:" in lower:
            _append_once("scope", "Scope check: passed")
        elif "retrieve_local_context: skipped" in lower:
            _append_once("local", "Local retrieval: skipped because this was an app-workflow question")
        elif "retrieve_local_context:" in lower and "curated=" in lower:
            _append_once("local", "Local retrieval: completed from curated KB and official snapshots")
        elif "select_live_sources: skipped" in lower:
            _append_once("live_select", "Live docs: skipped because no live source was needed")
        elif "select_live_sources:" in lower and "selected=" in lower:
            _append_once("live_select", "Live docs: selected approved official-doc sources")
        elif "live_fetch:" in lower and "fetched" in lower:
            _append_once("live_fetch", "Live docs: fetched approved official-doc context")
        elif "live_fetch:" in lower and "failed" in lower:
            _append_once("live_fetch", "Live docs: fetch attempt failed but the answer continued safely")
        elif "follow_up_context:" in lower:
            _append_once("conversation", "Conversation context: reused the prior Help Assistant turn")
        elif "follow_up_answer:" in lower:
            _append_once("conversation", "Conversation context: answered directly from session chat history")
        elif "answer_generation: started" in lower and "model=" in lower:
            _append_once("generation_start", f"Generation: started with {text.split('model=', 1)[-1]}")
        elif "answer_generation: completed" in lower:
            _append_once("generation_end", "Generation: completed")
        elif "answer_generation: failed" in lower:
            _append_once("generation_end", "Generation: failed")

    if not friendly:
        return ["No friendly trace summary available."]
    return friendly


def _format_help_execution_trace(trace: list[str]) -> list[str]:
    """Return a reviewer-facing execution lifecycle derived from backend trace events."""
    model_name: str | None = None
    curated_count: int | None = None
    official_count: int | None = None
    selected_live = 0
    fetched_live = 0
    failed_live = 0
    app_workflow_context = False
    chat_history_answer = False
    live_skipped = False
    generation_completed = False
    generation_failed = False
    scope_refused = False

    for entry in trace:
        text = str(entry)
        lower = text.lower()

        if "validate_scope: refused" in lower:
            scope_refused = True
        elif "retrieve_local_context: skipped" in lower:
            app_workflow_context = True
        elif "retrieve_local_context:" in lower and "curated=" in lower:
            curated_match = re.search(r"curated=(\d+)", text)
            official_match = re.search(r"official_snapshot=(\d+)", text)
            if curated_match:
                curated_count = int(curated_match.group(1))
            if official_match:
                official_count = int(official_match.group(1))
        elif "select_live_sources: skipped" in lower:
            live_skipped = True
        elif "select_live_sources:" in lower and "selected=" in lower:
            selected_match = re.search(r"selected=(\d+)", text)
            if selected_match:
                selected_live = int(selected_match.group(1))
        elif "live_fetch:" in lower and "fetched" in lower:
            fetched_live += 1
        elif "live_fetch:" in lower and "failed" in lower:
            failed_live += 1
        elif "live_enrichment:" in lower and "fetch failure" in lower:
            failure_match = re.search(r"(\d+) fetch failure", lower)
            if failure_match:
                failed_live = max(failed_live, int(failure_match.group(1)))
        elif "follow_up_answer:" in lower and "chat history" in lower:
            chat_history_answer = True
        elif "answer_generation: started" in lower and "model=" in lower:
            model_name = text.split("model=", 1)[-1].strip()
        elif "answer_generation: completed" in lower:
            generation_completed = True
        elif "answer_generation: failed" in lower:
            generation_failed = True

    lifecycle: list[str] = []
    if scope_refused:
        lifecycle.append("⚠ Scope validation refused")
        return lifecycle

    lifecycle.append("✓ Scope validation passed")

    if chat_history_answer:
        lifecycle.append(
            "✓ Conversation context selected\n"
            "  • Answered directly from session chat history"
        )
    elif app_workflow_context:
        lifecycle.append(
            "✓ App workflow context selected\n"
            "  • Local KB retrieval skipped\n"
            "  • Live docs skipped"
        )
    elif curated_count is not None or official_count is not None:
        lifecycle.append(
            "✓ Local KB retrieval completed\n"
            f"  • Curated KB sources: {curated_count or 0}\n"
            f"  • Official snapshot sources: {official_count or 0}"
        )

    if selected_live > 0:
        if failed_live > 0 and fetched_live > 0:
            lifecycle.append(
                "⚠ Live official-doc enrichment partially completed\n"
                f"  • Approved live sources selected: {selected_live}\n"
                f"  • Live sources fetched: {fetched_live}\n"
                f"  • Live source fetch failures: {failed_live}"
            )
        elif fetched_live > 0:
            lifecycle.append(
                "✓ Live official-doc enrichment completed\n"
                f"  • Approved live sources selected: {selected_live}\n"
                f"  • Live sources fetched: {fetched_live}"
            )
        elif failed_live > 0:
            lifecycle.append(
                "⚠ Live official-doc enrichment failed\n"
                f"  • Approved live sources selected: {selected_live}\n"
                f"  • Live source fetch failures: {failed_live}"
            )
    elif live_skipped and not app_workflow_context:
        lifecycle.append("✓ Live official-doc enrichment skipped")

    if generation_completed:
        details = f"\n  • Model: {model_name}" if model_name else ""
        lifecycle.append(f"✓ Response generated{details}")
    elif generation_failed:
        details = f"\n  • Model: {model_name}" if model_name else ""
        lifecycle.append(f"⚠ Response generation failed{details}")

    return lifecycle or ["No execution trace available."]


def _render_help_turn(turn: dict) -> None:
    """Render one user/assistant turn pair."""
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(turn.get("question", ""))

    with st.chat_message("assistant", avatar="🤖"):
        status = turn.get("status")
        if status == "refused":
            st.warning(turn.get("message", "Question refused."))
        elif status == "error":
            st.error(turn.get("message", "The Help Assistant could not answer right now."))
        else:
            st.markdown(turn.get("answer_markdown", ""))
            st.caption(_get_help_answer_caption(turn))

        _render_help_sources(turn.get("sources", []))
        if status == "answered":
            _render_help_context_panel(turn)

        with st.expander("Request Trace"):
            for entry in _format_help_trace_entries(turn.get("trace", [])):
                st.markdown(f"- {entry}")
        with st.expander("Execution Trace"):
            for entry in _format_help_execution_trace(turn.get("trace", [])):
                st.markdown(entry.replace("\n", "  \n"))
            with st.expander("Raw debug events"):
                for entry in turn.get("trace", []):
                    st.text(entry)


def _compact_help_preview(text: str, max_chars: int = 220) -> str:
    """Keep source preview lines compact so the grouped expanders stay readable."""
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) > max_chars:
        return compact[: max_chars - 3].rstrip() + "..."
    return compact


def _validate_help_submit(raw_question: str | None) -> tuple[str, str | None, str | None]:
    """Validate the submitted Help Assistant question."""
    if raw_question is None:
        return "", "Something went wrong. Please try again.", "error"
    if not isinstance(raw_question, str):
        return "", "Something went wrong. Please try again.", "error"

    question = raw_question.strip()
    if not question:
        return "", "Enter a question before submitting.", "warning"
    return question, None, None


def render_help_assistant() -> None:
    """Render the advanced scoped Help Assistant page."""
    st.header("Help Assistant")
    st.markdown(
        "Scoped AI engineering assistant for this app. It answers in-domain questions using "
        "curated KB content, official-doc snapshots, and approved live official docs when relevant."
    )
    st.info(
        "In-domain only. Live enrichment is limited to the approved official source registry and does not crawl arbitrary websites."
    )

    scope = get_help_assistant_scope()
    supported = ", ".join(scope["approved_domains"])
    st.caption(f"Approved live domains: {supported}")

    st.session_state.setdefault("help_assistant_chat_history", [])
    st.session_state.setdefault("help_assistant_personality_mode", "Technical")
    _sync_help_assistant_draft()
    _ensure_help_runtime_settings()

    st.caption("Agent Personality")
    _render_help_style_selector()
    _render_help_advanced_settings()

    history = _get_help_chat_history()
    if history:
        for turn in history:
            _render_help_turn(turn)
    else:
        st.caption(
            "Ask about app workflow, KB behavior, official docs usage, or scoped AI-engineering topics."
        )

    pending_answer = st.empty()
    input_feedback = st.empty()
    with st.form("help_assistant_form"):
        draft_question = st.text_area(
            "Question",
            key="help_assistant_draft_question",
            height=120,
            placeholder=(
                "Ask about app workflow, LangGraph state, agentic RAG, tool calling, "
                "evaluation, observability, official docs usage, or related AI engineering topics."
            ),
        )
        action_cols = st.columns([7, 3])
        with action_cols[0]:
            submit = st.form_submit_button(
                "Ask Help Assistant",
                key="help_assistant_submit",
                type="primary",
                use_container_width=True,
            )
        with action_cols[1]:
            clear_chat = st.form_submit_button(
                "Clear chat",
                key="help_assistant_clear_chat",
                use_container_width=True,
            )

    if clear_chat:
        _clear_help_chat()
        st.rerun()

    if submit:
        question, feedback_message, feedback_level = _validate_help_submit(draft_question)
        if feedback_message:
            if feedback_level == "error":
                input_feedback.error(feedback_message)
            else:
                input_feedback.warning(feedback_message)
        else:
            history = _get_help_chat_history()
            with pending_answer.container():
                with st.chat_message("assistant", avatar="🤖"):
                    st.info("Grounding answer with local KB and approved live official docs...")
            result = answer_help_question(
                question,
                conversation_history=history,
                personality_mode=st.session_state.get("help_assistant_personality_mode", "Technical"),
                runtime_config=_get_help_runtime_config(),
            )
            _append_help_chat_turn(result)
            if result.get("usage_records"):
                _accumulate_usage_records(result["usage_records"])
            st.session_state["help_assistant_question"] = ""
            st.session_state["help_assistant_reset_draft"] = True
            pending_answer.empty()
            st.rerun()
