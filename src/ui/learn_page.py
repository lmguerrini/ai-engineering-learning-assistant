"""Streamlit Learn page renderer."""

import re
from typing import Iterator

import streamlit as st


from src.schemas import DifficultyLevel, StudyGuide
from src.ui.display_helpers import downgrade_headings
from src.ui.shared import (
    LEARN_TOPICS,
    _DEPTH_TO_STYLE,
    _LEARN_PATH_DISPLAY_NAMES,
    _LEARN_PATH_LABELS,
    _LEARN_PATH_STABLE_TOPICS,
    _LEARN_PATH_TO_ENUM,
    _LEARN_PATH_TOPIC_DESCRIPTIONS,
    _LEARN_PATH_TOPIC_EFFORT,
    _LEARN_PATH_TOPIC_MAP,
    _LEARNING_DEPTH_LABELS,
    _LEARNING_MODE_LABELS,
    _accumulate_usage_records,
    _display_debug_trace,
    _display_feedback_widget,
    _display_memory_section,
    _display_sources_section,
)


def _heading_to_anchor(heading: str) -> str:
    """Convert a Markdown heading to a Streamlit-compatible anchor slug.

    Handles numbered headings (``## 1. LangChain Chains``), punctuation,
    slashes, parentheses, and other special characters consistently.
    The result matches what Streamlit's markdown renderer generates as
    the ``id`` attribute for ``<h*>`` elements.
    """
    slug = heading.strip().lower()
    # Remove everything that is not alphanumeric, space, or hyphen
    slug = re.sub(r"[^a-z0-9 -]", "", slug)
    # Collapse whitespace to single hyphens
    slug = re.sub(r"\s+", "-", slug.strip())
    # Collapse multiple consecutive hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    return slug


def _show_cached_learn_result() -> None:
    """Display a previously generated learn result stored in session state."""
    result = st.session_state.get("last_learn_result")
    if not result:
        return
    depth = st.session_state.get("last_learn_depth", "Summary")
    mode = st.session_state.get("last_learn_mode", "Topic")
    _display_learn_result(result, depth=depth, mode=mode, stream=False)


def _display_learn_result(result: dict, *, depth: str, mode: str,
                          stream: bool = False) -> None:
    """Display a Learn workflow result with optional UI-only streaming replay."""
    error = result.get("error")
    if error:
        st.error(error)
        return
    guide = result.get("study_guide")
    if guide:
        _display_study_guide(guide, depth=depth, mode=mode, stream=stream)
    _display_memory_section(result)
    _display_debug_trace(result, "Learn Workflow Trace")


def _has_cache_hit(result: dict) -> bool:
    """Return whether a Learn result came from the cache."""
    trace = result.get("trace", [])
    return any("cache hit" in str(step).lower() for step in trace)


def _should_stream_learn_result(result: dict) -> bool:
    """Only fresh successful Learn generations should be streamed."""
    if not result:
        return False
    if result.get("error") or result.get("generation_failed"):
        return False
    if not result.get("study_guide"):
        return False
    return not _has_cache_hit(result)


def _iter_markdown_blocks(text: str) -> Iterator[str]:
    """Yield Markdown in paragraph-safe blocks while keeping fenced code intact."""
    if not text:
        return

    lines = text.splitlines(keepends=True)
    block: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            block.append(line)
            if not in_code_fence:
                yield "".join(block)
                block = []
            continue

        block.append(line)
        if in_code_fence:
            continue

        if stripped == "":
            if any(part.strip() for part in block):
                yield "".join(block)
            block = []

    if block and any(part.strip() for part in block):
        yield "".join(block)


def _stream_markdown(text: str, *, unsafe_allow_html: bool = False) -> None:
    """Replay final Markdown incrementally in the UI without changing the result."""
    if not text:
        return

    placeholder = st.empty()
    rendered = ""
    for chunk in _iter_markdown_blocks(text):
        rendered += chunk
        placeholder.markdown(rendered, unsafe_allow_html=unsafe_allow_html)

    placeholder.markdown(text, unsafe_allow_html=unsafe_allow_html)


# ---------------------------------------------------------------------------
# Learn helpers
# ---------------------------------------------------------------------------

def _display_study_guide(guide: StudyGuide, depth: str = "Deep Study",
                         mode: str = "Topic", stream: bool = False) -> None:
    """Render a structured Learn Path in the Streamlit UI."""
    if mode == "Learn Path":
        level_label = guide.difficulty.value.capitalize()
        display_name = _LEARN_PATH_DISPLAY_NAMES.get(level_label, level_label)
        st.subheader(display_name)
        st.caption(f"Learn Path Mode · {depth}")
    else:
        st.subheader(guide.topic)
        st.caption("Topic Mode · Deep Study")

    st.markdown("#### Overview")
    if stream:
        _stream_markdown(guide.summary)
    else:
        st.markdown(guide.summary)

    # --- Table of Contents for Topic mode ---
    if mode == "Topic" and guide.detailed_notes:
        toc_lines: list[str] = []
        for m in re.finditer(r"^##\s+(.+)$", guide.detailed_notes, re.MULTILINE):
            heading = m.group(1).strip()
            if heading.lower() == "overview":
                continue
            anchor = _heading_to_anchor(heading)
            toc_lines.append(f"- [{heading}](#{anchor})")
        if toc_lines:
            st.markdown("#### Table of Contents")
            st.markdown("\n".join(toc_lines))

    # --- Topics section (Learn Path only) ---
    if mode == "Learn Path" and depth == "Summary":
        # Deterministic Topics table for Summary
        level_label = guide.difficulty.value.capitalize()
        topics = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])
        descs = _LEARN_PATH_TOPIC_DESCRIPTIONS.get(level_label, {})
        if topics:
            st.markdown("#### Topics")
            rows = ["| # | Topic | Key Focus | Estimated Effort |"]
            rows.append("|---|-------|-----------|------------------|")
            efforts = _LEARN_PATH_TOPIC_EFFORT.get(level_label, {})
            for i, t in enumerate(topics, 1):
                effort = efforts.get(t, "~2–4 hours")
                rows.append(f"| {i} | {t} | {descs.get(t, '')} | {effort} |")
            st.markdown("\n".join(rows))
    elif mode == "Learn Path" and guide.key_concepts:
        st.markdown("#### Topics")
        if depth == "Deep Study":
            level_label = guide.difficulty.value.capitalize()
            descs = _LEARN_PATH_TOPIC_DESCRIPTIONS.get(level_label, {})
            # Build a set of actual anchors from generated headings for matching
            _heading_anchors: set[str] = set()
            if guide.detailed_notes:
                for _m in re.finditer(r"^##\s+(.+)$", guide.detailed_notes, re.MULTILINE):
                    _heading_anchors.add(_heading_to_anchor(_m.group(1).strip()))
            # Collect all topic lines into one markdown block to reduce spacing
            _topic_lines: list[str] = []
            for i, concept in enumerate(guide.key_concepts, 1):
                name = concept.split(":", 1)[0].strip() if ":" in concept else concept
                desc_text = descs.get(name, "")
                # Build candidate anchor: try "N-topic-name" first, then plain
                _candidate = _heading_to_anchor(f"{i}. {name}")
                if _candidate not in _heading_anchors:
                    _candidate = _heading_to_anchor(name)
                anchor_link = f"[{name}](#{_candidate})"
                if desc_text:
                    _topic_lines.append(f"{i}. **{anchor_link}** — {desc_text}")
                else:
                    _topic_lines.append(f"{i}. **{anchor_link}**")
            if _topic_lines:
                st.markdown("\n".join(_topic_lines))
        else:
            for i, concept in enumerate(guide.key_concepts, 1):
                if ":" in concept:
                    name, desc = concept.split(":", 1)
                    st.markdown(f"{i}. **{name.strip()}** — {desc.strip()}")
                else:
                    st.markdown(f"{i}. **{concept}**")

    if guide.detailed_notes:
        notes = _clean_generated_markdown(guide.detailed_notes, guide, depth, mode)
        notes = downgrade_headings(notes)
        # Learn Path Deep Study injects HTML <a id="..."> anchors for topic links
        _has_html = '<a id="' in notes
        if stream:
            _stream_markdown(notes, unsafe_allow_html=_has_html)
        else:
            st.markdown(notes, unsafe_allow_html=_has_html)

    st.markdown("---")
    _display_sources_section(guide)


# ---------------------------------------------------------------------------
# Deterministic markdown cleanup
# ---------------------------------------------------------------------------

def _clean_generated_markdown(text: str, guide: StudyGuide,
                              depth: str, mode: str) -> str:
    """Post-process generated markdown to fix common LLM rendering issues."""
    # 1. Remove duplicate title that matches the app-level title (Deep Study)
    if mode == "Learn Path" and depth == "Deep Study":
        level_label = guide.difficulty.value.capitalize()
        display_name = _LEARN_PATH_DISPLAY_NAMES.get(level_label, "")
        if display_name:
            # Remove leading heading that duplicates the app title
            text = re.sub(
                r"^\s*#{1,3}\s*" + re.escape(display_name) + r"\s*\n+",
                "", text, count=1,
            )

    # 2. Convert fake checkbox lines (- [ ] or - [x]) to plain bullets
    text = re.sub(r"^(\s*)- \[[ xX]\] ", r"\1- ", text, flags=re.MULTILINE)

    # 3. Remove duplicate/empty "Learn Path Overview" headings
    text = re.sub(
        r"^#{1,4}\s+Learn Path Overview\s*\n(?=\s*#{1,4}\s+Learn Path Overview)",
        "", text, flags=re.MULTILINE,
    )
    # Remove empty "Learn Path Overview" heading with no content before next heading
    text = re.sub(
        r"^(#{1,4}\s+Learn Path Overview)\s*\n+(?=#{1,4}\s)",
        "", text, flags=re.MULTILINE,
    )

    # 4. Summary: strip any Topics block from generated notes (rendered deterministically)
    if mode == "Learn Path" and depth == "Summary":
        # Remove Topics heading + table
        text = re.sub(
            r"^#{1,4}\s+Topics\s*\n(?:(?:\|.+\n)+)",
            "", text, flags=re.MULTILINE,
        )
        # Remove Topics heading + bullet/numbered list
        text = re.sub(
            r"^#{1,4}\s+Topics\s*\n(?:(?:\s*[-*\d]+\.?\s+.+\n)+)",
            "", text, flags=re.MULTILINE,
        )
        # Remove duplicate Overview heading (already rendered by app)
        text = re.sub(
            r"^#{1,4}\s+Overview\s*\n",
            "", text, count=1, flags=re.MULTILINE,
        )
        # Remove standalone "Content" heading
        text = re.sub(
            r"^#{1,4}\s+Content\s*\n",
            "", text, flags=re.MULTILINE,
        )
        # Remove paragraph immediately after Topics table that repeats the Overview
        if guide.summary:
            overview_start = guide.summary.strip()[:80]
            if overview_start:
                text = re.sub(
                    r"^" + re.escape(overview_start[:40]) + r"[^\n]*\n*",
                    "", text, count=1,
                )

    # 5. Topic mode: remove duplicate Overview section (heading + body) already
    #    rendered by the app via guide.summary.  Strip everything from the
    #    Overview heading up to (but not including) the next ## heading.
    #    NOTE: this runs BEFORE downgrade_headings, so ## is the top-level.
    if mode == "Topic":
        text = re.sub(
            r"^#{1,4}\s+Overview\s*\n(.*?)(?=^##\s|\Z)",
            "", text, count=1, flags=re.MULTILINE | re.DOTALL,
        )
        # Remove Topics heading + bullet/numbered list (rendered by app for Learn Path only)
        text = re.sub(
            r"^#{1,4}\s+Topics\s*\n(?:(?:\s*[-*\d]+\.?\s+.+\n)+)",
            "", text, flags=re.MULTILINE,
        )
        # Remove Topics heading + table
        text = re.sub(
            r"^#{1,4}\s+Topics\s*\n(?:(?:\|.+\n)+)",
            "", text, flags=re.MULTILINE,
        )

    # 6. Learn Path Deep Study: inject explicit HTML anchors before ## headings
    #    so that Topic-list links reliably jump to the right section.
    if mode == "Learn Path" and depth == "Deep Study":
        def _inject_anchor(m: re.Match) -> str:
            heading_text = m.group(1).strip()
            anchor_id = _heading_to_anchor(heading_text)
            return f'<a id="{anchor_id}"></a>\n## {heading_text}'
        text = re.sub(r"^##\s+(.+)$", _inject_anchor, text, flags=re.MULTILINE)

    # 7. Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_learn() -> None:
    """Render the Learn section with Learning Mode selection."""
    # Scoped CSS: only code blocks, no global layout override
    st.markdown(
        "<style>"
        "pre { white-space: pre-wrap !important; word-wrap: break-word !important; "
        "overflow-x: auto !important; max-width: 100% !important; "
        "box-sizing: border-box !important; }"
        "pre code { display: block !important; width: 100% !important; }"
        ".stMarkdown pre { min-width: 0 !important; width: 100% !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.header("Learn")
    st.markdown(
        "Generate a focused **Topic** study or a guided multi-topic "
        "**Learn Path** powered by Agentic RAG."
    )

    # Learning Mode selector
    learning_mode = st.selectbox(
        "Learning Mode", _LEARNING_MODE_LABELS,
        index=0, key="learn_mode",
    )

    if learning_mode == "Learn Path":
        col1, col2 = st.columns(2)
        with col1:
            learn_path = st.selectbox(
                "Learn Path",
                _LEARN_PATH_LABELS,
                format_func=lambda x: f"{x} — {_LEARN_PATH_DISPLAY_NAMES[x]}",
                index=0,
                key="learn_path",
            )
        with col2:
            depth_label = st.selectbox(
                "Learning Depth",
                _LEARNING_DEPTH_LABELS,
                index=0,
                key="learn_depth",
            )

        if depth_label == "Deep Study":
            st.info(
                "🔬 **Deep Study — Intensive Mode**\n\n"
                "This mode generates comprehensive professional-grade content "
                "with one major section per topic. "
                "It uses significantly more tokens and produces much longer output "
                "than Summary mode. Estimated cost and generation time are higher."
            )

        topic = _LEARN_PATH_TOPIC_MAP[learn_path]
        difficulty = _LEARN_PATH_TO_ENUM[learn_path]
        style = _DEPTH_TO_STYLE[depth_label]

    else:  # Topic mode
        topic = st.selectbox("Topic", LEARN_TOPICS, key="learn_topic")
        # Topic mode always uses Deep Study (detailed) internally
        depth_label = "Deep Study"
        difficulty = DifficultyLevel.INTERMEDIATE
        style = _DEPTH_TO_STYLE[depth_label]

    btn_label = "Generate Learn Path" if learning_mode == "Learn Path" else "Generate Topic"
    force_regenerate = st.checkbox(
        "Regenerate (bypass cache)",
        value=False,
        key="force_regenerate",
    )
    st.caption(
        "Cached results save tokens. Enable regeneration when testing "
        "or when you want a fresh answer."
    )
    generate = st.button(btn_label, key="btn_learn")
    displayed_result_this_run = False

    if generate:
        if learning_mode != "Learn Path":
            spinner_msg = "Generating topic — this may take a moment…"
        elif depth_label == "Deep Study":
            spinner_msg = "Generating Learning Path (Deep Study) — this may take a moment…"
        else:
            spinner_msg = "Generating Learn Path Summary…"
        with st.spinner(spinner_msg):
            from src.graphs.learn_graph import run_learn_workflow

            result = run_learn_workflow(
                topic=topic,
                difficulty=difficulty,
                style=style,
                force_regenerate=force_regenerate,
            )

        st.session_state["last_learn_result"] = result
        st.session_state["last_learn_depth"] = depth_label
        st.session_state["last_learn_mode"] = learning_mode

        guide = result.get("study_guide")
        if guide:
            st.session_state["last_study_guide"] = guide
        st.session_state["last_learn_topic"] = topic
        st.session_state["last_learn_trace"] = result.get("trace", [])
        st.session_state["last_learn_tokens"] = result.get("token_usage", {})
        _accumulate_usage_records(result.get("usage_records", []))
        if _should_stream_learn_result(result):
            _display_learn_result(
                result,
                depth=depth_label,
                mode=learning_mode,
                stream=True,
            )
            displayed_result_this_run = True

    # Display generation failure warning if applicable
    result = st.session_state.get("last_learn_result")
    if result and result.get("generation_failed"):
        st.warning(
            "⚠️ **Generation failed** — the output below is a minimal fallback.\n\n"
            "The LLM call encountered an error during generation. "
            "Please try clicking **Generate Topic** again. "
            "If the problem persists, try a different topic or check your API key."
        )

    # Show cache-hit indicator when result came from cache (zero-cost reuse)
    if result and not result.get("generation_failed"):
        if _has_cache_hit(result):
            st.caption("Cached result — no additional tokens used. Cache persists across app restarts.")

    # Stream only fresh successful generations in the current run.
    # Cached results and revisits stay instant and render from session state.
    if result and not displayed_result_this_run:
        _show_cached_learn_result()

    _display_feedback_widget("learn", st.session_state.get("last_learn_topic", ""))
