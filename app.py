"""AI Engineering Learning App — Streamlit entrypoint."""

import streamlit as st

from src.logging_config import setup_logging
from src.services.observability import configure_langsmith_tracing
from src.ui.pages import render_intro, render_quiz, render_progress, render_advanced
from src.ui.learn_page import render_learn

setup_logging()
configure_langsmith_tracing()

st.set_page_config(
    page_title="AI Engineering Learning App",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Scoped sidebar CSS: neutral active-button colour, left-aligned text
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #1565c0;
        border-color: #1565c0;
        color: white;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #1976d2;
        border-color: #1976d2;
    }
    section[data-testid="stSidebar"] button {
        text-align: left;
    }
    /* Global readable max-width for all page content */
    .block-container {
        max-width: 52rem;
    }
    /* Constrain code blocks and markdown content within readable width */
    .stMarkdown {
        max-width: 100%;
        overflow-wrap: break-word;
    }
    .stMarkdown pre {
        max-width: 100%;
        overflow-x: auto;
        white-space: pre;
    }
    .stMarkdown code {
        max-width: 100%;
        overflow-wrap: break-word;
    }
    .stMarkdown li pre,
    .stMarkdown li code {
        max-width: 100%;
        overflow-x: auto;
    }
    .stMarkdown ul, .stMarkdown ol {
        max-width: 100%;
        overflow: visible;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation — visible buttons, active section highlighted
# ---------------------------------------------------------------------------

SECTIONS = {
    "Home": render_intro,
    "Learn": render_learn,
    "Quiz": render_quiz,
    "Progress": render_progress,
    "Dashboard": render_advanced,
}

if "active_section" not in st.session_state:
    st.session_state["active_section"] = "Home"

st.sidebar.markdown(
    '<p style="font-size:1.5rem; font-weight:700; margin:0 0 0.5rem 0;">'
    'AI Engineering Learning App</p>',
    unsafe_allow_html=True,
)

for section_name in SECTIONS:
    is_active = st.session_state["active_section"] == section_name
    btn_type = "primary" if is_active else "secondary"
    if st.sidebar.button(section_name, key=f"nav_{section_name}", type=btn_type, use_container_width=True):
        st.session_state["active_section"] = section_name
        st.rerun()

# Compact sidebar status
st.sidebar.markdown("---")
SECTIONS[st.session_state["active_section"]]()

with st.sidebar.expander("Runtime Info", expanded=True):
    try:
        from src.config import get_settings as _get_settings
        _s = _get_settings()
        _api_status = "Ready" if _s.openai_api_key else "Missing"
        _tracing = "On" if _s.langchain_tracing_v2 else "Off"
        st.caption(f"OpenAI: {_api_status}")
        st.caption(f"LangSmith: {_tracing}")
        st.caption(f"Model: {_s.app_default_model}")
        st.caption("Input cost: $0.150000 / 1M tokens")
        st.caption("Output cost: $0.600000 / 1M tokens")
    except Exception:
        pass

    try:
        from src.kb.index_health import get_kb_index_health as _get_kb_index_health

        _kb_status = _get_kb_index_health()["status_label"]
    except Exception:
        _kb_status = "Missing"
    st.caption(f"KB Index: {_kb_status}")

    _usage = st.session_state.get("session_usage_records", [])
    _total_tokens = sum(r.get("total_tokens", 0) for r in _usage) if _usage else 0
    st.caption(f"Session tokens: {_total_tokens:,}")

    try:
        if _usage:
            from src.services.cost_tracker import aggregate_usage as _agg
            _cost = _agg(_usage)["estimated_cost_usd"]
        else:
            _cost = 0.0
    except Exception:
        _cost = 0.0
    st.caption(f"Session cost: ${_cost:.6f}")

with st.sidebar.expander("Help"):
    st.markdown(
        "**1. Learn**\n\n"
        "Study AI engineering topics through focused deep-dives or guided curricula.\n\n"
        "- **Topic** — Focused deep-dive into one subject. Generates a "
        "comprehensive reference with 10 structured sections covering "
        "theory, architecture, implementation, examples, and review.\n"
        "- **Learn Path** — Guided multi-topic curriculum at Beginner, "
        "Intermediate, or Advanced level. Summary mode gives a quick "
        "overview; Deep Study produces full professional-grade content "
        "for every topic in the path.\n"
        "- **Caching** — All generated content is cached by topic, "
        "difficulty, and prompt version. Revisiting the same configuration "
        "reuses the cached result instantly with zero API cost.\n\n"
        "---\n\n"
        "**2. Quiz**\n\n"
        "Generate RAG-grounded questions from studied topics and save results for progress tracking.\n\n"
        "- **Question generation** — RAG-grounded multiple-choice "
        "questions drawn from your studied topics.\n"
        "- **Scoring and explanations** — Instant per-question feedback "
        "with correct answer rationale.\n"
        "- **Save to memory** — Results are stored in your learner "
        "profile and feed into Progress analytics.\n\n"
        "---\n\n"
        "**3. Progress**\n\n"
        "Review quiz performance, identify weak areas, and track learning over time.\n\n"
        "- **Scores and trends** — Per-topic accuracy and score "
        "trajectories across all quiz attempts.\n"
        "- **Weak areas** — Highlights recurring knowledge gaps "
        "to guide further study.\n"
        "- **Learning profile** — Difficulty-level breakdown and "
        "cumulative performance overview.\n\n"
        "---\n\n"
        "**4. Dashboard**\n\n"
        "Monitor costs, inspect workflow traces, and review runtime diagnostics.\n\n"
        "- **Costs and token usage** — Cumulative token counts and "
        "estimated costs per operation.\n"
        "- **Workflow traces** — Full Learn and Quiz pipeline traces "
        "with latency and step details.\n"
        "- **Settings and diagnostics** — Memory profile, feedback "
        "history, and runtime configuration.\n\n"
        "---\n\n"
        "**5. LangSmith**\n\n"
        "External observability platform for end-to-end LLM tracing and debugging.\n\n"
        "- **Tracing** — End-to-end traces for every LLM call, "
        "retrieval step, and graph node.\n"
        "- **Debugging** — Latency, token counts, prompt/completion "
        "pairs, and error details per trace.\n"
        "- **Required configuration** — Set `LANGCHAIN_TRACING_V2=true` "
        "and `LANGCHAIN_API_KEY` in your `.env` file."
    )
