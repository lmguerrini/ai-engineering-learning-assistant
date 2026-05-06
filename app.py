"""AI Engineering Learning App — Streamlit entrypoint."""

import streamlit as st

from src.logging_config import setup_logging
from src.services.observability import configure_langsmith_tracing
from src.ui.pages import (
    render_advanced,
    render_intro,
    render_learn,
    render_progress,
    render_quiz,
)

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
        background-color: #2e7d32;
        border-color: #2e7d32;
        color: white;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #388e3c;
        border-color: #388e3c;
    }
    section[data-testid="stSidebar"] button {
        text-align: left;
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

st.sidebar.markdown("### AI Engineering Learning App")

for section_name in SECTIONS:
    is_active = st.session_state["active_section"] == section_name
    btn_type = "primary" if is_active else "secondary"
    if st.sidebar.button(section_name, key=f"nav_{section_name}", type=btn_type, use_container_width=True):
        st.session_state["active_section"] = section_name
        st.rerun()

# Compact sidebar status
st.sidebar.markdown("---")
try:
    from src.config import get_settings as _get_settings
    _s = _get_settings()
    _api_status = "Ready" if _s.openai_api_key else "Missing"
    _tracing = "On" if _s.langchain_tracing_v2 else "Off"
    st.sidebar.caption(
        f"Model: {_s.app_default_model}  \n"
        f"OpenAI: {_api_status} · LangSmith: {_tracing}"
    )
except Exception:
    pass

_usage = st.session_state.get("session_usage_records", [])
if _usage:
    _total_tokens = sum(r.get("total_tokens", 0) for r in _usage)
    st.sidebar.caption(f"Session tokens: {_total_tokens:,}")

SECTIONS[st.session_state["active_section"]]()
