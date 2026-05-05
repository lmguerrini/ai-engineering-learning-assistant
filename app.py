"""AI Engineering Learning Assistant — Streamlit entrypoint."""

import streamlit as st

from src.logging_config import setup_logging
from src.ui.pages import (
    render_advanced,
    render_intro,
    render_learn,
    render_progress,
    render_quiz,
)

setup_logging()

st.set_page_config(
    page_title="AI Engineering Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

SECTIONS = {
    "🏠 Intro / Help": render_intro,
    "📖 Learn": render_learn,
    "🧠 Quiz": render_quiz,
    "📊 Progress / Feedback": render_progress,
    "⚙️ Advanced / Debug": render_advanced,
}

st.sidebar.title("🎓 AI Learning Assistant")
selection = st.sidebar.radio("Section", list(SECTIONS.keys()))

SECTIONS[selection]()
