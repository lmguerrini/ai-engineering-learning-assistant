"""Streamlit Progress page renderer."""

import streamlit as st

from src.ui.display_helpers import format_memory_transparency
from src.ui.shared import _show_friendly_error


def render_progress() -> None:
    """Render the Progress section with learning memory data."""
    st.header("Progress")
    st.markdown("Track your studied topics, quiz scores, weak areas, and feedback.")
    st.info(
        "Progress is recorded when you complete a quiz and choose **Save** "
        "in the review step (Quiz → Submit → Save memory → Progress tracking)."
    )

    from src.memory.memory_service import get_recent_topics, get_weak_areas_summary

    recent = get_recent_topics(limit=10)
    if not recent:
        _show_friendly_error("empty_progress")
    else:
        st.subheader("Recent Learning Sessions")
        for evt in recent:
            weak_str = ", ".join(evt["weak_areas"]) if evt["weak_areas"] else "—"
            st.markdown(
                f"- **{evt['topic']}** — Score: {evt['score']:.0f}% · "
                f"Weak areas: {weak_str} · {evt['timestamp'][:10]}"
            )

        st.subheader("Weak Areas Summary")
        summary = get_weak_areas_summary()
        if summary:
            for area, count in sorted(summary.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{area}** — appeared {count} time(s)")
        else:
            st.info("No weak areas recorded yet.")

    # Memory transparency
    st.subheader("Memory Profile")
    try:
        from src.memory.memory_service import get_user_profile_summary

        profile = get_user_profile_summary()
        mem = format_memory_transparency(profile)
        if mem["loaded"]:
            if mem.get("recent_topics"):
                st.markdown("**Recent topics:** " + ", ".join(mem["recent_topics"]))
            if mem.get("weak_areas"):
                st.markdown("**Recurring weak areas:** " + ", ".join(mem["weak_areas"]))
            if mem.get("average_score") is not None:
                st.markdown(f"**Average score:** {mem['average_score']:.0f}%")
            if mem.get("suggested_focus"):
                st.markdown("**Suggested focus topics:** " + ", ".join(mem["suggested_focus"]))
        else:
            st.info(
                "Memory profile will be built automatically as you study "
                "and save quiz results."
            )
    except Exception:
        st.info("Memory profile not available.")

    # Feedback section
    st.subheader("Recent Feedback")
    from src.memory.feedback_service import get_recent_feedback, get_feedback_summary

    fb_entries = get_recent_feedback(limit=5)
    if fb_entries:
        for fb in fb_entries:
            stars = fb["rating"]
            comment = fb["comment"] if fb["comment"] else "—"
            st.markdown(
                f"- Rating: {stars}/5 — **{fb['context_type']}** / {fb['topic']} — "
                f"{comment} · {fb['timestamp'][:10]}"
            )
    else:
        st.info("No feedback recorded yet.")

    fb_summary = get_feedback_summary()
    if fb_summary.get("total_count", 0) > 0:
        st.subheader("Feedback Summary")
        st.markdown(f"- **Average rating:** {fb_summary['average_rating']}")
        st.markdown(f"- **Total feedback entries:** {fb_summary['total_count']}")
        if fb_summary.get("suggestion"):
            st.markdown(f"- **Personalization suggestion:** {fb_summary['suggestion']}")
