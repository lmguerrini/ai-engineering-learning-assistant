"""Streamlit Progress page renderer."""

import streamlit as st


def _compact_learn_title(metadata: dict, topic: str) -> str:
    """Format one Learn session title for reviewer-friendly display."""
    mode = metadata.get("learning_mode", "Learn")
    difficulty = metadata.get("difficulty", "")
    depth = metadata.get("learning_depth", "")

    if mode == "Learn Path":
        title = f"Learn Path · {difficulty} - {topic}" if difficulty else f"Learn Path · {topic}"
        if depth:
            title += f" ({depth})"
        return title

    return f"Topic · {topic}"


def _format_learning_event_display(evt: dict) -> dict[str, str]:
    """Return a compact display model for one persisted learning event."""
    metadata = evt.get("metadata", {}) or {}
    timestamp = (evt.get("timestamp") or "")[:10] or "—"
    topic = evt.get("topic", "Untitled")
    event_source = metadata.get("source", "")

    if event_source == "learn_studied":
        return {
            "event_type": "learn_studied",
            "context_title": _compact_learn_title(metadata, topic),
            "date": timestamp,
            "detail": "✅ Completed",
            "summary": metadata.get("summary", "") or "",
        }

    weak_str = ", ".join(evt.get("weak_areas", [])) if evt.get("weak_areas") else "—"
    context_title = f"Quiz · {topic}"
    if metadata.get("difficulty"):
        context_title += f" · {metadata['difficulty']}"
    return {
        "event_type": "quiz_result",
        "context_title": context_title,
        "date": timestamp,
        "detail": f"Score: {evt.get('score', 0):.0f}% · Weak areas: {weak_str}",
        "summary": "",
    }


def _fallback_feedback_topic_title(topic: str) -> str:
    """Collapse older raw Learn Path topic strings to the main path title when possible."""
    if ": " in topic:
        return topic.split(": ", 1)[0]
    return topic


def _format_feedback_entry_display(entry: dict) -> dict[str, str]:
    """Return a compact display model for one persisted feedback entry."""
    metadata = entry.get("metadata", {}) or {}
    context_type = entry.get("context_type")
    topic = entry.get("topic", "Untitled")
    if context_type == "learn":
        learning_mode = metadata.get("learning_mode")
        difficulty = metadata.get("difficulty")
        learning_depth = metadata.get("learning_depth")
        context_title = metadata.get("context_title") or _fallback_feedback_topic_title(topic)
        if learning_mode == "Learn Path":
            label = f"Learn Path · {difficulty} - {context_title}" if difficulty else f"Learn Path · {context_title}"
            if learning_depth:
                label += f" ({learning_depth})"
            context_label = label
        elif learning_mode == "Topic":
            context_label = f"Topic · {context_title}"
        else:
            context_label = f"Learn · {context_title}"
    else:
        difficulty = metadata.get("difficulty")
        context_title = metadata.get("context_title", topic)
        context_label = f"Quiz · {context_title}"
        if difficulty:
            context_label += f" · {difficulty}"
    return {
        "id": str(entry.get("id", "")),
        "rating_label": f"⭐ Rating: {entry.get('rating', 0)}/5",
        "context_title": context_label,
        "timestamp": (entry.get("timestamp") or "")[:10] or "—",
        "comment": entry.get("comment") or "No comment provided.",
    }


def _humanize_feedback_suggestion(suggestion: str | None) -> str:
    """Format one stored suggestion label for reviewer-friendly display."""
    if suggestion == "increase_difficulty":
        return "Increase difficulty"
    if suggestion == "simplify":
        return "Simplify explanations"
    return "—"


def render_progress() -> None:
    """Render the Progress section with learning memory and feedback data."""
    st.header("Progress")
    st.markdown("Track your studied topics, quiz scores, weak areas, and feedback.")
    st.info(
        "Progress is recorded when you explicitly save a Learn result as studied or save a completed Quiz result."
    )

    from src.memory.feedback_service import delete_feedback, get_feedback_summary, get_recent_feedback
    from src.memory.memory_service import (
        LEARN_STUDIED_SOURCE,
        QUIZ_EVALUATION_SOURCE,
        delete_learning_event,
        get_completed_learn_sessions,
        get_quiz_performance_events,
        get_user_profile_summary,
        get_weak_areas_summary,
    )

    completed_sessions = get_completed_learn_sessions(limit=10)
    quiz_events = get_quiz_performance_events(limit=10)
    feedback_entries = get_recent_feedback(limit=5)
    feedback_summary = get_feedback_summary()

    st.subheader("Completed Learn Sessions")
    if completed_sessions:
        for evt in completed_sessions:
            display = _format_learning_event_display(evt)
            with st.container(border=True):
                st.caption(display["date"])
                st.markdown(f"**{display['context_title']}**")
                st.markdown(display["detail"])
                if display["summary"]:
                    st.write(display["summary"])
                if st.button("Mark as Not Completed", key=f"delete_completed_learn_{evt['id']}"):
                    delete_learning_event(int(evt["id"]), source=LEARN_STUDIED_SOURCE)
                    st.rerun()
    else:
        st.info("No completed Learn sessions saved yet.")

    st.subheader("Quiz Performance")
    if quiz_events:
        for evt in quiz_events:
            display = _format_learning_event_display(evt)
            with st.container(border=True):
                st.caption(display["date"])
                st.markdown(f"**{display['context_title']}**")
                st.markdown(display["detail"])
                if st.button("🗑️ Delete", key=f"delete_quiz_event_{evt['id']}"):
                    delete_learning_event(int(evt["id"]), source=QUIZ_EVALUATION_SOURCE)
                    st.rerun()

        profile = get_user_profile_summary()
        st.markdown("#### Aggregate Signals")
        if profile.get("average_score") is not None:
            st.markdown(f"**Average score:** {profile['average_score']:.0f}%")
        weak_areas = profile.get("recurring_weak_areas", [])
        if weak_areas:
            st.markdown("**Recurring weak areas:** " + ", ".join(weak_areas))
        suggested_focus = profile.get("suggested_focus_topics", [])
        if suggested_focus:
            st.markdown("**Suggested focus topics:** " + ", ".join(suggested_focus))

        weak_area_summary = get_weak_areas_summary()
        if weak_area_summary:
            st.markdown("#### Weak Areas Summary")
            for area, count in sorted(weak_area_summary.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{area}** — appeared {count} time(s)")
    else:
        st.info("No saved quiz performance yet.")

    st.subheader("Recent Feedback")
    if feedback_entries:
        for fb in feedback_entries:
            display = _format_feedback_entry_display(fb)
            with st.container(border=True):
                st.caption(display["timestamp"])
                st.markdown(f"**{display['context_title']}**")
                st.markdown(display["rating_label"])
                st.write(display["comment"])
                if st.button("🗑️ Delete", key=f"delete_feedback_{display['id']}"):
                    delete_feedback(int(display["id"]))
                    st.rerun()
    else:
        st.info("No feedback recorded yet.")

    if feedback_summary.get("total_count", 0) > 0:
        st.subheader("Feedback Summary")
        st.markdown(f"- **Average rating:** {feedback_summary['average_rating']}")
        st.markdown(f"- **Total feedback entries:** {feedback_summary['total_count']}")
        st.markdown(
            f"- **Personalization suggestion:** {_humanize_feedback_suggestion(feedback_summary.get('suggestion'))}"
        )
