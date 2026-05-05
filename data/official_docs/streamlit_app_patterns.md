# Streamlit App Patterns

- **Official source**: https://docs.streamlit.io/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Building interactive web UIs for data and AI applications.
- Rapid prototyping of dashboards and internal tools.
- Creating demo interfaces for LLM-powered applications.

## Key Concepts

### App Execution Model

- Streamlit reruns the entire script on each user interaction.
- Top-to-bottom execution — every widget and display call runs every time.
- Use `st.session_state` to persist data across reruns.
- Avoid expensive computations at module level; use caching.

### Layout & Navigation

- `st.sidebar` for navigation menus and settings.
- `st.columns(n)` for horizontal layouts.
- `st.tabs(["Tab1", "Tab2"])` for tabbed content.
- `st.expander("Title")` for collapsible sections.
- `st.container()` for grouping related elements.

### Session State

- `st.session_state["key"] = value` persists across reruns.
- Check existence: `if "key" not in st.session_state:`.
- Widgets with `key` parameter auto-sync with session state.
- Session state resets when the browser tab is closed.

### Caching

- `@st.cache_data` caches function results based on input hash.
- `@st.cache_resource` caches expensive objects (DB connections, models).
- `ttl` parameter sets time-to-live for cache entries.
- Cache clears on code change or manual `st.cache_data.clear()`.

### Input Widgets

- `st.text_input`, `st.text_area` for text entry.
- `st.selectbox`, `st.radio`, `st.multiselect` for selection.
- `st.slider`, `st.number_input` for numeric input.
- `st.button` returns `True` on click (one rerun only).
- `st.form` with `st.form_submit_button` batches inputs before rerun.

### Display Elements

- `st.markdown`, `st.write` for rich text output.
- `st.json` for formatted JSON display.
- `st.dataframe`, `st.table` for tabular data.
- `st.success`, `st.error`, `st.warning`, `st.info` for status messages.
- `st.spinner("Loading...")` for progress indication.

## Practical Implementation Notes

- Use `st.form` for multi-field submissions to reduce reruns.
- Keep page functions modular — one function per page section.
- Use `st.session_state` for workflow state (quiz answers, graph results).
- Display errors with `st.error` — never raise uncaught exceptions in UI code.
- Use `st.spinner` around slow operations (LLM calls, retrieval).

## Common Mistakes

- Putting expensive operations outside caching — they rerun on every interaction.
- Not using `st.session_state` — losing state on each rerun.
- Nesting `st.button` checks — inner buttons never trigger due to rerun model.
- Using `st.rerun()` excessively, causing infinite rerun loops.
- Forgetting that `st.button` is `True` for only one script execution.

## Related Project Usage

- `app.py`: Streamlit entrypoint with sidebar navigation.
- `src/ui/pages.py`: Page rendering functions for Learn, Quiz, Progress, Advanced.
- `src/ui/display_helpers.py`: Reusable display formatting helpers.
