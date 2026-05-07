# Streamlit App Patterns

- **Official source**: https://docs.streamlit.io/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs
- **Versions**: `streamlit>=1.30`

## When to Use

- Building interactive web UIs for data and AI applications.
- Rapid prototyping of dashboards and internal tools.
- Creating demo interfaces for LLM-powered applications.

## Key Concepts

### App Execution Model

Streamlit reruns the **entire script** on each user interaction (widget click, input change).

```python
import streamlit as st

st.title("AI Learning Assistant")

# This runs on EVERY interaction
topic = st.text_input("Enter a topic:")

if topic:
    with st.spinner("Generating study guide..."):
        result = generate_guide(topic)  # expensive — should be cached or in session_state
    st.markdown(result)
```

- Top-to-bottom execution — every widget and display call runs every time.
- Avoid expensive computations at module level; use caching or session state.
- Each user gets an independent session — no shared state between browser tabs by default.
- Script reruns are fast (~50ms overhead) but accumulated widget calls add up — keep pages focused.

### Layout & Navigation

```python
# Sidebar navigation
with st.sidebar:
    page = st.radio("Navigation", ["Learn", "Quiz", "Progress"])

# Columns
col1, col2 = st.columns(2)
with col1:
    st.metric("Score", "85%")
with col2:
    st.metric("Topics", 12)

# Tabs
tab1, tab2 = st.tabs(["Study Guide", "Sources"])
with tab1:
    st.markdown(guide_content)
with tab2:
    st.json(sources)

# Collapsible sections
with st.expander("Show trace details"):
    st.code(trace_log)
```

### Session State

Persist data across reruns using `st.session_state`:

```python
# Initialize state
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = []

# Update state
st.session_state.quiz_answers.append(user_answer)

# Widgets auto-sync with session state via key parameter
st.text_input("Topic", key="current_topic")
# Access: st.session_state.current_topic
```

- Session state resets when the browser tab is closed.
- Use for workflow state: quiz answers, graph results, conversation history.
- Session state persists across reruns but not across server restarts.
- Use `del st.session_state["key"]` to explicitly remove state entries.

**Callback pattern for immediate state updates**:

```python
def on_topic_change():
    # Runs BEFORE the rerun — state is updated immediately
    st.session_state.study_guide = None  # clear stale results

st.text_input("Topic", key="topic", on_change=on_topic_change)
```

- Callbacks execute before the page reruns — useful for clearing dependent state.
- Callbacks receive no arguments by default; use `args` and `kwargs` parameters to pass data.
- Avoid heavy computation in callbacks — they block the rerun.

### Caching

```python
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_knowledge_base(path: str) -> list[dict]:
    """Cached — result stored based on input hash."""
    return load_documents(path)

@st.cache_resource
def get_vector_store():
    """Cached — singleton resource shared across reruns and sessions."""
    return initialize_chroma_client()
```

| Decorator | Use Case | Scope |
|-----------|----------|-------|
| `@st.cache_data` | Data transforms, API calls, file reads | Per-input hash, serialized |
| `@st.cache_resource` | DB connections, ML models, heavy objects | Global singleton |

- `ttl` parameter sets time-to-live in seconds.
- Cache clears on code change or manual `st.cache_data.clear()`.
- `@st.cache_data` serializes return values (via `pickle`) — returned objects are copies, not references.
- `@st.cache_resource` returns the **same object instance** — mutations affect all users.

> **Caveat**: `@st.cache_resource` objects are shared across all sessions. Do not store user-specific data in cached resources. Use session state for per-user data.

**Cache with spinner**:

```python
@st.cache_data(show_spinner="Loading knowledge base...")
def load_kb(path: str) -> list[dict]:
    return load_documents(path)
```

- `show_spinner` displays a custom message during the first (uncached) call.
- `@st.cache_data(experimental_allow_widgets=True)` allows widgets inside cached functions (use with caution).

### Input Widgets

| Widget | Returns | Use Case |
|--------|---------|----------|
| `st.text_input(label)` | `str` | Short text entry |
| `st.text_area(label)` | `str` | Multi-line text |
| `st.selectbox(label, options)` | selected value | Single selection |
| `st.multiselect(label, options)` | `list` | Multiple selection |
| `st.slider(label, min, max)` | numeric | Range input |
| `st.button(label)` | `bool` | Action trigger (True for one rerun only) |

```python
# Form batches inputs — single rerun on submit
with st.form("quiz_form"):
    answer = st.radio("Your answer:", options)
    submitted = st.form_submit_button("Submit")
    if submitted:
        check_answer(answer)
```

> **Note**: Forms prevent intermediate reruns — no rerun occurs until the submit button is pressed. This is critical for multi-field input pages where each keystroke would otherwise trigger a rerun.

### Display Elements

```python
# Rich text
st.markdown("## Study Guide")
st.write(content)  # auto-detects type: str, dict, DataFrame, etc.

# Status messages
st.success("Guide generated successfully!")
st.error("Failed to connect to OpenAI API.")
st.warning("No documents found for this topic.")
st.info("Tip: Try a more specific query.")

# Structured data
st.json({"topic": "RAG", "sections": [...]})
st.dataframe(progress_df)

# Progress indication
with st.spinner("Searching knowledge base..."):
    results = search_kb(query)
```

## Advanced Patterns

### Multi-Page Architecture

```python
# app.py — page routing pattern
import streamlit as st

def main():
    with st.sidebar:
        page = st.radio("Navigation", ["Learn", "Quiz", "Progress"])
    
    if page == "Learn":
        render_learn_page()
    elif page == "Quiz":
        render_quiz_page()
    elif page == "Progress":
        render_progress_page()

# Each page function is self-contained
def render_learn_page():
    st.header("Learn")
    # ... page-specific logic
```

- Keep page functions in separate modules for maintainability.
- Use `st.session_state` to pass data between pages (e.g., quiz results → progress page).
- Streamlit also supports native multi-page apps via `pages/` directory — each file becomes a page.

### Error Handling Pattern

```python
def safe_generate(topic: str) -> str | None:
    try:
        with st.spinner("Generating study guide..."):
            return generate_guide(topic)
    except Exception as e:
        st.error(f"Generation failed: {e}")
        logger.exception("Guide generation error")
        return None

result = safe_generate(topic)
if result:
    st.markdown(result)
```

- Never let exceptions propagate uncaught — Streamlit shows a generic error page.
- Log errors server-side while showing user-friendly messages in the UI.
- Use `st.warning()` for recoverable issues; `st.error()` for failures.

### Performance Optimization

- **Minimize widget count**: Each widget adds overhead to the rerun cycle. Use `st.form` to batch inputs.
- **Lazy loading**: Only compute/display what’s visible. Use `st.expander` for optional details.
- **Fragment reruns** (Streamlit ≥1.33): `@st.fragment` decorator allows a section to rerun independently without rerunning the full page.

```python
@st.fragment
def quiz_section():
    # This section reruns independently when its widgets change
    answer = st.radio("Your answer:", options, key="q1")
    if st.button("Check", key="check_q1"):
        st.write("Correct!" if answer == correct else "Try again")
```

- Fragments reduce unnecessary recomputation on large pages.
- Use fragments for interactive sections that don’t affect the rest of the page.

### Deployment Considerations

- **Streamlit Community Cloud**: Deploy directly from GitHub; add `requirements.txt` and `secrets.toml` (`.streamlit/secrets.toml`).
- **Docker**: Use `streamlit run app.py --server.headless=true --server.port=8501` as the entrypoint.
- **Environment variables**: Access via `os.environ` or `st.secrets` (for Streamlit Cloud).
- **Health checks**: Streamlit exposes `/_stcore/health` endpoint for load balancer probes.
- **Resource limits**: Set `server.maxUploadSize`, `server.maxMessageSize` in `.streamlit/config.toml` for production.

```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
maxUploadSize = 10

[browser]
gatherUsageStats = false
```

## Practical Implementation Notes

- Use `st.form` for multi-field submissions to reduce reruns.
- Keep page functions modular — one function per page section.
- Use `st.session_state` for workflow state (quiz answers, graph results).
- Display errors with `st.error()` — never raise uncaught exceptions in UI code.
- Use `st.spinner` around slow operations (LLM calls, retrieval).
- Prefer `@st.cache_resource` for heavy objects (vector store, embedding model).
- Use `st.toast()` for non-blocking success notifications (Streamlit ≥1.28).
- Set `key` parameter on all widgets to avoid duplicate widget ID errors.
- Use `st.empty()` as a placeholder for dynamically updated content.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Widget state lost on interaction | Not using `st.session_state` | Store computed results in session state |
| `DuplicateWidgetID` error | Multiple widgets with same auto-generated key | Add explicit `key` parameter to each widget |
| Button inside button doesn’t work | Nested `st.button` checks; inner resets on rerun | Use `st.session_state` flag instead of nested button checks |
| Infinite rerun loop | `st.rerun()` called unconditionally | Guard `st.rerun()` with a condition; prefer callbacks |
| Cached function returns stale data | Cache not invalidated after data change | Call `st.cache_data.clear()` or change function input |
| `@st.cache_resource` object mutated | Shared singleton modified by one session | Use `@st.cache_data` for mutable data; `@st.cache_resource` for read-only |
| Slow page load | Heavy computation in page body | Move to `@st.cache_data` or `@st.cache_resource`; use lazy loading |
| `st.form_submit_button` outside form | Button not inside `with st.form():` block | Ensure `form_submit_button` is within the form context manager |

## Common Mistakes

- Putting expensive operations outside caching — they rerun on every interaction.
- Not using `st.session_state` — losing state on each rerun.
- Nesting `st.button` checks — inner buttons never trigger due to rerun model.
- Using `st.rerun()` excessively, causing infinite rerun loops.
- Forgetting that `st.button` returns `True` for only one script execution.
- Mutating `@st.cache_resource` objects — affects all users sharing that resource.
- Not handling exceptions in UI code — users see raw Python tracebacks.
- Using `time.sleep()` for delays — blocks the entire server thread; use `st.spinner` for UX.

## Related Project Usage

- `app.py`: Streamlit entrypoint with sidebar navigation.
- `src/ui/pages.py`: Page rendering functions for Learn, Quiz, Progress, Advanced.
- `src/ui/display_helpers.py`: Reusable display formatting helpers.
