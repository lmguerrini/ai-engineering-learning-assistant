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

## Practical Implementation Notes

- Use `st.form` for multi-field submissions to reduce reruns.
- Keep page functions modular — one function per page section.
- Use `st.session_state` for workflow state (quiz answers, graph results).
- Display errors with `st.error()` — never raise uncaught exceptions in UI code.
- Use `st.spinner` around slow operations (LLM calls, retrieval).
- Prefer `@st.cache_resource` for heavy objects (vector store, embedding model).

## Common Mistakes

- Putting expensive operations outside caching — they rerun on every interaction.
- Not using `st.session_state` — losing state on each rerun.
- Nesting `st.button` checks — inner buttons never trigger due to rerun model.
- Using `st.rerun()` excessively, causing infinite rerun loops.
- Forgetting that `st.button` returns `True` for only one script execution.

## Related Project Usage

- `app.py`: Streamlit entrypoint with sidebar navigation.
- `src/ui/pages.py`: Page rendering functions for Learn, Quiz, Progress, Advanced.
- `src/ui/display_helpers.py`: Reusable display formatting helpers.
