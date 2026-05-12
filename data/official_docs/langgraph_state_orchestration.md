# LangGraph State & Orchestration

- **Official source**: https://langchain-ai.github.io/langgraph/
- **Last refreshed**: 2026-05-12
- **source_type**: official_docs
- **Versions**: `langgraph>=0.2`

## When to Use

- Building stateful, multi-step agent workflows.
- Implementing conditional routing, loops, and human-in-the-loop patterns.
- Orchestrating LLM calls with explicit control flow.

## Key Concepts

### StateGraph

Define a graph with a typed state schema. Each node receives the full state and returns a partial update.

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator

class LearningState(TypedDict, total=False):
    topic: str
    context: str
    study_guide: str
    trace: Annotated[list[str], operator.add]  # accumulates across nodes

graph = StateGraph(LearningState)
```

- State is typically a `TypedDict` with `total=False` for optional fields.
- State updates are merged automatically after each node execution.
- Use `Annotated` types with reducer functions to control how state fields are updated (see Reducers).
- For complex state, consider splitting into sub-dicts to keep node signatures clean.

> **Caveat**: `StateGraph` validates that every node return type matches the state schema at compile time. Returning unexpected keys raises `InvalidUpdateError`.

### Nodes

```python
def retrieve_context(state: LearningState) -> dict:
    """Node: retrieve relevant documents for the topic."""
    docs = search_kb(state["topic"])
    return {"context": docs, "trace": ["retrieve_context"]}

def generate_guide(state: LearningState) -> dict:
    """Node: generate a study guide from context."""
    guide = llm.invoke(f"Create a study guide about {state['topic']}:\n{state['context']}")
    return {"study_guide": guide, "trace": ["generate_guide"]}

graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_guide)
```

- Node signature: `def node(state: State) -> dict` — return only changed fields.
- Nodes should be pure functions where possible (side-effect-free).
- Async nodes are supported: `async def node(state: State) -> dict`.
- Nodes can access config via a second parameter: `def node(state: State, config: RunnableConfig) -> dict`.
- Node execution time is measured automatically; visible in LangSmith traces.

**Error handling in nodes**:

```python
def safe_retrieve(state: LearningState) -> dict:
    try:
        docs = search_kb(state["topic"])
        return {"context": docs, "trace": ["retrieve_ok"]}
    except Exception as e:
        return {"context": "", "error": str(e), "trace": ["retrieve_failed"]}
```

- Unhandled exceptions in nodes propagate to the caller and halt the graph.
- Use try/except within nodes for graceful degradation; store error info in state for downstream routing.

### Edges & Conditional Routing

```python
# Unconditional edge
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "assess")

# Conditional routing
def should_refine(state: LearningState) -> str:
    if state.get("quality_score", 0) < 0.7 and state.get("attempts", 0) < 3:
        return "refine"
    return "finalize"

graph.add_conditional_edges("assess", should_refine)
graph.add_edge("finalize", END)
```

- `START` and `END` are special constants for graph entry and exit.
- Routing functions receive state and return the next node name as a string.
- Implement loop control via counter fields to prevent infinite loops.
- Routing functions must return a string that matches an existing node name; returning an unknown name raises `ValueError` at runtime.

**Multi-path conditional routing**:

```python
def route_by_quality(state: LearningState) -> str:
    score = state.get("quality_score", 0)
    attempts = state.get("attempts", 0)
    if score >= 0.8:
        return "finalize"
    elif attempts >= 3:
        return "finalize"  # give up after max attempts
    else:
        return "refine"

graph.add_conditional_edges(
    "assess",
    route_by_quality,
    {"finalize": "finalize", "refine": "refine"},  # explicit mapping
)
```

- The optional third argument to `add_conditional_edges` maps return values to node names — use it for clarity and to catch routing errors early.
- Conditional edges support `END` as a target: `{"done": END, "continue": "next_node"}`.

### Reducers

```python
from typing import Annotated
import operator

class State(TypedDict, total=False):
    messages: Annotated[list[str], operator.add]  # appends, not replaces
    count: int  # latest value overwrites
```

- `Annotated[list, operator.add]` appends to lists instead of replacing.
- Without a reducer, the latest value overwrites the previous one.
- Custom reducers can be defined for any state field.

**Custom reducer example**:

```python
def merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """Reducer that appends only unique items."""
    return list(dict.fromkeys(existing + new))

class State(TypedDict, total=False):
    sources: Annotated[list[str], merge_unique]  # deduplicates on merge
    messages: Annotated[list[str], operator.add]  # standard append
    result: str  # overwrite (no reducer)
```

- Reducer functions receive `(existing_value, new_value)` and return the merged result.
- Reducer errors (e.g., type mismatch) surface as `InvalidUpdateError` at runtime.

### Checkpoints & Persistence

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # or SqliteSaver for persistence
app = graph.compile(checkpointer=checkpointer)

# Invoke with thread ID for checkpoint tracking
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"topic": "RAG"}, config)
```

- `MemorySaver()` for in-memory checkpointing (development).
- `SqliteSaver` for persistent checkpoints across sessions.
- `PostgresSaver` for production multi-process deployments.
- Checkpoints enable resumption, time-travel debugging, and HITL.

**Time-travel debugging**:

```python
# List checkpoint history for a thread
history = list(app.get_state_history(config))

# Inspect state at any checkpoint
for state in history:
    print(state.values, state.created_at)

# Resume from a specific checkpoint
old_config = history[2].config  # third checkpoint
result = app.invoke(None, old_config)
```

- Each node execution creates a checkpoint; `get_state_history()` returns them in reverse chronological order.
- Checkpoints store the full state — large state objects increase storage requirements.

### Human-in-the-Loop (HITL)

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["generate"],  # pause before this node
)

# Graph pauses at interrupt point; resume after external input:
result = app.invoke(None, config)
```

- Use `interrupt_before` or `interrupt_after` with node names.
- Enables approval workflows and manual review steps.
- Combine with `app.update_state(config, {"approved": True})` to inject external input before resuming.
- Multiple interrupt points are supported: `interrupt_before=["step_a", "step_b"]`.

**State update before resuming**:

```python
# Pause before "generate"
result = app.invoke({"topic": "RAG"}, config)

# External review happens here...

# Inject reviewer feedback into state
app.update_state(config, {"reviewer_note": "Add more examples"})

# Resume execution
result = app.invoke(None, config)
```

## Advanced Patterns

### Subgraphs

```python
# Create a subgraph for a reusable workflow
sub_graph = StateGraph(SubState)
sub_graph.add_node("sub_step", sub_step_fn)
sub_graph.add_edge(START, "sub_step")
sub_graph.add_edge("sub_step", END)

# Embed in parent graph
parent_graph = StateGraph(ParentState)
parent_graph.add_node("main_step", main_fn)
parent_graph.add_node("sub_workflow", sub_graph.compile())
parent_graph.add_edge("main_step", "sub_workflow")
```

- Subgraphs encapsulate reusable workflows (e.g., a RAG retrieval sub-pipeline).
- State mapping between parent and subgraph must be compatible — shared field names pass through.

### Dynamic Node Selection

```python
def select_node(state: State) -> str:
    """Dynamically pick next node based on state."""
    if state.get("needs_enrichment"):
        return "enrich"
    return "generate"

graph.add_conditional_edges(START, select_node)
```

### Graph Visualization

```python
# Mermaid diagram (copy to mermaid.live for rendering)
print(graph.get_graph().draw_mermaid())

# ASCII representation for debugging
print(graph.get_graph().draw_ascii())

# PNG export (requires graphviz)
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

## Practical Implementation Notes

- Use `total=False` on `TypedDict` so nodes only need to return changed fields.
- Add a `trace: Annotated[list[str], operator.add]` field for debugging execution flow.
- Keep node functions small and focused on a single responsibility.
- Compile the graph once and reuse the compiled app for multiple invocations.
- Visualize during development: `graph.get_graph().draw_mermaid()`.
- Use `recursion_limit` in config to cap maximum node executions per invocation: `config={"recursion_limit": 25}`.
- Default `recursion_limit` is 25; set higher for deeply iterative workflows, lower for safety.
- Test graphs with mocked node functions to verify routing logic independently of LLM calls.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `InvalidUpdateError` | Node returned a key not in state schema | Check return dict keys match `TypedDict` fields |
| Graph hangs indefinitely | Infinite loop in conditional routing | Add `attempts` counter; set `recursion_limit` in config |
| `ValueError: Unknown node` | Routing function returned invalid node name | Verify routing return values match `add_node` names |
| State data missing between nodes | Field overwritten without reducer | Use `Annotated[list, operator.add]` for accumulation fields |
| Checkpoint grows unbounded | Large state stored at every node | Minimize state size; avoid storing full documents in state |
| Compilation error | Missing `START` or `END` edges | Ensure at least one edge from `START` and to `END` |

## Common Mistakes

- Forgetting `START` and `END` edges, causing compilation errors.
- Returning the full state from nodes instead of only changed fields.
- Not adding loop control (max attempts) leading to infinite loops.
- Using mutable default state values that persist across invocations.
- Confusing `add_edge` (unconditional) with `add_conditional_edges`.
- Not compiling the graph before invoking — `graph.invoke()` doesn't exist; use `app = graph.compile()` then `app.invoke()`.
- Forgetting `checkpointer` when using `interrupt_before`/`interrupt_after` — interrupts require checkpointing.
- Storing large objects (full documents, embeddings) in state — bloats checkpoints and slows serialization.

## Related Project Usage

- `src/graphs/learn_graph.py`: Learn workflow with conditional Agentic RAG routing.
- `src/graphs/learn_state.py`: Typed `LearningState` with trace accumulation.
- `src/graphs/quiz_graph.py`: Quiz generation and evaluation graphs.
- `src/graphs/learn_nodes.py`: Node functions following state-in/dict-out pattern.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://langchain-ai.github.io/langgraph/

```
Documentation has moved The LangGraph documentation has moved to docs.langchain.com. Redirecting you now...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
