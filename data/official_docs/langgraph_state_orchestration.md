# LangGraph State & Orchestration

- **Official source**: https://langchain-ai.github.io/langgraph/
- **Last refreshed**: 2025-05-05
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
- Checkpoints enable resumption, time-travel debugging, and HITL.

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

## Practical Implementation Notes

- Use `total=False` on `TypedDict` so nodes only need to return changed fields.
- Add a `trace: Annotated[list[str], operator.add]` field for debugging execution flow.
- Keep node functions small and focused on a single responsibility.
- Compile the graph once and reuse the compiled app for multiple invocations.
- Visualize during development: `graph.get_graph().draw_mermaid()`.

## Common Mistakes

- Forgetting `START` and `END` edges, causing compilation errors.
- Returning the full state from nodes instead of only changed fields.
- Not adding loop control (max attempts) leading to infinite loops.
- Using mutable default state values that persist across invocations.
- Confusing `add_edge` (unconditional) with `add_conditional_edges`.

## Related Project Usage

- `src/graphs/learn_graph.py`: Learn workflow with conditional Agentic RAG routing.
- `src/graphs/learn_state.py`: Typed `LearningState` with trace accumulation.
- `src/graphs/quiz_graph.py`: Quiz generation and evaluation graphs.
- `src/graphs/learn_nodes.py`: Node functions following state-in/dict-out pattern.
