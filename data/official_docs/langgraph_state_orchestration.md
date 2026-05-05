# LangGraph State & Orchestration

- **Official source**: https://langchain-ai.github.io/langgraph/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Building stateful, multi-step agent workflows.
- Implementing conditional routing, loops, and human-in-the-loop patterns.
- Orchestrating LLM calls with explicit control flow.

## Key Concepts

### StateGraph

- `StateGraph(StateType)` creates a graph with a typed state schema.
- State is typically a `TypedDict` with `total=False` for optional fields.
- Each node receives the full state and returns a partial state update.
- State updates are merged automatically after each node execution.

### Nodes

- Added via `graph.add_node("name", function)`.
- Node functions signature: `def node(state: State) -> dict`.
- Return dict is merged into state — only returned keys are updated.
- Nodes should be pure functions where possible (side-effect-free).

### Edges & Routing

- `graph.add_edge("source", "target")` for unconditional transitions.
- `graph.add_conditional_edges("source", routing_fn)` for branching.
- Routing functions return the name of the next node as a string.
- `START` and `END` are special constants for graph entry/exit.

### Reducers

- `Annotated[list, operator.add]` appends to lists instead of replacing.
- Custom reducers can be defined for any state field.
- Without a reducer, the latest value overwrites the previous one.
- Reducers enable accumulation patterns (traces, messages, results).

### Conditional Routing

- Routing function receives state, returns next node name.
- Can implement loops: node A → assess → (if bad) refine → node A.
- Loop control via counter fields to prevent infinite loops.
- Multiple outgoing edges from a single node based on state conditions.

### Checkpoints & Persistence

- `MemorySaver()` provides in-memory checkpointing.
- `SqliteSaver` for persistent checkpoints across sessions.
- Checkpoints enable resumption, time-travel debugging, and HITL.
- Pass `checkpointer` to `graph.compile(checkpointer=saver)`.

### Human-in-the-Loop (HITL)

- Use `interrupt_before=["node_name"]` or `interrupt_after=["node_name"]`.
- Graph pauses at the interrupt point, awaiting external input.
- Resume with `graph.invoke(None, config)` after providing input.
- Enables approval workflows and manual review steps.

## Practical Implementation Notes

- Use `total=False` on TypedDict so nodes only need to return changed fields.
- Add a `trace: list[str]` field with `Annotated[list, operator.add]` for debugging.
- Keep node functions small and focused on a single responsibility.
- Compile the graph once and reuse the compiled app for multiple invocations.
- Use `graph.get_graph().draw_mermaid()` for visualization during development.

## Common Mistakes

- Forgetting `START` and `END` edges, causing compilation errors.
- Returning the full state from nodes instead of only changed fields.
- Not adding loop control (max attempts) leading to infinite loops.
- Using mutable default state values that persist across invocations.
- Confusing `add_edge` (unconditional) with `add_conditional_edges`.

## Related Project Usage

- `src/graphs/learn_graph.py`: Learn workflow with conditional Agentic RAG routing.
- `src/graphs/learn_state.py`: Typed LearningState with trace accumulation.
- `src/graphs/quiz_graph.py`: Quiz generation and evaluation graphs.
- `src/graphs/learn_nodes.py`: Node functions following state-in/dict-out pattern.
