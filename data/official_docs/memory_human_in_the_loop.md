# Long-Term Memory & Human-in-the-Loop

- **Official source**: https://langchain-ai.github.io/langgraph/concepts/memory/, https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- **Last refreshed**: 2026-05-12
- **source_type**: official_docs
- **Versions**: `langgraph>=0.2`, `langchain>=0.2`

## When to Use

- Building agents that remember user preferences and past interactions across sessions.
- Implementing approval gates, review steps, or interactive corrections in agent workflows.
- Creating personalized learning or assistant systems that improve over time.

## Key Concepts

### Long-Term Memory

Long-term memory allows agents to persist information beyond a single conversation, enabling personalization and continuity.

**Memory types**:
1. **Conversation memory** — recent messages within a session (short-term).
2. **User profile memory** — aggregated preferences, strengths, weaknesses (long-term).
3. **Episodic memory** — specific past interactions or events.
4. **Semantic memory** — learned facts and knowledge.

```python
import json
from pathlib import Path

class UserMemoryStore:
    """Simple file-based long-term memory store."""

    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_profile(self, user_id: str, profile: dict) -> None:
        path = self.storage_dir / f"{user_id}.json"
        path.write_text(json.dumps(profile, indent=2))

    def load_profile(self, user_id: str) -> dict | None:
        path = self.storage_dir / f"{user_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def update_profile(self, user_id: str, updates: dict) -> dict:
        profile = self.load_profile(user_id) or {}
        profile.update(updates)
        self.save_profile(user_id, profile)
        return profile
```

### Memory Profile Aggregation

```python
def build_user_profile(quiz_results: list[dict], learn_history: list[dict]) -> dict:
    """Aggregate user activity into a memory profile."""
    topics = [r["topic"] for r in learn_history[-10:]]
    scores = [r["score"] for r in quiz_results[-20:]]
    weak_areas = identify_weak_areas(quiz_results)

    return {
        "recent_topics": topics,
        "average_score": sum(scores) / len(scores) if scores else 0,
        "recurring_weak_areas": weak_areas,
        "total_sessions": len(learn_history),
        "preferred_style": infer_preferred_style(learn_history),
        "suggested_focus_topics": weak_areas[:3],
    }
```

### Using Memory for Personalization

```python
def build_personalized_prompt(topic: str, memory_profile: dict) -> str:
    parts = [f"Generate a study guide about {topic}."]

    if memory_profile.get("recurring_weak_areas"):
        areas = ", ".join(memory_profile["recurring_weak_areas"])
        parts.append(f"The learner struggles with: {areas}. Provide extra detail on these.")

    if memory_profile.get("average_score", 100) < 60:
        parts.append("Use simpler explanations and more examples — the learner is still building foundations.")

    if memory_profile.get("preferred_style") == "examples_heavy":
        parts.append("Include many practical code examples.")

    return "\n".join(parts)
```

### Human-in-the-Loop (HITL)

HITL patterns insert human decision points into automated agent workflows.

**Common HITL patterns**:

1. **Approval gate** — pause before executing a dangerous action.
2. **Review and edit** — human reviews and modifies agent output before it's used.
3. **Feedback loop** — human rates output quality; agent adjusts.
4. **Escalation** — agent recognizes uncertainty and hands off to a human.

```python
from langgraph.graph import StateGraph, START, END

class HITLState(TypedDict, total=False):
    plan: str
    human_approved: bool
    result: str
    trace: list[str]

def generate_plan(state: HITLState) -> dict:
    plan = llm.invoke("Create an action plan for: " + state.get("task", ""))
    return {"plan": plan, "trace": ["plan_generated"]}

def check_approval(state: HITLState) -> str:
    if state.get("human_approved"):
        return "execute"
    return "wait_for_approval"

graph = StateGraph(HITLState)
graph.add_node("plan", generate_plan)
graph.add_node("execute", execute_plan)
graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", check_approval)
graph.add_edge("execute", END)
```

### Checkpointing for HITL

LangGraph checkpointers enable pausing and resuming workflows:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, interrupt_before=["execute"])

# Start the workflow — it pauses before "execute"
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"task": "deploy model"}, config)

# Human reviews the plan, then resumes
app.invoke({"human_approved": True}, config)
```

### Production Considerations

- **Memory storage**: Use databases (SQLite, PostgreSQL) for production; file-based for prototyping.
- **Privacy**: Encrypt sensitive memory data; implement data retention policies.
- **Memory decay**: Implement TTL or relevance scoring to prevent stale memories from dominating.
- **Conflict resolution**: Define merge strategies when concurrent sessions update the same profile.
- **HITL timeouts**: Set deadlines for human responses; auto-escalate or auto-reject on timeout.
- **Audit trail**: Log all human decisions for compliance and debugging.
- **Graceful degradation**: If memory is unavailable, the system should still function with defaults.

### Common Mistakes

1. **Unbounded memory** — storing everything without cleanup leads to context pollution.
2. **No fallback** — system crashes when memory store is unavailable.
3. **Stale profiles** — never refreshing aggregated profiles after new data arrives.
4. **Blocking on HITL** — workflow hangs indefinitely waiting for human input.
5. **Missing validation** — trusting memory data without schema validation.
6. **Over-personalization** — memory biases the agent so strongly it ignores the current query.

### Memory vs. Context Window

| Aspect | Context Window | Long-Term Memory |
|---|---|---|
| **Scope** | Current conversation | Cross-session |
| **Storage** | In-memory (prompt) | Persistent (DB/file) |
| **Size limit** | Token limit | Practically unlimited |
| **Freshness** | Always current | May be stale |
| **Cost** | Tokens per call | Storage + retrieval |

## Anti-Patterns

- Dumping entire memory into every prompt (token waste, noise).
- Using memory as a replacement for proper retrieval (RAG).
- Not testing workflows with empty/corrupted memory states.
- Implementing HITL without timeout or escalation paths.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://langchain-ai.github.io/langgraph/concepts/memory/

```
Redirecting...
```

### https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/

```
Redirecting...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
