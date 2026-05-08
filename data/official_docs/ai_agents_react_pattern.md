# AI Agents & ReAct Pattern

- **Official source**: https://arxiv.org/abs/2210.03629, https://python.langchain.com/docs/concepts/agents/
- **Last refreshed**: 2025-05-08
- **source_type**: official_docs
- **Versions**: `langchain>=0.2`, `langgraph>=0.2`

## When to Use

- Building autonomous systems that reason about tasks and take actions.
- Implementing multi-step problem solving with tool use.
- Creating assistants that interleave thinking and acting.

## Key Concepts

### What Is an AI Agent?

An AI agent is a system that uses an LLM as its reasoning engine to decide which actions to take, execute those actions via tools, observe results, and iterate until a goal is achieved.

Core components:
1. **Reasoning engine** — the LLM that interprets instructions and plans steps.
2. **Tools** — functions the agent can invoke (search, calculate, API calls).
3. **Memory** — short-term (conversation context) and long-term (persisted knowledge).
4. **Orchestration** — the control flow that connects reasoning, action, and observation.

```python
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict, total=False):
    messages: list[dict]
    tool_results: list[str]
    final_answer: str

def reason(state: AgentState) -> dict:
    """LLM decides next action or produces final answer."""
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def act(state: AgentState) -> dict:
    """Execute the tool call from the LLM response."""
    tool_call = extract_tool_call(state["messages"][-1])
    result = execute_tool(tool_call)
    return {"tool_results": [result]}
```

### The ReAct Pattern

ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning with action execution:

1. **Thought** — the LLM reasons about what to do next.
2. **Action** — the LLM selects and invokes a tool.
3. **Observation** — the tool result is fed back to the LLM.
4. **Repeat** until the LLM produces a final answer.

```python
REACT_PROMPT = """Answer the question using the following format:

Thought: I need to figure out ...
Action: tool_name(arguments)
Observation: <tool result inserted here>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information.
Final Answer: <your answer>
"""
```

**Why ReAct works**:
- Explicit reasoning traces improve accuracy on multi-step tasks.
- Grounding actions in observations reduces hallucination.
- The pattern is inspectable — you can audit the agent's reasoning.

### Agent Architectures

| Architecture | Description | Best For |
|---|---|---|
| **ReAct loop** | Single LLM alternates thinking and acting | Simple tool-use tasks |
| **Plan-and-execute** | Planner creates steps, executor runs them | Complex multi-step tasks |
| **Multi-agent** | Multiple specialized agents collaborate | Domain-specific workflows |
| **Hierarchical** | Supervisor delegates to sub-agents | Large-scale orchestration |

### Implementing ReAct with LangGraph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", reason)
graph.add_node("tools", ToolNode(tools=[search, calculate]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

### Production Considerations

- **Token budgets**: Set max iterations to prevent runaway loops (typically 5-15).
- **Timeout handling**: Wrap agent execution with timeouts for production reliability.
- **Error recovery**: Catch tool failures gracefully; let the agent retry or skip.
- **Observability**: Log every Thought/Action/Observation step for debugging.
- **Cost control**: Track token usage per iteration; abort if budget exceeded.
- **Determinism**: Use `temperature=0` for reproducible agent behavior in tests.

### Common Mistakes

1. **No iteration limit** — agent loops forever on ambiguous queries.
2. **Overly broad tools** — agent struggles to choose the right tool.
3. **Missing error handling** — one tool failure crashes the entire agent.
4. **Ignoring token costs** — ReAct loops can consume 10-50x more tokens than single calls.
5. **Not validating tool outputs** — agent trusts malformed tool results.

### When to Use Agents vs. Chains

- Use **chains** when the workflow is predictable and linear.
- Use **agents** when the task requires dynamic decision-making.
- Use **agents** when the number of steps is unknown in advance.
- Prefer **chains** for latency-sensitive applications (agents add multiple LLM calls).

### Structured Output from Agents

Agents often need to return structured results rather than free-form text:

```python
from pydantic import BaseModel, Field

class AgentResult(BaseModel):
    answer: str = Field(description="Final answer to the user query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    sources_used: list[str] = Field(default_factory=list)
    steps_taken: int = Field(ge=0)

# Force structured output at the final step
def finalize(state: AgentState) -> dict:
    result = llm.with_structured_output(AgentResult).invoke(
        state["messages"] + [{"role": "system", "content": "Summarize your findings."}]
    )
    return {"final_result": result.model_dump()}
```

### Evaluation and Testing

Agent evaluation requires measuring both individual steps and end-to-end outcomes:

- **Step-level metrics**: Was the correct tool selected? Were arguments valid?
- **Trajectory evaluation**: Did the agent take an efficient path to the answer?
- **End-to-end accuracy**: Is the final answer correct and complete?
- **Cost efficiency**: How many LLM calls and tokens were consumed?

```python
def evaluate_agent_run(trajectory: list[dict], expected_answer: str) -> dict:
    steps = len(trajectory)
    tool_calls = sum(1 for s in trajectory if s.get("type") == "tool_call")
    final = trajectory[-1].get("content", "")
    correct = expected_answer.lower() in final.lower()
    return {
        "correct": correct,
        "steps": steps,
        "tool_calls": tool_calls,
        "efficiency": 1.0 / max(steps, 1),
    }
```

### Memory Integration

Agents benefit from both short-term and long-term memory:

- **Short-term (conversation buffer)**: The message history within a single run.
- **Long-term (persisted profile)**: User preferences, past interactions, weak areas stored across sessions.
- **Episodic memory**: Summaries of past agent runs that inform future decisions.

```python
class AgentStateWithMemory(TypedDict, total=False):
    messages: list[dict]
    user_profile: dict          # long-term: loaded at start
    episode_summaries: list[str] # episodic: past run summaries
    tool_results: list[str]
    final_answer: str
```

### Debugging Agent Failures

Common debugging workflow:
1. **Inspect the trace** — review every Thought/Action/Observation step.
2. **Check tool outputs** — verify tools returned expected formats.
3. **Examine routing decisions** — confirm conditional edges fired correctly.
4. **Review token usage** — identify if context window was exceeded.
5. **Test with deterministic inputs** — use `temperature=0` and fixed tool responses.

## Anti-Patterns

- Building agents for tasks that a simple prompt can solve.
- Giving agents access to dangerous tools without sandboxing.
- Using agents without observability — you cannot debug what you cannot see.
- Skipping evaluation — agent quality degrades silently without benchmarks.
- Allowing unlimited tool access without permission scoping.
- Not distinguishing between retriable and fatal errors in tool execution.

## References

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
- LangChain Agents documentation
- LangGraph prebuilt agent patterns
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
