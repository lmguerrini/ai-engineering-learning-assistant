# Topic: Building Agents with Chains
# Sprint: 3
# Part: 2
# Tags: langchain, create_agent, agent-state, toolruntime, middleware, checkpointers, thread-id, production-agents, command-update, human-in-the-loop

## Overview
Building agents with chains means moving from simple message-based agents to production-ready systems with structured state, tool access to shared state, persistence, middleware, validation, authorization, logging, and error handling.

This section focuses on LangChain’s `create_agent`, custom agent state, `ToolRuntime`, `Command(update={...})`, checkpointers, and middleware patterns.

## Key Concepts

- **Production-Ready Agent**  
  An agent designed to handle persistence, validation, authorization, guardrails, retries, logging, and real-world complexity.

- **`create_agent`**  
  LangChain’s high-level API for building conversational agents with tools, memory, and execution flow.

- **Agent State**  
  Structured working memory that stores domain-specific data beyond raw message history.

- **Messages-Only Memory Limitation**  
  Message history alone is not enough for agents that need structured information like carts, findings, account status, or tasks.

- **Custom State**  
  A developer-defined extension of `AgentState` that adds domain-specific fields.

- **Checkpointer + `thread_id`**  
  Mechanism for persisting agent state across turns in the same conversation thread.

- **`ToolRuntime`**  
  Runtime object that lets tools read internal state or context without exposing that data in the LLM-facing tool schema.

- **State vs Context**  
  State is mutable conversation/task data. Context is stable configuration such as user ID, API keys, subscription tier, or database connections.

- **`Command(update={...})`**  
  Mechanism for tools to write updates back into agent state.

- **Middleware**  
  A control layer that intercepts agent execution to validate, authorize, log, retry, modify, block, or inspect behavior.

- **Middleware Hooks**  
  Main hooks include `before_agent`, `before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`, and `after_agent`.

- **Built-In Middleware**  
  Reusable middleware such as summarization, human-in-the-loop, and PII handling.

## How It Works

### 1. Start from a Basic Conversational Agent
A simple `create_agent` setup usually tracks conversation messages.

This is useful for chat but insufficient for complex production workflows.

### 2. Extend Memory with Custom Agent State
Developers extend agent state with structured fields.

Examples:
- cart items
- research findings
- task list
- budget
- account status
- user preferences

### 3. Persist State Across Turns
A checkpointer and `thread_id` allow state to survive across invocations.

Flow:
1. load state for `thread_id`
2. process new user request
3. update state
4. save state again

### 4. Give Tools Safe Access to State
`ToolRuntime` lets tools read current state or context without exposing those internals as tool arguments.

This keeps tool schemas cleaner and safer.

### 5. Let Tools Update State
Tools can return `Command(update={...})`.

This allows a tool to mutate state in a controlled way.

Example:
- add item to cart
- mark task complete
- append research finding
- update workflow status

### 6. Add Middleware Around Execution
Middleware can intercept execution to:

- validate input
- load user data
- trim messages
- inject context
- choose models dynamically
- authorize tools
- log actions
- validate outputs
- filter unsafe content
- save analytics

### 7. Respect Middleware Order
Execution order matters:

- `before_*` hooks run first-to-last
- `wrap_*` hooks nest around the intercepted call
- `after_*` hooks run last-to-first

Incorrect ordering can break control logic.

### 8. Combine State, Tools, and Middleware
A production-ready agent can:

1. validate user input
2. load persistent state
3. choose tools
4. update state through tools
5. validate output
6. log metrics and audit data

## Example

### Shopping Assistant Flow
User:
"Add a laptop to my cart and check if I can afford it."

Possible flow:
1. `before_agent` validates authentication and loads budget.
2. `before_model` trims old messages and injects preferences.
3. model calls `add_item_to_cart("laptop")`.
4. `wrap_tool_call` authorizes and logs the tool call.
5. tool reads cart from state using `ToolRuntime`.
6. tool returns `Command(update={"cart_items": ["laptop"]})`.
7. model calls `check_budget_remaining(800)`.
8. tool computes remaining budget.
9. `after_model` validates output and checks for PII.
10. `after_agent` saves metrics.
11. assistant returns final answer.

### State Persistence Example
A research agent invoked with the same `thread_id` can remember earlier findings.

User follow-up:
"What have we learned so far?"

The agent answers using persisted structured state.

### Middleware Chain Example
A customer service agent with layered middleware:

1. `before_agent`: authenticate user, load subscription tier from database
2. `before_model`: trim message history to last 20 messages, inject user preferences
3. `wrap_tool_call`: check if tool is authorized for user tier, log call to audit trail
4. `after_model`: scan output for PII, validate response format
5. `after_agent`: record response latency, update analytics dashboard

## When to Use

- **Custom Agent State**
  - structured task data
  - carts
  - findings
  - preferences
  - tickets
  - budgets

- **`ToolRuntime`**
  - when tools need state or context but should not expose internals to the LLM

- **Middleware**
  - validation
  - authorization
  - rate limiting
  - retries
  - logging
  - PII handling
  - human approval

- **Checkpointers and `thread_id`**
  - when state must persist across turns

- **Built-In Middleware**
  - when common production controls need fast implementation

## Common Mistakes

- **Relying only on message history**
  - Complex agents need structured state.

- **Exposing internal state to the LLM**
  - Internal data should not always appear in tool schemas.

- **Confusing state and context**
  - Mutable data belongs in state; stable configuration belongs in context.

- **Skipping persistence**
  - Without a checkpointer and `thread_id`, state will not survive across turns.

- **Read-only tools only**
  - Many production workflows require tools to update state.

- **No middleware**
  - Demo agents often miss validation, retries, authorization, and audit trails.

- **Ignoring middleware order**
  - Hook order affects behavior and correctness.

- **No error handling in tools**
  - Tools that throw unhandled exceptions crash the agent loop. Always return structured error responses.

## Best Practices

- Extend `AgentState` with domain-specific fields early — retrofitting state later is harder.
- Use `ToolRuntime` to give tools safe read access to state without exposing internals in tool schemas.
- Use `Command(update={...})` for controlled state mutations from tools instead of direct state manipulation.
- Start with `InMemorySaver` for development, then migrate to `SqliteSaver` or `PostgresSaver` for production persistence.
- Layer middleware in the correct order: authentication → validation → execution → output checking → logging.
- Always include error handling in tools — return structured error responses instead of raising exceptions.
- Test middleware hooks independently with unit tests before combining them.
- Log every state mutation and tool call for debugging and audit compliance.

## Related Concepts

- AI Agents  
- Function Calling  
- Tool Use  
- LangGraph  
- Checkpointers  
- Agent State  
- Middleware  
- Human-in-the-Loop  
- PII Redaction  
- Production Agent Reliability  
- Error Handling and Retries  
- Observability and Tracing  