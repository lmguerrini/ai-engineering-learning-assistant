# Topic: Function Calling, Tool Use, and Model Context Protocol
# Sprint: 2
# Part: 4
# Tags: function-calling, tools, langchain, create_agent, checkpointers, tool-evaluation, mcp, model-context-protocol, tool-calling, langgraph

## Overview
Function calling lets large language models move beyond text generation by invoking external tools, APIs, databases, calculators, and other systems.

This section explains tool execution loops, LangChain’s `create_agent` pattern, memory through checkpointers, tool-calling evaluation, and the Model Context Protocol (MCP) for interoperable tool integrations.

Function calling is a bridge between prompt-based LLM applications and agentic systems.

## Key Concepts

- **Function Calling**  
  A mechanism that lets an LLM request execution of a predefined function or tool.

- **Tools**  
  Callable capabilities exposed to an LLM, such as calculators, APIs, search, databases, or code execution.

- **Tool Execution Loop**  
  The model selects a tool, provides structured arguments, the application executes it, and the result is returned to the model.

- **LangChain `create_agent`**  
  A high-level agent pattern that combines model calls, tool use, memory, and execution flow.

- **Checkpointers**  
  Components that persist conversation or graph state across turns using a `thread_id`.

- **Saver Implementations**  
  - `InMemorySaver` → temporary state  
  - `SqliteSaver` → local persistence  
  - `PostgresSaver` → production-grade persistence  

- **Tool Selection**  
  Whether the model chose the correct tool.

- **Parameter Extraction**  
  Whether the model passed correct arguments to the selected tool.

- **LLM-as-a-Judge for Tools**  
  Using an LLM to evaluate tool usage when deterministic matching is insufficient.

- **Model Context Protocol (MCP)**  
  An open protocol for connecting AI applications to external tools and data sources.

- **MCP Client and Server**  
  Servers expose capabilities; clients discover and use them.

- **MCP Capabilities**  
  Main types: tools, resources, and prompts.

- **MCP Transports**  
  Communication methods such as `stdio`, SSE, and Streamable HTTP.

## How It Works

### 1. Define Tools
A developer exposes tools to the model.

Each tool should include:
- clear name
- clear description
- input schema
- function implementation
- expected behavior

### 2. Let the Model Choose
The LLM decides whether to:
- answer directly
- call a tool

Tool use is appropriate when the answer requires external data, computation, or action.

### 3. Execute the Tool
The application parses:
- tool name
- arguments

Then it runs the corresponding function, API call, or backend service.

### 4. Return Tool Result
The result is sent back to the model.

The model then uses the result to produce a final natural language answer.

### 5. Use `create_agent`
LangChain’s `create_agent` automates:
- tool execution loop
- message handling
- memory integration
- agent runtime

It is powered by LangGraph under the hood.

### 6. Persist Memory with Checkpointers
A checkpointer:
1. loads previous state for a `thread_id`
2. adds the new interaction
3. saves updated state

This supports:
- multi-turn conversations
- multiple users
- persistence across restarts

### 7. Evaluate Tool Calling
Tool-calling evaluation checks:

- Did the agent choose the correct tool?
- Did it pass the correct parameters?
- Was the final answer useful?

Evaluation datasets should include:
- expected tool
- expected arguments
- difficulty label
- no-tool-expected examples

### 8. Use MCP for Interoperability
MCP standardizes tool and data-source integration.

Flow:
1. MCP server exposes capabilities
2. MCP client connects
3. client discovers tools/resources/prompts
4. model uses capabilities through the protocol

MCP reduces the need for custom integrations between every AI app and every tool.

## Example

### Calculator Tool Example
User:
"Add 5 + 7."

Flow:
1. model selects calculator tool
2. model passes `5` and `7`
3. application executes addition
4. tool returns `12`
5. assistant answers naturally

### Food Delivery Tool Example
User:
"Order 2 pizzas to 123 Main Street."

Flow:
1. model extracts item, quantity, and address
2. model calls ordering tool
3. backend confirms order
4. assistant confirms delivery details

### Tool Evaluation Example
Query:
"Average salary for data scientists in Berlin."

Expected:
- tool: `compare_salaries`
- role: `data scientist`
- location: `Berlin`

Failures:
- wrong tool: `search_jobs`
- wrong parameter: `Germany` instead of `Berlin`

### MCP Example
An MCP server exposes:
- `add`
- `multiply`

A client connects over Streamable HTTP and invokes those tools through the MCP protocol.

## When to Use

- **Function Calling**
  - APIs
  - database queries
  - calculations
  - search
  - external actions

- **LangChain `create_agent`**
  - conversational apps with tools
  - memory-enabled assistants
  - production-friendly agent workflows

- **Checkpointers**
  - multi-turn conversations
  - multiple users
  - persistent thread state

- **Tool Evaluation**
  - when correctness of tool selection and parameters matters
  - when testing agents in CI or regression suites

- **MCP**
  - user-extensible AI applications
  - cross-platform tool ecosystems
  - dynamic tool discovery
  - shared integrations across AI clients

- **Avoid MCP**
  - when tools are internal, simple, and tightly controlled in one application

## Common Mistakes

- **Generic tools**
  - Broad tools are harder for models to select correctly.

- **Poor tool descriptions**
  - Vague descriptions cause wrong tool selection.

- **Weak schemas**
  - Ambiguous parameters lead to bad argument extraction.

- **Manual-only testing**
  - A few manual tests miss wrong parameters and over-eager tool use.

- **No “no tool expected” cases**
  - Agents may overuse tools if this is not tested.

- **Assuming correct tool use guarantees good answers**
  - Final response quality still matters.

- **Too many MCP servers**
  - Tool descriptions consume context window space.

- **Trusting community MCP servers blindly**
  - Servers can introduce security and supply-chain risks.

- **Confusing curated with verified**
  - Directory listings do not guarantee security auditing.

## Best Practices

- Write clear, specific tool descriptions — the model uses them to decide when and how to call tools.
- Define strict input schemas with types, constraints, and descriptions for every parameter.
- Include "no tool expected" test cases in evaluation datasets to catch over-eager tool use.
- Use checkpointers for any multi-turn application; start with `InMemorySaver` for development, then migrate to `SqliteSaver` or `PostgresSaver` for production.
- Log every tool call with input arguments, output, and latency for debugging and audit trails.
- Validate tool outputs before returning them to the model — malformed results cause cascading errors.
- Prefer MCP when tools must be shared across multiple AI applications; use direct tool binding for internal-only tools.
- Test tool-calling agents with automated evaluation datasets, not just manual spot checks.

## Related Concepts

- Prompt Evaluation  
- LLM-as-a-Judge  
- LangChain Tools  
- LangGraph  
- Conversational Memory  
- Checkpointers  
- Function Schemas  
- Structured Output  
- Retrieval-Augmented Generation (RAG)  
- MCP Tools, Resources, and Prompts  
- MCP Transports (`stdio`, SSE, Streamable HTTP)  