# Tool Calling & Function Calling

- **Official source**: https://platform.openai.com/docs/guides/function-calling, https://python.langchain.com/docs/concepts/tool_calling/
- **Last refreshed**: 2025-05-08
- **source_type**: official_docs
- **Versions**: `openai>=1.0`, `langchain-core>=0.2`

## When to Use

- Enabling LLMs to interact with external systems (APIs, databases, file systems).
- Structured data extraction from natural language.
- Building agents that need to take real-world actions.

## Key Concepts

### Function Calling (OpenAI)

Function calling lets the model output a structured JSON object matching a function schema, rather than free-form text.

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
    tools=tools,
    tool_choice="auto",
)
```

### LangChain Tool Abstraction

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the internal knowledge base for relevant documents.

    Args:
        query: The search query string.
        top_k: Number of results to return.
    """
    results = vector_store.similarity_search(query, k=top_k)
    return "\n".join(r.page_content for r in results)
```

Key points:
- The `@tool` decorator converts a function into a LangChain `Tool` object.
- The docstring becomes the tool description sent to the LLM.
- Type hints are converted to the JSON schema automatically.
- Always provide clear, specific descriptions — the LLM uses them to decide when to call the tool.

### Tool Schemas and Validation

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")

@tool(args_schema=SearchInput)
def search(query: str, top_k: int = 5) -> str:
    """Search documents."""
    return do_search(query, top_k)
```

- Use Pydantic models for complex input validation.
- `Field` descriptions improve LLM tool selection accuracy.
- Validation runs before tool execution — malformed calls fail fast.

### Parallel Tool Calling

Modern models can request multiple tool calls in a single response:

```python
# Response may contain multiple tool_calls
for tool_call in response.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = execute_tool(name, args)
```

### Tool Result Handling

```python
# Feed tool results back to the model
messages.append(response.choices[0].message)  # assistant message with tool_calls
for tool_call in response.choices[0].message.tool_calls:
    result = execute_tool(tool_call)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    })
# Continue the conversation
follow_up = client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Production Considerations

- **Timeout tools**: Wrap external API calls with timeouts to prevent hanging.
- **Rate limiting**: Implement rate limits on tools that call external services.
- **Input sanitization**: Validate and sanitize tool inputs before execution.
- **Error messages**: Return structured error messages so the LLM can recover.
- **Idempotency**: Design tools to be safely retryable where possible.
- **Logging**: Log every tool invocation with inputs, outputs, and latency.
- **Sandboxing**: Never let tools execute arbitrary code without sandboxing.

### Common Mistakes

1. **Vague tool descriptions** — LLM cannot distinguish between similar tools.
2. **Missing error handling** — tool exceptions crash the agent loop.
3. **Trusting LLM arguments blindly** — always validate inputs.
4. **Not handling parallel calls** — assuming only one tool call per response.
5. **Exposing dangerous operations** — delete/write tools without confirmation.

### Tool Design Principles

| Principle | Description |
|---|---|
| **Single responsibility** | Each tool does one thing well |
| **Clear naming** | Tool name describes the action precisely |
| **Rich descriptions** | Docstring explains when and why to use the tool |
| **Typed parameters** | Use type hints and Pydantic for validation |
| **Graceful errors** | Return error strings instead of raising exceptions |
| **Minimal scope** | Expose only necessary functionality |

### Advanced: Dynamic Tool Selection

For agents with many tools, dynamically filter the tool set per query:

```python
def select_tools(query: str, all_tools: list, max_tools: int = 5) -> list:
    """Select the most relevant tools for a given query using embeddings."""
    query_embedding = embed(query)
    scored = []
    for t in all_tools:
        desc_embedding = embed(t.description)
        score = cosine_similarity(query_embedding, desc_embedding)
        scored.append((score, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:max_tools]]
```

Benefits:
- Reduces token usage by sending fewer tool schemas.
- Improves selection accuracy by removing irrelevant options.
- Scales to hundreds of tools without degrading LLM performance.

### Testing Tools

```python
import pytest

def test_search_tool_returns_results():
    result = search_knowledge_base.invoke({"query": "LangGraph", "top_k": 3})
    assert isinstance(result, str)
    assert len(result) > 0

def test_search_tool_validates_input():
    with pytest.raises(ValidationError):
        search_knowledge_base.invoke({"query": "", "top_k": -1})

def test_tool_schema_matches_expected():
    schema = search_knowledge_base.args_schema.model_json_schema()
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
```

### Tool Calling vs. Structured Output

| Feature | Tool Calling | Structured Output |
|---|---|---|
| Purpose | Execute actions | Extract data |
| Returns to LLM | Yes (observation loop) | No (final result) |
| Multiple calls | Yes (parallel) | Single response |
| Use case | Agents, workflows | Parsing, classification |
| Error handling | Tool returns error string | Validation error |

Choose **tool calling** when the LLM needs to act and observe results.
Choose **structured output** when you need a typed response without side effects.

## Anti-Patterns

- Creating one mega-tool that does everything.
- Using tool descriptions as prompt injection vectors.
- Returning raw database rows or API responses without formatting.
- Not testing tools independently before integrating with agents.
- Allowing tools to mutate shared state without synchronization.
- Ignoring tool call latency in agent timeout budgets.
