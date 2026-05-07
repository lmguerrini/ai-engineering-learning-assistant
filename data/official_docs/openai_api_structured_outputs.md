# OpenAI API & Structured Outputs

- **Official source**: https://platform.openai.com/docs/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs
- **Versions**: API `v1`, Python SDK `openai>=1.0`

## When to Use

- Calling OpenAI chat completions, embeddings, or function-calling endpoints.
- Generating structured (JSON) outputs from LLMs.
- Integrating OpenAI models into agentic pipelines.

## Key Concepts

### Chat Completions API

Primary endpoint: `POST /v1/chat/completions`.

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."},
    ],
    temperature=0.3,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`) |
| `messages` | `list[dict]` | Conversation history with `role` and `content` |
| `temperature` | `float` | Randomness: `0` = deterministic, `1` = creative |
| `max_tokens` | `int` | Maximum tokens in the response |
| `top_p` | `float` | Nucleus sampling alternative to temperature |

Message roles: `system`, `user`, `assistant`, `tool`.

### Structured Outputs (JSON Mode)

Two approaches for JSON output:

**JSON mode** — guarantees valid JSON, no schema enforcement:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "Return JSON with keys: topic, summary"}],
    response_format={"type": "json_object"},
)
```

**Structured Outputs** — enforces a strict JSON schema:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "study_guide",
            "schema": StudyGuide.model_json_schema(),
        },
    },
)
```

> **Note**: The system prompt must instruct the model to produce JSON when using `json_object` mode. Structured Outputs with `json_schema` handle this automatically.

### Function Calling / Tool Use

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # "auto", "required", or {"type": "function", "function": {"name": "..."}}
)

# Handle tool calls
for tool_call in response.choices[0].message.tool_calls or []:
    result = execute_function(tool_call.function.name, tool_call.function.arguments)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
```

### Embeddings API

```python
response = client.embeddings.create(
    model="text-embedding-3-small",  # 1536 dims; "text-embedding-3-large" = 3072 dims
    input=["What is retrieval-augmented generation?"],
)
vector = response.data[0].embedding  # list[float]
```

- Input accepts a single string or a list of strings (batch).
- Returns float arrays suitable for cosine similarity search.

## Practical Implementation Notes

- Always set `max_tokens` to control cost and latency.
- Use `response_format` with `json_schema` for reliable structured extraction.
- Wrap API calls in retry logic with exponential backoff for `429`, `500`, `503` errors.
- Use `tiktoken` to count tokens before sending requests.
- API keys must be stored in environment variables, never hardcoded.
- Check `response.choices[0].finish_reason` — `"length"` means truncation occurred.

## Common Mistakes

- Forgetting to instruct JSON output in the system prompt when using `json_object` mode.
- Not handling rate limit (`429`) errors with proper backoff.
- Setting temperature too high for structured output tasks.
- Ignoring `finish_reason` — `"length"` means the response was truncated.
- Sending overly large context without checking token limits via `tiktoken`.

## Related Project Usage

- `src/graphs/learn_nodes.py`: Uses chat completions for study guide generation.
- `src/graphs/quiz_nodes.py`: Uses chat completions for quiz generation.
- `src/kb/embeddings.py`: Uses embeddings API for document vectorization.
- `src/services/cost_tracker.py`: Tracks token usage and estimates cost.
- `src/services/retry.py`: Implements retry logic for API calls.
