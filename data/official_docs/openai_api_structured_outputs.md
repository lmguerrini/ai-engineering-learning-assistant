# OpenAI API & Structured Outputs

- **Official source**: https://platform.openai.com/docs/
- **Last refreshed**: 2026-05-11
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
| `stop` | `list[str]` | Up to 4 sequences where the model stops generating |
| `seed` | `int` | Deterministic sampling (best-effort; check `system_fingerprint`) |
| `stream` | `bool` | Enable server-sent event streaming |
| `logprobs` | `bool` | Return log probabilities for output tokens |

Message roles: `system`, `user`, `assistant`, `tool`.

> **Caveat**: `temperature` and `top_p` should not both be set away from defaults simultaneously — the API accepts it but results are unpredictable.

### Streaming Responses

Streaming reduces time-to-first-token and is recommended for user-facing applications:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
    # Check for tool calls in streaming mode
    if delta.tool_calls:
        # Accumulate tool call arguments across chunks
        pass
```

- Streamed chunks arrive as `ChatCompletionChunk` objects with partial `delta` fields.
- `finish_reason` is `None` on all chunks except the final one.
- Tool call arguments arrive split across multiple chunks — accumulate `delta.tool_calls[i].function.arguments` before parsing.
- Streaming is incompatible with `response_format: json_schema` in some model versions — test before deploying.

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

**Schema constraints for Structured Outputs**:
- All fields must have explicit types — no `anyOf`, `oneOf` at the top level.
- `additionalProperties: false` is required on all object schemas.
- Maximum schema depth: 5 levels of nesting.
- Recursive (self-referencing) schemas are supported but must use `$ref`.
- First request with a new schema incurs a one-time latency penalty for schema compilation (can be several seconds).

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
    dimensions=512,  # optional: reduce dimensionality (supported on v3 models)
)
vector = response.data[0].embedding  # list[float]
```

| Model | Dimensions | Max Input Tokens | Relative Cost |
|-------|-----------|-----------------|---------------|
| `text-embedding-3-small` | 1536 (default) | 8191 | 1x |
| `text-embedding-3-large` | 3072 (default) | 8191 | ~6.5x |
| `text-embedding-ada-002` | 1536 (fixed) | 8191 | ~1x (legacy) |

- Input accepts a single string or a list of strings (batch up to 2048 items).
- Returns float arrays suitable for cosine similarity search.
- The `dimensions` parameter enables Matryoshka dimensionality reduction on v3 models — shorter vectors trade accuracy for storage/speed.
- Inputs exceeding `8191` tokens are silently truncated — pre-check with `tiktoken`.

## Advanced Patterns

### Rate Limiting & Retry Strategy

OpenAI enforces rate limits on tokens-per-minute (TPM) and requests-per-minute (RPM). Limits vary by model and tier.

```python
import time
from openai import RateLimitError, APITimeoutError, APIConnectionError

def call_with_backoff(client, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = 2 ** attempt  # exponential backoff: 1, 2, 4, 8, 16s
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError):
            time.sleep(1)
    raise RuntimeError("Max retries exceeded")
```

- The `Retry-After` header (when present) gives the exact wait time — prefer it over fixed backoff.
- For batch workloads, track cumulative TPM and throttle proactively rather than reacting to 429s.
- The Python SDK has built-in retry (`max_retries` constructor parameter) but no jitter — add jitter for high-concurrency scenarios.

### Token Counting

```python
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4  # per-message overhead: role + content wrappers
        total += len(enc.encode(msg["content"]))
    total += 2  # assistant reply priming
    return total
```

- Token limits include both input and output: `context_window = input_tokens + max_tokens`.
- `gpt-4o-mini` has a 128K context window; `gpt-4o` also 128K.
- Cost scales linearly with token count — input tokens are typically cheaper than output tokens.

### Multi-Turn Conversation Management

- Keep conversation history in a list of message dicts; pass the full list on each call.
- Implement a sliding window or summarization strategy when conversation exceeds ~75% of context window.
- System messages count toward the token limit — keep them concise in long conversations.
- For function-calling loops, include `tool` messages with results; omitting them causes model confusion.

## Practical Implementation Notes

- Always set `max_tokens` to control cost and latency.
- Use `response_format` with `json_schema` for reliable structured extraction.
- Wrap API calls in retry logic with exponential backoff for `429`, `500`, `503` errors.
- Use `tiktoken` to count tokens before sending requests.
- API keys must be stored in environment variables, never hardcoded.
- Check `response.choices[0].finish_reason` — `"length"` means truncation occurred.
- Set `timeout` on the client constructor (default is 600s) — lower it for user-facing requests.
- Use `seed` parameter for reproducible outputs during evaluation; verify `system_fingerprint` matches across calls.
- For parallel tool calls, the model may return multiple `tool_calls` in a single response — execute all before continuing.

## Cost & Performance

- Monitor `usage.prompt_tokens` and `usage.completion_tokens` in every response to track cost.
- Cache identical requests (same messages + parameters) to avoid redundant API calls.
- Prefer `gpt-4o-mini` for high-volume, lower-complexity tasks; reserve `gpt-4o` for tasks requiring stronger reasoning.
- Embedding calls are significantly cheaper than chat completions — batch embed during ingestion, not at query time.
- Structured Outputs (`json_schema`) add minimal latency after initial schema compilation.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `json_object` returns non-JSON | System prompt doesn't mention JSON | Add explicit JSON instruction to system message |
| `429 Too Many Requests` | Rate limit exceeded | Add exponential backoff with jitter; check `Retry-After` header |
| Response truncated mid-sentence | `max_tokens` too low | Increase `max_tokens`; check `finish_reason == "length"` |
| Structured output schema error | Unsupported schema feature | Ensure `additionalProperties: false`; check nesting depth ≤5 |
| `context_length_exceeded` error | Input + max_tokens > model limit | Pre-count tokens with `tiktoken`; truncate or summarize context |
| Tool calls not appearing | `tool_choice="none"` or tools not provided | Set `tool_choice="auto"` or `"required"`; verify tools array |
| Inconsistent outputs across calls | No `seed` set; or `system_fingerprint` changed | Set `seed` parameter; check fingerprint for server-side changes |
| Streaming chunks have empty content | Normal for tool-call or metadata chunks | Check `delta.tool_calls` alongside `delta.content` |

## Common Mistakes

- Forgetting to instruct JSON output in the system prompt when using `json_object` mode.
- Not handling rate limit (`429`) errors with proper backoff.
- Setting temperature too high for structured output tasks.
- Ignoring `finish_reason` — `"length"` means the response was truncated.
- Sending overly large context without checking token limits via `tiktoken`.
- Using `json_object` mode when `json_schema` would guarantee the shape you need.
- Not accumulating streamed tool call arguments before JSON-parsing them.
- Mixing `temperature` and `top_p` modifications simultaneously.

## Related Project Usage

- `src/graphs/learn_nodes.py`: Uses chat completions for study guide generation.
- `src/graphs/quiz_nodes.py`: Uses chat completions for quiz generation.
- `src/kb/embeddings.py`: Uses embeddings API for document vectorization.
- `src/services/cost_tracker.py`: Tracks token usage and estimates cost.
- `src/services/retry.py`: Implements retry logic for API calls.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://developers.openai.com/api/docs

```
Home API Docs Guides and concepts for the OpenAI API API reference Endpoints, parameters, and responses Codex Docs Guides, concepts, and product docs for Codex Use cases Example workflows and tasks teams hand to Codex ChatGPT Apps SDK Build apps to extend ChatGPT Commerce Build commerce flows in ChatGPT Ads Publish and measure ads in ChatGPT Resources Showcase Demo apps to get inspired Blog Learnings and experiences from developers Cookbook Notebook examples for building with OpenAI models Learn Docs, videos, and demo apps for building with OpenAI Community Programs, meetups, and support for builders Start searching API Dashboard Search the API docs Search docs Suggested responses create reasoning_effort realtime prompt caching Primary navigation API API Reference Codex ChatGPT Resources Search docs Suggested responses create reasoning_effort realtime prompt caching Get started Overvi...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
