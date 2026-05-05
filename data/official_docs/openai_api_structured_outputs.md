# OpenAI API & Structured Outputs

- **Official source**: https://platform.openai.com/docs/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Calling OpenAI chat completions, embeddings, or function-calling endpoints.
- Generating structured (JSON) outputs from LLMs.
- Integrating OpenAI models into agentic pipelines.

## Key Concepts

### Chat Completions API

- Primary endpoint: `POST /v1/chat/completions`.
- Messages array with roles: `system`, `user`, `assistant`, `tool`.
- Temperature controls randomness (0 = deterministic, 1 = creative).
- `max_tokens` limits response length.
- `top_p` is an alternative to temperature (nucleus sampling).

### Structured Outputs (JSON Mode)

- Set `response_format={"type": "json_object"}` to force JSON output.
- The system prompt must instruct the model to produce JSON.
- Structured Outputs with `response_format={"type": "json_schema", "json_schema": {...}}` enforce a strict schema.
- Pydantic models can generate the JSON schema automatically.

### Function Calling / Tool Use

- Define `tools` array with function name, description, and parameters JSON schema.
- The model returns `tool_calls` with function name and arguments.
- Application executes the function and returns results via `tool` role messages.
- `tool_choice` can be `auto`, `required`, or a specific function name.

### Embeddings API

- Endpoint: `POST /v1/embeddings`.
- Model `text-embedding-3-small` (1536 dims) or `text-embedding-3-large` (3072 dims).
- Input can be a string or array of strings.
- Returns float arrays suitable for cosine similarity search.

## Practical Implementation Notes

- Always set a reasonable `max_tokens` to control cost and latency.
- Use `response_format` with JSON schema for reliable structured extraction.
- Wrap API calls in retry logic with exponential backoff for transient errors (429, 500, 503).
- Use `tiktoken` library to count tokens before sending requests.
- API keys should be stored in environment variables, never hardcoded.

## Common Mistakes

- Forgetting to instruct JSON output in the system prompt when using JSON mode.
- Not handling rate limit (429) errors with proper backoff.
- Setting temperature too high for structured output tasks.
- Ignoring `finish_reason` — `length` means the response was truncated.
- Sending overly large context without checking token limits.

## Related Project Usage

- `src/graphs/learn_nodes.py`: Uses chat completions for study guide generation.
- `src/graphs/quiz_nodes.py`: Uses chat completions for quiz generation.
- `src/kb/embeddings.py`: Uses embeddings API for document vectorization.
- `src/services/cost_tracker.py`: Tracks token usage and estimates cost.
- `src/services/retry.py`: Implements retry logic for API calls.
