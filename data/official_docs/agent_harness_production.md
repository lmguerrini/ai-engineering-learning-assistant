# Agent Harness & Production Deployment

- **Official source**: https://langchain-ai.github.io/langgraph/concepts/deployment/, https://python.langchain.com/docs/concepts/architecture/
- **Last refreshed**: 2026-05-11
- **source_type**: official_docs
- **Versions**: `langgraph>=0.2`, `langchain>=0.2`

## When to Use

- Wrapping agent workflows in production-ready infrastructure.
- Building testable, observable, and maintainable agent systems.
- Deploying agents with proper error handling, retries, and monitoring.

## Key Concepts

### What Is an Agent Harness?

An agent harness is the infrastructure layer that wraps an agent's core logic, providing:
1. **Configuration management** — API keys, model selection, feature flags.
2. **Error handling** — retries, fallbacks, graceful degradation.
3. **Observability** — tracing, logging, metrics collection.
4. **Cost control** — token budgets, rate limiting, usage tracking.
5. **Testing infrastructure** — mocking, fixtures, integration test patterns.

### Configuration Management

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str = ""
    app_default_model: str = "gpt-4o-mini"
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    max_tokens_per_request: int = 16384
    cache_ttl_seconds: int = 3600

    model_config = {"env_file": ".env", "extra": "ignore"}

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Best practices**:
- Use environment variables for secrets; never hardcode API keys.
- Validate settings at startup; fail fast on missing required config.
- Use `lru_cache` for singleton settings to avoid repeated parsing.
- Support `.env` files for local development.

### Retry and Fallback Patterns

```python
import time
from typing import Callable, TypeVar

T = TypeVar("T")

def with_retry(
    callable: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    handled_exceptions: tuple = (Exception,),
) -> T:
    """Execute with exponential backoff retry."""
    for attempt in range(max_attempts):
        try:
            return callable()
        except handled_exceptions as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    raise RuntimeError("Unreachable")
```

**Fallback chain**:
```python
def generate_with_fallback(prompt: str) -> str:
    try:
        return call_primary_model(prompt)  # GPT-4o
    except Exception:
        try:
            return call_fallback_model(prompt)  # GPT-4o-mini
        except Exception:
            return STATIC_FALLBACK_RESPONSE
```

### Caching Layer

```python
import hashlib
import json

def build_cache_key(*parts: str) -> str:
    """Build a deterministic cache key from components."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()

class CacheService:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        if key in self._store:
            ts, value = self._store[key]
            if time.time() - ts < self._ttl:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)
```

**Cache invalidation strategies**:
- **Prompt versioning**: Include a version string in cache keys; bump on prompt changes.
- **TTL-based**: Expire entries after a fixed duration.
- **Content-hash**: Hash the prompt + retrieval context for content-aware caching.

### Token and Cost Tracking

```python
def build_usage_record(model: str, operation: str, usage: dict) -> dict:
    """Build a standardized usage record for cost tracking."""
    # Approximate costs per 1M tokens (as of 2024)
    COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    rates = COSTS.get(model, {"input": 0.15, "output": 0.60})
    input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * rates["input"]
    output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * rates["output"]

    return {
        "model": model,
        "operation": operation,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "estimated_cost_usd": input_cost + output_cost,
    }
```

### Testing Agent Systems

**Unit testing nodes**:
```python
def test_retrieve_sources():
    state = {"query": "RAG", "trace": [], "style": "detailed"}
    with patch("src.kb.retrieval.retrieve_documents") as mock:
        mock.return_value = [Document(content="RAG content", metadata={})]
        result = retrieve_sources(state)
    assert len(result["retrieved_docs"]) == 1
```

**Integration testing workflows**:
```python
def test_full_workflow_no_api_key():
    """Workflow completes with fallback when no API key is set."""
    result = run_learn_workflow("AI Agents")
    assert result["study_guide"] is not None
    assert any("fallback" in t for t in result["trace"])
```

**Testing patterns**:
- Mock external services (OpenAI, vector stores) in unit tests.
- Use integration tests with real services sparingly (CI/CD only).
- Test error paths: missing API keys, network failures, malformed responses.
- Test cache behavior: hits, misses, invalidation.
- Test state transitions: verify routing logic with different state combinations.

### Production Deployment Checklist

| Category | Item | Status |
|---|---|---|
| **Config** | All secrets in env vars | ☐ |
| **Config** | Settings validated at startup | ☐ |
| **Reliability** | Retry logic on LLM calls | ☐ |
| **Reliability** | Fallback responses for failures | ☐ |
| **Observability** | LangSmith tracing enabled | ☐ |
| **Observability** | Structured logging (loguru) | ☐ |
| **Cost** | Token usage tracked per request | ☐ |
| **Cost** | Budget alerts configured | ☐ |
| **Caching** | Response caching with TTL | ☐ |
| **Caching** | Cache version bumped on prompt changes | ☐ |
| **Testing** | Unit tests for all nodes | ☐ |
| **Testing** | Integration tests for workflows | ☐ |
| **Security** | Input validation on user inputs | ☐ |
| **Security** | Rate limiting on API endpoints | ☐ |

### Common Mistakes

1. **No retry logic** — single LLM failures crash the entire workflow.
2. **Hardcoded API keys** — security risk and deployment friction.
3. **No cost tracking** — surprise bills from runaway token usage.
4. **Missing fallbacks** — system is completely unavailable during outages.
5. **Untested error paths** — failures in production reveal untested code paths.
6. **No cache versioning** — stale cached responses after prompt updates.

## Anti-Patterns

- Building agents without observability from day one.
- Deploying without load testing or cost projections.
- Using production API keys in development/testing.
- Not implementing graceful shutdown for long-running agent tasks.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://docs.langchain.com/oss/python/langchain/overview

```
Skip to main content Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets > Docs by LangChain home page Open source Search... ⌘ K Ask AI GitHub Try LangSmith Try LangSmith Search... Navigation LangChain overview Deep Agents LangChain LangGraph Integrations Learn Reference Contribute Python Overview Get started Install Quickstart Changelog Philosophy Core components Agents Models Messages Tools Short-term memory Streaming Structured output Middleware Overview Prebuilt middleware Custom middleware Frontend Overview Patterns Integrations Advanced usage Guardrails Runtime Context engineering Model Context Protocol (MCP) Human-in-the-loop Multi-agent Retrieval Long-term memory Agent development LangSmith Studio Test Agent Chat UI Deploy with LangSmith Deployment Observability On this page Create an agent Core benefits LangChain overview Copy page LangCha...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
