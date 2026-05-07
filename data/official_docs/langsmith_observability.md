# LangSmith Observability

- **Official source**: https://docs.smith.langchain.com/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs
- **Versions**: `langsmith>=0.1`

## When to Use

- Tracing and debugging LLM application runs.
- Monitoring latency, token usage, and error rates in production.
- Evaluating LLM output quality with datasets and scoring.

## Key Concepts

### Tracing

Automatic tracing captures the full call hierarchy for every LLM application run.

```
[Chain: learn_workflow] 230ms, 1,450 tokens
├── [Retriever: search_kb] 45ms
│   └── [Embedding: text-embedding-3-small] 30ms, 12 tokens
├── [LLM: gpt-4o-mini] 180ms, 1,438 tokens
│   ├── Input: "Generate a study guide about RAG..."
│   └── Output: {"topic": "RAG", "sections": [...]}
└── [Parser: JsonOutputParser] 2ms
```

- Enable with `LANGCHAIN_TRACING_V2=true` environment variable.
- Traces capture inputs, outputs, latency, and token usage per step.
- Each trace has a unique run ID for lookup and debugging.
- Traces are hierarchical: parent runs contain child runs (LLM calls, retriever calls, tool executions).
- Trace data is sent asynchronously — minimal impact on application latency.
- When tracing is disabled or misconfigured, the application continues functioning normally (traces are silently dropped).

### Environment Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGCHAIN_TRACING_V2` | Yes | Enable tracing (`true`/`false`) |
| `LANGCHAIN_API_KEY` | Yes | API key for LangSmith authentication |
| `LANGCHAIN_PROJECT` | No | Project name for grouping traces (default: `"default"`) |
| `LANGCHAIN_ENDPOINT` | No | API endpoint (default: `https://api.smith.langchain.com`) |

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_...
export LANGCHAIN_PROJECT=ai-tutor-dev
```

### Projects & Runs

- Projects group related traces (e.g., per environment or feature).
- Runs represent individual executions with full input/output capture.
- Filter runs by status, latency, token count, or custom metadata.
- Tag runs with metadata for categorization and search.

**Custom metadata and tagging**:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    metadata={"version": "v2.1", "user_tier": "premium"},
    tags=["learn-workflow", "production"],
    run_name="study_guide_generation",
)
result = app.invoke({"topic": "RAG"}, config)
```

- Tags are free-form strings; use consistent conventions (e.g., `workflow-name`, `environment`).
- Metadata supports arbitrary key-value pairs; values must be JSON-serializable.
- `run_name` overrides the default node/chain name in the trace UI.
- Use metadata for A/B testing: tag runs with experiment IDs for comparison.

### Programmatic Trace Access

```python
from langsmith import Client

client = Client()

# List recent runs in a project
runs = client.list_runs(
    project_name="ai-tutor-dev",
    filter='eq(status, "error")',  # filter syntax
    limit=50,
)

# Get detailed run by ID
run = client.read_run(run_id="...")
print(run.inputs, run.outputs, run.total_tokens, run.latency)

# Export runs for analysis
import pandas as pd
df = pd.DataFrame([{"id": r.id, "latency": r.latency, "tokens": r.total_tokens} for r in runs])
```

- Filter syntax supports `eq`, `has`, `gt`, `lt`, `and`, `or` operators.
- Use `client.list_runs()` for automated monitoring and alerting pipelines.

### Evaluation & Datasets

```python
from langsmith import Client

client = Client()

# Create dataset from examples
dataset = client.create_dataset("rag-eval-set")
client.create_examples(
    inputs=[{"question": "What is RAG?"}],
    outputs=[{"answer": "RAG combines retrieval with generation..."}],
    dataset_id=dataset.id,
)

# Run evaluation
results = client.run_on_dataset(
    dataset_name="rag-eval-set",
    llm_or_chain_factory=my_chain,
    evaluation=evaluators,
)
```

- Create datasets from production traces or manual examples.
- Built-in evaluators: correctness, helpfulness, relevance.
- Compare runs across different prompts, models, or configurations.

### Feedback & Annotations

```python
from langsmith import Client

client = Client()

# Attach feedback to a specific run
client.create_feedback(
    run_id="run-uuid-here",
    key="correctness",      # feedback category
    score=0.8,              # numeric score (0-1)
    comment="Mostly correct but missing RAG details",
)

# Binary feedback (thumbs up/down)
client.create_feedback(
    run_id="run-uuid-here",
    key="user_rating",
    score=1,  # 1 = positive, 0 = negative
)
```

- Attach human feedback (thumbs up/down, scores, comments) to runs.
- Use feedback to build evaluation datasets from production data.
- Track quality trends over time using feedback aggregation.
- Feedback keys are free-form strings — standardize across the team (e.g., `correctness`, `helpfulness`, `relevance`).
- Feedback can be attached programmatically from the application (e.g., after user thumbs-up in UI).

## Advanced Patterns

### Custom Evaluators

```python
from langsmith.evaluation import EvaluationResult, run_evaluator

@run_evaluator
def check_sources_cited(run, example) -> EvaluationResult:
    """Check if the response mentions source documents."""
    output = run.outputs.get("study_guide", "")
    has_sources = "Sources:" in output or "References:" in output
    return EvaluationResult(key="sources_cited", score=int(has_sources))
```

- Custom evaluators receive the `run` (actual) and `example` (expected) objects.
- Return `EvaluationResult` with a `key` (metric name) and `score`.
- Evaluators can be LLM-based (use a judge LLM) or deterministic (string matching, regex).

### Cost & Token Monitoring

- `run.total_tokens` gives combined input + output tokens for the entire trace.
- `run.prompt_tokens` and `run.completion_tokens` break down per-step costs.
- Aggregate token usage across runs with `client.list_runs()` for billing estimates.
- Set up alerts on runs exceeding token thresholds (e.g., > 10K tokens per request).

### Production Sampling

- In high-traffic production, trace a sample of requests to reduce cost:
  ```python
  import random
  import os
  
  if random.random() < 0.1:  # 10% sampling
      os.environ["LANGCHAIN_TRACING_V2"] = "true"
  else:
      os.environ["LANGCHAIN_TRACING_V2"] = "false"
  ```
- Alternatively, use the `tracing_enabled()` context manager for per-request control.
- Ensure sampling rate is sufficient for error detection (typically ≥10%).

## Practical Implementation Notes

- Set tracing env vars at application startup, not per-request.
- Use project names to separate `dev` / `staging` / `production` traces.
- Add custom metadata to runs for filtering: `config={"metadata": {"version": "v2"}}`.
- LangSmith is optional — the application must work without it configured.
- Check `LANGCHAIN_API_KEY` presence before enabling tracing features.
- LangSmith has a free tier with limited trace retention — check current limits for planning.
- Traces are immutable once created — attach feedback or annotations for corrections.
- For CI/CD, use `client.run_on_dataset()` to automate regression testing against eval datasets.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No traces appearing in UI | `LANGCHAIN_TRACING_V2` not set or API key missing | Verify both env vars; check `LANGCHAIN_ENDPOINT` |
| Traces incomplete (missing steps) | Exception in a node/step | Check application logs; LangSmith shows partial traces up to the error |
| High latency on first request | LangSmith SDK initialization | SDK initializes on first trace; subsequent requests are faster |
| `AuthenticationError` from LangSmith | Invalid or expired API key | Regenerate key in LangSmith UI; check for whitespace in env var |
| Traces in wrong project | `LANGCHAIN_PROJECT` not set | Explicitly set project name; default is `"default"` |
| Large trace payloads | Logging full document content in inputs | Truncate or summarize large inputs before passing to traced chains |

## Common Mistakes

- Enabling tracing without setting the API key (causes silent failures).
- Not separating projects by environment, mixing dev and production traces.
- Forgetting that tracing adds latency — disable in latency-critical paths.
- Logging sensitive data (PII, API keys) in trace inputs/outputs.
- Not using `run_name` — traces default to class names which are hard to distinguish.
- Forgetting to filter sensitive fields before tracing (input sanitization).

## Related Project Usage

- `src/config.py`: LangSmith settings (tracing flag, API key, project name).
- `src/services/observability.py`: Tracing configuration and status utility.
- `app.py`: Calls `configure_langsmith_tracing()` at startup.
