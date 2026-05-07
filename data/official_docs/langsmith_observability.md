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

- Attach human feedback (thumbs up/down, scores, comments) to runs.
- Use feedback to build evaluation datasets.
- Track quality trends over time using feedback aggregation.

## Practical Implementation Notes

- Set tracing env vars at application startup, not per-request.
- Use project names to separate `dev` / `staging` / `production` traces.
- Add custom metadata to runs for filtering: `config={"metadata": {"version": "v2"}}`.
- LangSmith is optional — the application must work without it configured.
- Check `LANGCHAIN_API_KEY` presence before enabling tracing features.

## Common Mistakes

- Enabling tracing without setting the API key (causes silent failures).
- Not separating projects by environment, mixing dev and production traces.
- Forgetting that tracing adds latency — disable in latency-critical paths.
- Logging sensitive data (PII, API keys) in trace inputs/outputs.

## Related Project Usage

- `src/config.py`: LangSmith settings (tracing flag, API key, project name).
- `src/services/observability.py`: Tracing configuration and status utility.
- `app.py`: Calls `configure_langsmith_tracing()` at startup.
