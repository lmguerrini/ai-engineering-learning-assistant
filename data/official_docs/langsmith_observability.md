# LangSmith Observability

- **Official source**: https://docs.smith.langchain.com/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Tracing and debugging LLM application runs.
- Monitoring latency, token usage, and error rates in production.
- Evaluating LLM output quality with datasets and scoring.

## Key Concepts

### Tracing

- Automatic tracing when `LANGCHAIN_TRACING_V2=true` is set.
- Traces capture inputs, outputs, latency, and token usage per step.
- Nested spans show the full call hierarchy (chain → LLM → tool).
- Each trace has a unique run ID for lookup and debugging.

### Environment Configuration

- `LANGCHAIN_TRACING_V2`: Enable/disable tracing (`true`/`false`).
- `LANGCHAIN_API_KEY`: API key for LangSmith authentication.
- `LANGCHAIN_PROJECT`: Project name for grouping traces.
- `LANGCHAIN_ENDPOINT`: API endpoint (default: `https://api.smith.langchain.com`).

### Projects & Runs

- Projects group related traces (e.g., per environment or feature).
- Runs represent individual executions with full input/output capture.
- Filter runs by status, latency, token count, or custom metadata.
- Tag runs with metadata for categorization and search.

### Evaluation & Datasets

- Create datasets from production traces or manual examples.
- Run evaluations with custom scoring functions.
- Built-in evaluators: correctness, helpfulness, relevance.
- Compare runs across different prompts, models, or configurations.

### Feedback & Annotations

- Attach human feedback (thumbs up/down, scores, comments) to runs.
- Use feedback to build evaluation datasets.
- Track quality trends over time using feedback aggregation.

## Practical Implementation Notes

- Set tracing env vars at application startup, not per-request.
- Use project names to separate dev/staging/production traces.
- Add custom metadata to runs for filtering: `run.metadata = {"version": "v2"}`.
- LangSmith is optional — application must work without it configured.
- Check `LANGCHAIN_API_KEY` presence before enabling tracing.

## Common Mistakes

- Enabling tracing without setting the API key (causes silent failures).
- Not separating projects by environment, mixing dev and production traces.
- Forgetting that tracing adds latency — disable in latency-critical paths.
- Logging sensitive data (PII, API keys) in trace inputs/outputs.

## Related Project Usage

- `src/config.py`: LangSmith settings (tracing flag, API key, project name).
- `src/services/observability.py`: Tracing configuration and status utility.
- `app.py`: Calls `configure_langsmith_tracing()` at startup.
