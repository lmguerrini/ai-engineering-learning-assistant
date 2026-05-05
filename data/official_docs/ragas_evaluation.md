# RAGAs Evaluation

- **Official source**: https://docs.ragas.io/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Evaluating the quality of RAG pipeline outputs.
- Measuring retrieval relevance, answer faithfulness, and context quality.
- Benchmarking RAG improvements across iterations.

## Key Concepts

### Core Metrics

- **Faithfulness**: Measures whether the answer is grounded in the retrieved context.
- **Answer Relevancy**: Measures how well the answer addresses the question.
- **Context Precision**: Measures whether relevant items are ranked higher in retrieved contexts.
- **Context Recall**: Measures whether all relevant information was retrieved.

### Evaluation Dataset

- Each sample needs: `question`, `answer`, `contexts`, and optionally `ground_truth`.
- Use `datasets.Dataset.from_dict(...)` or HuggingFace dataset format.
- Ground truth is required for context recall but optional for other metrics.
- Minimum ~20-50 samples recommended for statistically meaningful evaluation.

### Running Evaluation

- `evaluate(dataset, metrics=[faithfulness, answer_relevancy, ...])`.
- Returns a `Result` object with per-sample and aggregate scores.
- Scores range from 0.0 to 1.0 (higher is better).
- Evaluation requires an LLM (uses OpenAI by default for judging).

### Custom Metrics

- Extend `Metric` base class with `score` method.
- Can use LLM-based or deterministic scoring.
- Register custom metrics in the evaluation pipeline.

## Practical Implementation Notes

- Start with faithfulness and answer relevancy as baseline metrics.
- Create evaluation datasets from real user queries when possible.
- Run evaluations offline, not in the hot path of user requests.
- Compare metrics before and after RAG pipeline changes.
- Use deterministic metrics (e.g., filename matching) for quick automated checks.

## Common Mistakes

- Evaluating with too few samples (results are noisy and unreliable).
- Not including ground truth when measuring context recall.
- Running RAGAs evaluation in production request paths (adds latency and cost).
- Ignoring context precision — high recall but low precision means noisy retrieval.

## Related Project Usage

- `src/eval/retrieval_validation.py`: Deterministic retrieval evaluation using expected filenames.
- `data/eval/retrieval_eval_cases.md`: Evaluation cases with query/expected-file pairs.
- `src/kb/retrieval.py`: Retrieval pipeline being evaluated.
