# RAGAs Evaluation

- **Official source**: https://docs.ragas.io/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs
- **Versions**: `ragas>=0.1`

## When to Use

- Evaluating the quality of RAG pipeline outputs.
- Measuring retrieval relevance, answer faithfulness, and context quality.
- Benchmarking RAG improvements across iterations.

## Key Concepts

### Core Metrics

| Metric | Measures | Requires Ground Truth |
|--------|----------|-----------------------|
| **Faithfulness** | Whether the answer is grounded in retrieved context | No |
| **Answer Relevancy** | How well the answer addresses the question | No |
| **Context Precision** | Whether relevant items are ranked higher in retrieved contexts | Yes |
| **Context Recall** | Whether all relevant information was retrieved | Yes |

Scores range from `0.0` to `1.0` (higher is better).

### Evaluation Dataset

Each evaluation sample requires:

```python
from datasets import Dataset

eval_data = {
    "question": [
        "What is retrieval-augmented generation?",
        "How do agents use tools?",
    ],
    "answer": [
        "RAG combines document retrieval with LLM generation...",
        "Agents invoke tools via function calling...",
    ],
    "contexts": [
        ["RAG is a technique that retrieves relevant documents..."],
        ["AI agents use tool-calling to interact with external systems..."],
    ],
    "ground_truth": [  # required for context_recall
        "RAG retrieves relevant documents and uses them as context for generation.",
        "Agents use function calling to invoke external tools and APIs.",
    ],
}

dataset = Dataset.from_dict(eval_data)
```

- `ground_truth` is required for `context_recall` but optional for other metrics.
- Minimum 20–50 samples recommended for statistically meaningful evaluation.

### Running Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)  # aggregate scores
df = result.to_pandas()  # per-sample breakdown
```

- Evaluation requires an LLM for judging (uses OpenAI by default).
- Run evaluations offline — not in the hot path of user requests.

### Custom Metrics

```python
from ragas.metrics.base import Metric

class FilenameMatchMetric(Metric):
    name = "filename_match"

    def score(self, row):
        expected = set(row["expected_files"])
        retrieved = set(row["retrieved_files"])
        return len(expected & retrieved) / len(expected) if expected else 1.0
```

- Extend `Metric` base class with a `score` method.
- Supports LLM-based or deterministic scoring.

## Practical Implementation Notes

- Start with `faithfulness` and `answer_relevancy` as baseline metrics.
- Create evaluation datasets from real user queries when possible.
- Run evaluations offline, not in the hot path of user requests.
- Compare metrics before and after RAG pipeline changes to measure impact.
- Use deterministic metrics (e.g., filename matching) for quick automated CI checks.

## Common Mistakes

- Evaluating with too few samples — results are noisy and unreliable.
- Not including `ground_truth` when measuring `context_recall`.
- Running RAGAs evaluation in production request paths (adds latency and cost).
- Ignoring `context_precision` — high recall but low precision means noisy retrieval.
- Using only aggregate scores without inspecting per-sample failures.

## Related Project Usage

- `src/eval/retrieval_validation.py`: Deterministic retrieval evaluation using expected filenames.
- `src/eval/rag_evaluation.py`: Source coverage metrics and RAGAs-readiness evaluation.
- `data/eval/retrieval_eval_cases.md`: Evaluation cases with query/expected-file pairs.
- `scripts/run_rag_eval.py`: CLI runner for evaluation pipeline.
