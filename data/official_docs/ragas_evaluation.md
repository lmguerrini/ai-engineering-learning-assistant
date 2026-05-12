# RAGAs Evaluation

- **Official source**: https://docs.ragas.io/
- **Last refreshed**: 2026-05-12
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

**Metric details**:

- **Faithfulness**: Decomposes the answer into claims and checks each against the context. Score = (supported claims) / (total claims). A score of 0.0 means the answer is entirely hallucinated.
- **Answer Relevancy**: Generates hypothetical questions from the answer and measures cosine similarity to the original question. Low scores indicate the answer is off-topic or too generic.
- **Context Precision**: Measures whether the most relevant context items are ranked higher. Uses ground truth to determine relevance. Important when context window is limited.
- **Context Recall**: Checks if all claims in the ground truth can be attributed to the retrieved contexts. Low recall means the retriever missed important information.

> **Caveat**: Faithfulness and Answer Relevancy use an LLM as judge — results may vary across models and runs. Use a consistent judge model for comparable evaluations.

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

- `ground_truth` is required for `context_recall` and `context_precision` but optional for `faithfulness` and `answer_relevancy`.
- Minimum 20–50 samples recommended for statistically meaningful evaluation.
- Questions should cover diverse topics, difficulty levels, and edge cases.
- Include adversarial examples (unanswerable questions, ambiguous queries) to test robustness.

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
- Each evaluation sample costs LLM tokens for judging — budget for evaluation API costs.

**Configuring the judge LLM**:

```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm,  # explicit judge model
)
```

- Use `temperature=0` for the judge LLM to maximize evaluation consistency.
- `gpt-4o-mini` is cost-effective for judging; `gpt-4o` may be more accurate for complex assessments.
- The judge LLM processes each sample independently — no cross-sample contamination.

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
- Custom metrics are useful for domain-specific evaluation (e.g., filename matching, keyword presence).

## Advanced Patterns

### Interpreting Evaluation Results

```python
df = result.to_pandas()

# Find worst-performing samples
low_faith = df[df["faithfulness"] < 0.5]
print(f"{len(low_faith)} samples with low faithfulness (hallucination risk)")

# Correlation analysis: do retrieval issues cause answer issues?
import numpy as np
corr = np.corrcoef(df["context_recall"], df["faithfulness"])[0, 1]
print(f"Correlation between context_recall and faithfulness: {corr:.2f}")
```

**Diagnostic patterns**:

| Metric Pattern | Diagnosis | Recommended Fix |
|---------------|-----------|----------------|
| Low faithfulness, high context recall | LLM ignoring context (hallucinating despite good retrieval) | Improve prompt to emphasize grounding; lower temperature |
| High faithfulness, low answer relevancy | Answer is grounded but doesn't address the question | Improve prompt to focus on the question; refine query |
| Low context recall, low faithfulness | Retriever missing relevant documents | Improve retrieval (better embeddings, more KB content, query refinement) |
| Low context precision, high context recall | Too many irrelevant documents retrieved | Reduce `n_results`; add distance threshold filtering |

### Evaluation Pipeline Automation

```python
import json
from datetime import datetime

def run_eval_pipeline(chain, dataset, output_path: str):
    """Run evaluation and save results with metadata."""
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "aggregate": {k: float(v) for k, v in result.items()},
        "sample_count": len(dataset),
        "per_sample": result.to_pandas().to_dict(orient="records"),
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report
```

- Save evaluation reports with timestamps for trend tracking.
- Integrate into CI/CD: fail builds if metrics drop below thresholds.
- Track metrics over time to detect regressions.

### Evaluation Without Ground Truth

When ground truth is unavailable, use reference-free metrics:

- **Faithfulness**: Does the answer stick to what’s in the retrieved context? (no ground truth needed)
- **Answer Relevancy**: Does the answer address the original question? (no ground truth needed)
- **Aspect Critique**: LLM-judged aspects like harmfulness, coherence, conciseness.

These metrics enable continuous monitoring of production outputs without manual labeling.

## Practical Implementation Notes

- Start with `faithfulness` and `answer_relevancy` as baseline metrics — they don’t require ground truth.
- Create evaluation datasets from real user queries when possible.
- Run evaluations offline, not in the hot path of user requests.
- Compare metrics before and after RAG pipeline changes to measure impact.
- Use deterministic metrics (e.g., filename matching) for quick automated CI checks.
- Budget for evaluation cost: each sample requires 1–3 LLM calls for judging.
- Version your evaluation datasets — changing the dataset invalidates historical comparisons.
- Aim for 80%+ faithfulness and 70%+ answer relevancy as production baselines.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ImportError: ragas` | RAGAs not installed | `pip install ragas`; note: requires `datasets` and `langchain` |
| All scores are 0.0 | Empty contexts or answers in dataset | Verify dataset fields are non-empty strings/lists |
| Inconsistent scores across runs | Non-zero temperature on judge LLM | Set `temperature=0` on the judge model |
| `context_recall` errors | Missing `ground_truth` field in dataset | Add `ground_truth` column; or remove `context_recall` from metrics |
| Very slow evaluation | Large dataset + expensive judge model | Use `gpt-4o-mini` for judging; reduce dataset size for iteration |
| `KeyError` on dataset columns | Column names don’t match expected schema | RAGAs expects `question`, `answer`, `contexts`, `ground_truth` |

## Common Mistakes

- Evaluating with too few samples — results are noisy and unreliable.
- Not including `ground_truth` when measuring `context_recall`.
- Running RAGAs evaluation in production request paths (adds latency and cost).
- Ignoring `context_precision` — high recall but low precision means noisy retrieval.
- Using only aggregate scores without inspecting per-sample failures.
- Changing the evaluation dataset between comparisons — invalidates before/after analysis.
- Not pinning the judge model version — model updates change evaluation baselines.
- Treating RAGAs scores as absolute quality measures — they are relative and best used for comparison.

## Related Project Usage

- `src/eval/retrieval_validation.py`: Deterministic retrieval evaluation using expected filenames.
- `src/eval/rag_evaluation.py`: Source coverage metrics and RAGAs-readiness evaluation.
- `data/eval/retrieval_eval_cases.md`: Evaluation cases with query/expected-file pairs.
- `scripts/run_rag_eval.py`: CLI runner for evaluation pipeline.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://docs.ragas.io/en/stable/

```
Skip to content Ragas Office Hours - If you need help setting up Evals for your AI application, sign up for our Office Hours here. Ragas Initializing search vibrantlabsai/ragas 🚀 Get Started 📚 Core Concepts 🛠️ How-to Guides 📖 References ❤️ Community Ragas vibrantlabsai/ragas Table of contents Why Ragas? Key Features Want help improving your AI application using evals? 🚀 Get Started 🚀 Get Started Installation Quick Start Tutorials Tutorials Evaluate a prompt Evaluate a simple RAG system Evaluate an AI Workflow Evaluate an AI Agent 📚 Core Concepts 📚 Core Concepts Experimentation Datasets Metrics Metrics Overview Available Metrics Available Metrics Retrieval Augmented Generation Retrieval Augmented Generation Context Precision Context Recall Context Entities Recall Noise Sensitivity Response Relevancy Faithfulness Nvidia Metrics Nvidia Metrics Answer Accuracy Context Relevance Response G...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
