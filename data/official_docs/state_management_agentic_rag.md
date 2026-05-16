# State Management & Agentic RAG

- **Official source**: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/, https://python.langchain.com/docs/concepts/rag/
- **Last refreshed**: 2026-05-15
- **source_type**: official_docs
- **Versions**: `langgraph>=0.2`, `langchain>=0.2`

## When to Use

- Managing complex workflow state across multi-step agent pipelines.
- Building RAG systems that dynamically decide retrieval strategies.
- Implementing adaptive retrieval with query refinement and quality assessment.

## Key Concepts

### State Management in Agent Workflows

State management is the backbone of any multi-step agent system. In LangGraph, state flows through every node and determines routing decisions.

**Design principles**:
1. **Typed state** — use `TypedDict` with `total=False` for incremental updates.
2. **Immutable updates** — nodes return partial dicts; the framework merges them.
3. **Reducer functions** — use `Annotated` types for accumulation (e.g., trace lists).
4. **Minimal state** — only store what downstream nodes need.

```python
from typing import TypedDict, Annotated
import operator

class RAGState(TypedDict, total=False):
    query: str
    refined_query: str
    retrieved_docs: list[Document]
    source_quality_ok: bool
    attempts: int
    answer: str
    trace: Annotated[list[str], operator.add]
```

### State Patterns

**Counter-based loop control**:
```python
def should_retry(state: RAGState) -> str:
    if state.get("source_quality_ok"):
        return "generate"
    if state.get("attempts", 0) >= 3:
        return "generate"  # proceed with best available
    return "refine_and_retry"
```

**Error propagation via state**:
```python
def retrieve(state: RAGState) -> dict:
    try:
        docs = retriever.invoke(state["query"])
        return {"retrieved_docs": docs, "trace": ["retrieve: ok"]}
    except Exception as e:
        return {"retrieved_docs": [], "error": str(e), "trace": ["retrieve: failed"]}
```

**State checkpointing** for human-in-the-loop:
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
# State is persisted at each node boundary
```

### Agentic RAG

Agentic RAG goes beyond simple retrieve-then-generate by making the retrieval process itself intelligent and adaptive.

**Standard RAG pipeline**:
1. Embed query → search vector store → retrieve top-k → generate answer.

**Agentic RAG pipeline**:
1. Analyze query complexity and intent.
2. Decide retrieval strategy (vector search, keyword search, or both).
3. Retrieve and assess source quality.
4. If quality insufficient: refine query, expand search, try alternative sources.
5. Generate answer with full source attribution.
6. Self-check answer quality; regenerate if needed.

```python
def assess_source_quality(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    total_chars = sum(len(d.page_content) for d in docs)
    quality_ok = len(docs) >= 2 and total_chars >= 200
    return {
        "source_quality_ok": quality_ok,
        "trace": [f"assess: {len(docs)} docs, {total_chars} chars, ok={quality_ok}"],
    }

def refine_query(state: RAGState) -> dict:
    original = state["query"]
    refined = f"{original} overview concepts implementation examples"
    attempts = state.get("attempts", 0) + 1
    return {"refined_query": refined, "attempts": attempts, "trace": ["refine: expanded"]}
```

### Retrieval Strategies

| Strategy | When to Use | Implementation |
|---|---|---|
| **Dense retrieval** | Semantic similarity | Vector store with embeddings |
| **Sparse retrieval** | Keyword matching | BM25 or TF-IDF |
| **Hybrid** | Best of both | Combine dense + sparse with reciprocal rank fusion |
| **Multi-query** | Ambiguous queries | Generate multiple query variants, merge results |
| **Parent document** | Need full context | Retrieve chunks, return parent documents |

### Source Quality Assessment

```python
_MIN_SOURCES = 2
_MIN_CONTENT_CHARS = 200

def is_quality_sufficient(docs: list[Document]) -> bool:
    if len(docs) < _MIN_SOURCES:
        return False
    total = sum(len(d.page_content) for d in docs)
    return total >= _MIN_CONTENT_CHARS
```

Quality dimensions:
- **Coverage** — do sources address the query topic?
- **Depth** — is there enough content for a thorough answer?
- **Diversity** — are sources from different documents/sections?
- **Relevance** — do similarity scores meet a minimum threshold?

### Production Considerations

- **Caching**: Cache retrieval results and generated answers by query hash + prompt version.
- **Fallback chains**: If primary retrieval fails, fall back to broader search or static content.
- **Token budgets**: Limit context window usage; truncate or summarize long source documents.
- **Source attribution**: Always track which sources contributed to the answer.
- **Monitoring**: Log retrieval latency, source counts, quality scores, and generation metrics.
- **Versioning**: Bump cache version when prompts or retrieval logic change.

### Common Mistakes

1. **No quality gate** — generating answers from irrelevant or insufficient sources.
2. **Infinite retry loops** — not capping retrieval attempts.
3. **Ignoring source diversity** — retrieving 10 chunks from the same paragraph.
4. **Over-stuffing context** — sending too many tokens to the LLM, degrading quality.
5. **No fallback** — system fails completely when retrieval returns nothing.

### Advanced State Patterns

**Branching state for parallel retrieval**:
```python
from langgraph.graph import StateGraph, START, END

class ParallelRAGState(TypedDict, total=False):
    query: str
    dense_docs: list[Document]
    sparse_docs: list[Document]
    merged_docs: list[Document]
    answer: str
    trace: Annotated[list[str], operator.add]

def dense_retrieve(state):
    docs = vector_store.similarity_search(state["query"], k=5)
    return {"dense_docs": docs, "trace": [f"dense: {len(docs)} docs"]}

def sparse_retrieve(state):
    docs = bm25_retriever.invoke(state["query"])
    return {"sparse_docs": docs, "trace": [f"sparse: {len(docs)} docs"]}

def merge_results(state):
    all_docs = state.get("dense_docs", []) + state.get("sparse_docs", [])
    seen, unique = set(), []
    for d in all_docs:
        key = d.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return {"merged_docs": unique[:10], "trace": [f"merged: {len(unique)} unique"]}
```

**State validation middleware**:
```python
def validate_state(state: RAGState) -> dict:
    errors = []
    if not state.get("query"):
        errors.append("missing query")
    if state.get("attempts", 0) > 10:
        errors.append("excessive retry count")
    if errors:
        return {"error": "; ".join(errors), "trace": [f"validate: {errors}"]}
    return {"trace": ["validate: ok"]}
```

### Agentic RAG Decision Framework

When to upgrade from standard RAG to agentic RAG:

| Signal | Standard RAG | Agentic RAG |
|---|---|---|
| Query complexity | Single-hop factual | Multi-hop, comparative |
| Source reliability | Curated, high-quality | Mixed quality, needs filtering |
| Answer format | Simple text | Structured, multi-section |
| User expectations | Quick lookup | Comprehensive analysis |
| Error tolerance | Low (fail fast) | High (retry, refine) |

### Context Window Management

Large RAG pipelines must manage context carefully:

```python
def truncate_context(docs: list[Document], max_tokens: int = 12000) -> list[Document]:
    """Keep docs within token budget, prioritizing by relevance score."""
    result, total = [], 0
    for doc in docs:
        # Rough estimate: 1 token ≈ 4 chars
        doc_tokens = len(doc.page_content) // 4
        if total + doc_tokens > max_tokens:
            break
        result.append(doc)
        total += doc_tokens
    return result
```

### Testing State Machines

```python
def test_retry_logic():
    """Verify retry caps and state transitions."""
    state = {"query": "test", "attempts": 0, "source_quality_ok": False}
    # Simulate 3 retries
    for i in range(3):
        state["attempts"] = i + 1
        decision = should_retry(state)
        if i < 2:
            assert decision == "refine_and_retry"
        else:
            assert decision == "generate"  # cap reached
```

## Anti-Patterns

- Treating RAG as a black box without inspecting retrieved sources.
- Using a single retrieval strategy for all query types.
- Not evaluating retrieval quality separately from generation quality.
- Skipping state validation between pipeline stages.
- Storing entire documents in state instead of references or summaries.
- Not versioning prompts alongside retrieval logic changes.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

```
Redirecting...
```

### https://docs.langchain.com/oss/python/langchain/overview

```
Skip to main content Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets > Docs by LangChain home page Open source Search... ⌘ K Ask AI GitHub Try LangSmith Try LangSmith Search... Navigation LangChain overview Deep Agents LangChain LangGraph Integrations Learn Reference Contribute Python Overview Get started Install Quickstart Changelog Philosophy Core components Agents Models Messages Tools Short-term memory Event streaming Streaming Structured output Middleware Overview Prebuilt middleware Custom middleware Frontend Overview Patterns Integrations Advanced usage Guardrails Runtime Context engineering Model Context Protocol (MCP) Human-in-the-loop Multi-agent Retrieval Long-term memory Agent development LangSmith Studio Test Agent Chat UI Deploy with LangSmith Deployment Observability On this page Create an agent Core benefits LangChain overview C...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
