# AI Engineering Learning Assistant

A guided educational AI agent that helps AI Engineering students study through a structured **Learn → Quiz → Feedback → Memory** workflow, powered by LangGraph, Agentic RAG, and OpenAI.

---

## Problem Definition

AI Engineering students face a vast, fragmented landscape of tools (LangChain, LangGraph, RAG, vector stores, evaluation frameworks) and need a structured, adaptive way to learn them. Traditional study resources are static and do not adapt to individual progress or knowledge gaps.

This assistant solves this by providing:
- **Personalized study guides** generated from a curated knowledge base using Agentic RAG
- **Adaptive quizzes** that focus on weak areas identified through learning history
- **Persistent memory** that tracks progress and adjusts difficulty over time
- **Human-in-the-loop** approval before saving learning results

## Target User

AI Engineering bootcamp students working through sprints covering LLMs, agents, RAG, LangGraph, evaluation, and production deployment.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                 Streamlit UI                     │
│  Intro │ Learn │ Quiz │ Progress │ Advanced      │
└────────┬──────────────┬─────────────────────────┘
         │              │
    ┌────▼────┐    ┌────▼────┐
    │  Learn  │    │  Quiz   │     LangGraph
    │  Graph  │    │  Graph  │     Workflows
    └────┬────┘    └────┬────┘
         │              │
    ┌────▼──────────────▼────┐
    │   KB Retrieval Layer   │
    │  Curated + Official    │
    │   Docs Fallback        │
    └────────┬───────────────┘
             │
    ┌────────▼───────────────┐
    │  Chroma Vector Store   │
    └────────────────────────┘

    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Memory     │  │   Feedback   │  │    Cache     │
    │  (SQLite)    │  │  (SQLite)    │  │  (SQLite)    │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### Key Components

| Layer | Description |
|-------|-------------|
| **Streamlit UI** | 5-section app: Intro, Learn, Quiz, Progress, Advanced/Debug |
| **LangGraph Workflows** | Explicit state graphs for Learn and Quiz flows |
| **KB Retrieval** | Curated KB (data/raw) + official docs fallback (data/official_docs) |
| **Memory** | SQLite-based learning history with HITL approval |
| **Services** | Cost tracking, retry, caching, feedback, observability |

---

## LangGraph Learn Flow

The Learn workflow is an explicit LangGraph StateGraph with 9 nodes:

1. **validate_input** — checks topic is non-empty
2. **load_user_memory** — loads memory profile for personalization
3. **retrieve_sources** — retrieves from curated KB via Chroma
4. **assess_source_quality** — evaluates whether retrieved sources are sufficient
5. **refine_query_if_needed** — if sources are insufficient, refines the query once
6. **generate_study_guide** — calls OpenAI to produce a structured study guide
7. **quality_check** — validates the generated guide
8. **persist_learning_event_placeholder** — prepares data for memory (no DB write)
9. **return_output** — packages the final result

**Agentic RAG behavior:** After initial retrieval, the graph assesses source quality. If insufficient, it refines the query and retrieves again (up to 2 attempts), then generates from the best available sources. Official docs fallback enriches weak curated KB results.

## LangGraph Quiz Flow

Two separate LangGraph StateGraphs:

**Generation graph:** `load_topic_context → load_user_memory → generate_quiz → validate_quiz → return_results`

**Evaluation graph:** `evaluate_answers → extract_weak_areas → create_memory_candidate → return_results`

- Scoring is **fully deterministic** (direct string comparison, no LLM)
- Quiz validation enforces: question count, option count (≥3), correct answer in options, explanations present
- Weak areas are extracted from incorrectly answered questions
- Memory candidate is created but **not saved** until user approves (HITL)

## Agentic RAG

This is not a simple retrieve-and-generate pipeline. The Learn workflow implements Agentic RAG:

1. **Retrieve** from curated KB
2. **Assess** source quality (sufficient/insufficient)
3. **Refine** the query if sources are insufficient
4. **Re-retrieve** with refined query
5. **Fallback** to official docs if curated KB is still weak
6. **Generate** from the best combined source set

This loop is controlled by an `attempts` counter (max 2) to prevent infinite retrieval cycles.

## Curated KB + Official Docs Fallback

- **Curated KB** (`data/raw/`): 13 Markdown files covering AI Agents, RAG, LangChain, LangGraph, prompt engineering, evaluation, memory, HITL, and production patterns. This is the primary retrieval source.
- **Official Docs** (`data/official_docs/`): 9 curated reference summaries from official documentation (OpenAI, LangChain, LangGraph, LangSmith, Chroma, RAGAs, Streamlit, Pydantic, Loguru). Used as fallback enrichment only when curated KB context is insufficient.
- **Separation**: Official docs are stored in a separate Chroma collection with `source_type="official_docs"` metadata. Domain-aware filtering prioritizes relevant official docs (e.g., LangGraph queries → LangGraph docs).

## Memory + HITL

- **SQLite memory** stores learning sessions: topic, score, weak areas, timestamp
- **HITL pattern**: After quiz evaluation, the user sees a summary and chooses "Save" or "Skip" — memory is never written automatically
- **Memory personalization**: Recent topics, recurring weak areas, and average score influence study guide generation and quiz difficulty
- **Graph purity**: No DB writes inside graph nodes — writes happen only in the UI/service layer

## Caching + Feedback Loop

- **Cache** (SQLite): Study guides and quizzes are cached by topic + difficulty + style + memory profile hash. Cache entries have TTL and expire automatically.
- **Feedback**: Users can rate (1–5) and comment on study guides and quiz results. Deterministic rules derive suggestions from feedback (e.g., low ratings → suggest simpler explanations, "too easy" comments → increase difficulty).
- **Personalization effect**: Feedback summary is loaded during memory injection and influences prompt generation.

## Token / Cost Tracking

- Tracks prompt_tokens, completion_tokens, total_tokens, and estimated_cost_usd per LLM operation
- OpenAI pricing constants for common models (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- Usage records are accumulated per session and displayed in Advanced/Debug
- Zero-value fallback when token usage is unavailable

## LangSmith Observability

- **Optional**: Tracing is disabled by default. Enable by setting `LANGCHAIN_TRACING_V2=true` and providing `LANGCHAIN_API_KEY`.
- **Configuration**: `LANGCHAIN_PROJECT`, `LANGCHAIN_ENDPOINT` supported
- **Safe no-op**: App runs normally without LangSmith credentials
- **Visibility**: Advanced/Debug page shows tracing status, project name, endpoint, and API key warning if tracing is enabled but key is missing
- **Graph logging**: Structured loguru logging at graph run start/end/error

## Retrieval Validation + RAG Evaluation

- **Eval cases**: `data/eval/retrieval_eval_cases.md` defines test queries with expected source filenames
- **Retrieval validation** (`src/eval/retrieval_validation.py`): Parses eval cases, runs retrieval, reports hit rate and per-case pass/fail
- **RAG evaluation** (`src/eval/rag_evaluation.py`): Extends retrieval validation with source coverage metrics and structured reporting
- **CLI script**: `scripts/run_rag_eval.py` runs offline evaluation without requiring API keys
- **RAGAs note**: RAGAs is not currently installed. The evaluation framework includes notes on how to enable RAGAs advanced metrics (faithfulness, answer relevancy, context precision) by installing the `ragas` package.

---

## How to Run the App

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` with five sections: Intro/Help, Learn, Quiz, Progress/Feedback, and Advanced/Debug.

## How to Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_learn_graph.py -v

# Run tests with coverage (if pytest-cov is installed)
python -m pytest tests/ --cov=src -v
```

Tests do not require OpenAI API keys — all LLM calls are mocked.

## How to Run Retrieval / RAG Evaluation

```bash
# Dry-run evaluation (no API key needed, uses mock retrieval)
python scripts/run_rag_eval.py

# Evaluation against real vector store (requires ingested KB + API key)
python scripts/run_rag_eval.py --real --top-k 10
```

## How to Refresh Official Docs

```bash
# List registered official doc sources
python scripts/refresh_official_docs.py --list

# Dry-run (show what would be refreshed without downloading)
python scripts/refresh_official_docs.py --dry-run

# Refresh all official docs
python scripts/refresh_official_docs.py
```

The refresh script fetches from official documentation URLs and updates files under `data/official_docs/`. It does not run during normal app startup.

---

## Optional Tasks Implemented

| Task | Status | Notes |
|------|--------|-------|
| Multi-agent orchestration (LangGraph Learn + Quiz) | ✅ | Two explicit StateGraph workflows |
| Agentic RAG with source assessment + refinement | ✅ | Quality loop with max 2 attempts |
| Knowledge base from official docs | ✅ | 9 curated official doc summaries |
| RAG evaluation | ✅ | Deterministic offline eval + RAGAs-ready |
| Long-term memory (SQLite) | ✅ | Learning history with weak area tracking |
| Human-in-the-loop | ✅ | Save/skip approval for quiz results |
| Caching | ✅ | SQLite cache with TTL for guides/quizzes |
| Token/cost tracking | ✅ | Per-operation tracking with session summary |
| LangSmith observability | ✅ | Optional tracing with safe fallback |
| Feedback loop | ✅ | Rating + comments → personalization rules |
| Memory-based personalization | ✅ | Weak areas, score, topics influence generation |

## Limitations

- **No live web scraping**: Official docs are refreshed manually via script, not during app runtime.
- **Single-user**: No authentication or multi-user session management.
- **LLM dependency**: Study guide and quiz generation require a valid OpenAI API key. Fallback behavior provides basic placeholder content.
- **RAGAs not installed**: Advanced evaluation metrics require installing the `ragas` package separately.
- **No streaming**: LLM responses are generated in full before display.

## Future Improvements

- Enable RAGAs-based evaluation metrics (faithfulness, context precision, answer relevancy)
- Add streaming LLM responses for better UX
- Implement spaced repetition scheduling based on memory data
- Add multi-user support with session isolation
- Expand curated KB with additional AI Engineering topics
- Add automated KB refresh scheduling
- Implement LangSmith dashboard integration for production monitoring

---

## Tech Stack

- **Python** · **Streamlit** · **LangGraph** · **LangChain** · **OpenAI**
- **ChromaDB** · **SQLite** · **Pydantic** · **pydantic-settings**
- **Loguru** · **Pytest**

## Project Structure

```
app.py                          # Streamlit entrypoint
src/
  config.py                     # pydantic-settings configuration
  logging_config.py             # Loguru logging setup
  schemas.py                    # Core Pydantic schemas
  ui/
    pages.py                    # Streamlit page renderers
    display_helpers.py          # Reusable UI formatting helpers
  graphs/
    learn_state.py              # Learn workflow typed state
    learn_nodes.py              # Learn workflow node functions
    learn_graph.py              # Learn StateGraph definition
    quiz_state.py               # Quiz workflow typed state
    quiz_nodes.py               # Quiz workflow node functions
    quiz_graph.py               # Quiz StateGraph definitions
  kb/
    loader.py                   # Document loading from data/raw
    chunker.py                  # Text splitting with overlap
    embeddings.py               # OpenAI embeddings wrapper
    vector_store.py             # Chroma vector store operations
    ingestion.py                # End-to-end ingestion pipeline
    retrieval.py                # Document retrieval
    official_docs.py            # Official docs loading/retrieval/fallback
  memory/
    db.py                       # SQLite database initialization
    memory_service.py           # Learning memory CRUD
    feedback_service.py         # Feedback storage and summary
  services/
    cost_tracker.py             # Token/cost tracking
    retry.py                    # Retry utility with backoff
    cache.py                    # SQLite caching service
    observability.py            # LangSmith tracing configuration
  eval/
    retrieval_validation.py     # Retrieval eval case validation
    rag_evaluation.py           # Offline RAG evaluation
  demo/
    review_examples.py          # Curated demo topics
  tools/                        # Agent tools (reserved)
scripts/
  refresh_official_docs.py      # Official docs refresh CLI
  run_rag_eval.py               # RAG evaluation CLI
tests/                          # Pytest test suite
data/
  raw/                          # Curated KB documents
  official_docs/                # Official documentation summaries
  eval/                         # Retrieval evaluation cases
  meta/                         # KB metadata (index)
  chroma/                       # Chroma vector store (gitignored)
  memory/                       # SQLite databases (gitignored)
```

---

*This project is part of an AI Engineering bootcamp sprint.*
