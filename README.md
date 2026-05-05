# AI Engineering Learning Assistant

A guided educational AI agent that helps AI Engineering students study through a structured **Learn → Quiz → Feedback → Memory** workflow.

## Status

🚧 **Phase 1 — Foundation** — Project structure, placeholder UI, config, logging, and schemas are in place. LLM and retrieval features are coming in later phases.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill environment variables
cp .env.example .env

# 4. Run the app
streamlit run app.py
```

## Project Structure

```
app.py              # Streamlit entrypoint
src/
  config.py         # pydantic-settings configuration
  logging_config.py # Loguru logging setup
  schemas.py        # Core Pydantic schemas
  ui/               # Streamlit UI pages
  graphs/           # LangGraph workflows (future)
  kb/               # Knowledge base / retrieval (future)
  memory/           # Long-term memory (future)
  services/         # Shared services (future)
  tools/            # Agent tools (future)
tests/              # Pytest test suite
data/
  raw/              # Curated learning documents (future)
  chroma/           # Chroma vector store (gitignored)
  memory/           # SQLite memory DB (gitignored)
```

## Tech Stack

- Python · Streamlit · LangGraph · LangChain · OpenAI · Chroma · SQLite
- Pydantic · pydantic-settings · Loguru · Pytest

## License

This project is part of an AI Engineering bootcamp sprint.
