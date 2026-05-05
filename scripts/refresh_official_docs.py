#!/usr/bin/env python3
"""Manual refresh pipeline for official documentation.

This script fetches summaries from official documentation URLs and
writes/updates curated Markdown files under data/official_docs/.

Usage:
    python scripts/refresh_official_docs.py          # refresh all sources
    python scripts/refresh_official_docs.py --list    # list registered sources
    python scripts/refresh_official_docs.py --dry-run # show what would be updated

Requirements:
    - Internet access (for fetching official docs)
    - This script should NOT run during normal app startup
    - Safe to run when offline — existing docs are preserved

Notes:
    - Does not delete existing files unless --clean flag is used
    - Fetches only landing/overview pages, not full doc trees
    - Content is summarized into practical engineering notes
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Official source registry — single source of truth for all official doc URLs
# ---------------------------------------------------------------------------

OFFICIAL_SOURCES: list[dict[str, str]] = [
    {
        "name": "OpenAI API & Structured Outputs",
        "filename": "openai_api_structured_outputs.md",
        "url": "https://platform.openai.com/docs/",
        "topics": "LLM API usage, structured outputs, function calling, embeddings",
    },
    {
        "name": "LangChain Core & Tools",
        "filename": "langchain_core_tools.md",
        "url": "https://docs.langchain.com/",
        "topics": "LCEL, prompt templates, output parsers, tools, document loaders",
    },
    {
        "name": "LangGraph State & Orchestration",
        "filename": "langgraph_state_orchestration.md",
        "url": "https://langchain-ai.github.io/langgraph/",
        "topics": "StateGraph, nodes, edges, reducers, conditional routing, checkpoints",
    },
    {
        "name": "LangSmith Observability",
        "filename": "langsmith_observability.md",
        "url": "https://docs.smith.langchain.com/",
        "topics": "Tracing, monitoring, evaluation, feedback",
    },
    {
        "name": "Chroma Vector Store",
        "filename": "chroma_vector_store.md",
        "url": "https://docs.trychroma.com/",
        "topics": "Collections, embeddings, similarity search, filtering",
    },
    {
        "name": "RAGAs Evaluation",
        "filename": "ragas_evaluation.md",
        "url": "https://docs.ragas.io/",
        "topics": "Faithfulness, relevancy, context precision/recall, evaluation datasets",
    },
    {
        "name": "Streamlit App Patterns",
        "filename": "streamlit_app_patterns.md",
        "url": "https://docs.streamlit.io/",
        "topics": "Session state, caching, layout, widgets, execution model",
    },
    {
        "name": "Pydantic Validation & Settings",
        "filename": "pydantic_validation_settings.md",
        "url": "https://docs.pydantic.dev/",
        "topics": "BaseModel, validators, serialization, BaseSettings, enums",
    },
    {
        "name": "Loguru Logging",
        "filename": "loguru_logging.md",
        "url": "https://loguru.readthedocs.io/",
        "topics": "Logger, sinks, formatting, exception handling, filtering",
    },
]


def get_source_registry() -> list[dict[str, str]]:
    """Return the official source registry."""
    return list(OFFICIAL_SOURCES)


def get_official_docs_dir() -> Path:
    """Return the path to the official docs directory."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "official_docs"


def list_sources() -> None:
    """Print all registered official documentation sources."""
    print(f"\nRegistered official documentation sources ({len(OFFICIAL_SOURCES)}):\n")
    for i, src in enumerate(OFFICIAL_SOURCES, 1):
        print(f"  {i}. {src['name']}")
        print(f"     URL:    {src['url']}")
        print(f"     File:   {src['filename']}")
        print(f"     Topics: {src['topics']}")
        print()


def check_existing_docs(docs_dir: Path) -> dict[str, bool]:
    """Check which official doc files already exist."""
    status = {}
    for src in OFFICIAL_SOURCES:
        filepath = docs_dir / src["filename"]
        status[src["filename"]] = filepath.exists()
    return status


def fetch_url_content(url: str) -> str | None:
    """Fetch text content from a URL.

    Returns None if the fetch fails (network error, timeout, etc.).
    """
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AI-Engineering-Learning-Assistant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  ⚠ Failed to fetch {url}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ Unexpected error fetching {url}: {e}")
        return None


def refresh_single_source(
    source: dict[str, str],
    docs_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Refresh a single official doc file.

    If dry_run is True, only reports what would happen without writing.
    Returns True if the file was updated (or would be updated in dry run).
    """
    filepath = docs_dir / source["filename"]
    exists = filepath.exists()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {source['name']}")
    print(f"  URL:  {source['url']}")
    print(f"  File: {filepath} ({'exists' if exists else 'new'})")

    if dry_run:
        action = "would update" if exists else "would create"
        print(f"  → {action} {source['filename']}")
        return True

    content = fetch_url_content(source["url"])
    if content is None:
        if exists:
            print(f"  → Keeping existing {source['filename']} (fetch failed)")
        else:
            print(f"  → Skipping {source['filename']} (fetch failed, no existing file)")
        return False

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snippet = content[:500].replace("\n", " ").strip()

    header = (
        f"# {source['name']}\n\n"
        f"- **Official source**: {source['url']}\n"
        f"- **Last refreshed**: {now}\n"
        f"- **source_type**: official_docs\n"
        f"- **Topics**: {source['topics']}\n\n"
        f"## Overview\n\n"
        f"Content fetched from official documentation.\n\n"
        f"## Source Preview\n\n"
        f"```\n{snippet}\n```\n\n"
        f"*Note: This is a preview of the official page content. "
        f"For full details, visit the official source URL above.*\n"
    )

    docs_dir.mkdir(parents=True, exist_ok=True)
    filepath.write_text(header, encoding="utf-8")
    print(f"  ✓ {'Updated' if exists else 'Created'} {source['filename']}")
    return True


def refresh_all(dry_run: bool = False, clean: bool = False) -> None:
    """Refresh all official documentation files.

    Args:
        dry_run: If True, only show what would be done.
        clean: If True, remove existing files before refresh.
    """
    docs_dir = get_official_docs_dir()

    if clean and not dry_run:
        print(f"\nCleaning existing official docs in {docs_dir}...")
        if docs_dir.exists():
            for f in docs_dir.glob("*.md"):
                f.unlink()
                print(f"  Removed: {f.name}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Refreshing official documentation...")
    print(f"Target directory: {docs_dir}")
    print(f"Sources to process: {len(OFFICIAL_SOURCES)}")

    existing = check_existing_docs(docs_dir)
    existing_count = sum(1 for v in existing.values() if v)
    print(f"Existing files: {existing_count}/{len(OFFICIAL_SOURCES)}")

    results = {"updated": 0, "failed": 0, "skipped": 0}
    for source in OFFICIAL_SOURCES:
        ok = refresh_single_source(source, docs_dir, dry_run=dry_run)
        if ok:
            results["updated"] += 1
        else:
            results["failed"] += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Refresh complete:")
    print(f"  Updated: {results['updated']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Total:   {len(OFFICIAL_SOURCES)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Refresh official documentation for the AI Engineering Learning Assistant KB.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_sources",
        help="List all registered official documentation sources.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be updated without writing files.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing official doc files before refreshing.",
    )

    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        sys.exit(0)

    refresh_all(dry_run=args.dry_run, clean=args.clean)


if __name__ == "__main__":
    main()
