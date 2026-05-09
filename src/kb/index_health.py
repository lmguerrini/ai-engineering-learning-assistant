"""Knowledge-base index health and manual rebuild helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_settings
from src.kb.ingestion import run_ingestion
from src.kb.loader import SUPPORTED_EXTENSIONS
from src.kb.official_docs import ingest_official_docs
from src.kb.vector_store import (
    DEFAULT_COLLECTION,
    delete_collection,
    get_chroma_client,
    get_collection,
)

METADATA_FILENAME = "kb_index_metadata.json"


def _snapshot_directory(directory: str | Path) -> list[dict[str, Any]]:
    """Return a lightweight file snapshot for supported KB source files."""
    path = Path(directory)
    if not path.exists():
        return []

    snapshot: list[dict[str, Any]] = []
    for file_path in sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        stat = file_path.stat()
        snapshot.append(
            {
                "path": str(file_path),
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def build_source_snapshot(
    raw_documents_dir: str | Path | None = None,
    official_docs_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build a source snapshot for curated and official KB files."""
    settings = get_settings()
    raw_dir = raw_documents_dir or settings.raw_documents_dir
    official_dir = official_docs_dir or settings.official_docs_dir
    return {
        "raw": _snapshot_directory(raw_dir),
        "official": _snapshot_directory(official_dir),
    }


def _metadata_path(persist_dir: str | Path | None = None) -> Path:
    """Return the metadata file path for KB index health."""
    settings = get_settings()
    base_dir = Path(persist_dir or settings.chroma_persist_dir)
    return base_dir / METADATA_FILENAME


def load_index_metadata(persist_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Load saved KB index metadata when present."""
    path = _metadata_path(persist_dir)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load KB index metadata from {}: {}", path, exc)
        return None


def save_index_metadata(
    metadata: dict[str, Any],
    persist_dir: str | Path | None = None,
) -> Path:
    """Persist KB index metadata alongside the Chroma data."""
    path = _metadata_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("KB index metadata saved to {}", path)
    return path


def _normalize_metadatas(raw_metadatas: Any) -> list[dict[str, Any]]:
    """Flatten Chroma metadata payloads into a simple list of dicts."""
    if not raw_metadatas:
        return []
    if isinstance(raw_metadatas, list) and raw_metadatas and isinstance(raw_metadatas[0], list):
        flattened: list[dict[str, Any]] = []
        for group in raw_metadatas:
            flattened.extend(m for m in group if isinstance(m, dict))
        return flattened
    return [m for m in raw_metadatas if isinstance(m, dict)]


def _get_collection_stats(collection_name: str, persist_dir: str | Path | None = None) -> dict[str, Any]:
    """Return chunk/source counts for an existing Chroma collection."""
    settings = get_settings()
    base_dir = Path(persist_dir or settings.chroma_persist_dir)
    if not base_dir.exists():
        return {
            "name": collection_name,
            "available": False,
            "chunk_count": None,
            "source_count": None,
        }

    client = get_chroma_client(persist_dir=str(base_dir))
    collection = get_collection(name=collection_name, client=client)
    if collection is None:
        return {
            "name": collection_name,
            "available": False,
            "chunk_count": None,
            "source_count": None,
        }

    chunk_count = collection.count()
    source_count: int | None = None
    try:
        payload = collection.get(include=["metadatas"])
        metadatas = _normalize_metadatas(payload.get("metadatas"))
        filenames = {
            metadata.get("filename")
            for metadata in metadatas
            if metadata.get("filename")
        }
        source_count = len(filenames)
    except Exception as exc:
        logger.debug(
            "Could not derive source count for collection '{}': {}",
            collection_name,
            exc,
        )

    return {
        "name": collection_name,
        "available": True,
        "chunk_count": chunk_count,
        "source_count": source_count,
    }


def _resolve_collection_counts(
    stats: dict[str, Any],
    metadata_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prefer live collection counts, then fall back to saved metadata counts."""
    metadata_stats = metadata_stats or {}
    return {
        "name": stats.get("name") or metadata_stats.get("name"),
        "available": bool(stats.get("available")),
        "chunk_count": (
            stats.get("chunk_count")
            if stats.get("chunk_count") is not None
            else metadata_stats.get("chunk_count")
        ),
        "source_count": (
            stats.get("source_count")
            if stats.get("source_count") is not None
            else metadata_stats.get("source_count")
        ),
    }


def _format_embedding_model_value(
    current_model: str,
    indexed_model: str | None,
) -> str:
    """Return a reviewer-facing embedding model label."""
    if indexed_model and indexed_model != current_model:
        return f"{indexed_model} -> {current_model}"
    return current_model


def get_kb_index_health(
    *,
    raw_documents_dir: str | Path | None = None,
    official_docs_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return current KB index freshness and reviewer-facing metadata."""
    settings = get_settings()
    snapshot = build_source_snapshot(raw_documents_dir, official_docs_dir)
    metadata = load_index_metadata(persist_dir)

    curated_live = _get_collection_stats(DEFAULT_COLLECTION, persist_dir)
    official_live = _get_collection_stats(settings.official_docs_collection, persist_dir)

    curated_meta = ((metadata or {}).get("collections") or {}).get("curated")
    official_meta = ((metadata or {}).get("collections") or {}).get("official")
    curated = _resolve_collection_counts(curated_live, curated_meta)
    official = _resolve_collection_counts(official_live, official_meta)

    notes: list[str] = []
    status = "up_to_date"

    if metadata is None:
        status = "missing"
        notes.append("No saved KB index metadata found.")

    if not curated["available"] or not official["available"]:
        status = "missing"
        notes.append("One or more KB collections are missing.")

    if curated["chunk_count"] in (None, 0) or official["chunk_count"] in (None, 0):
        status = "missing"
        notes.append("One or more KB collections have no indexed chunks.")

    if metadata is not None and metadata.get("snapshot") != snapshot and status != "missing":
        status = "outdated"
        notes.append("Markdown source files changed after the last KB rebuild.")

    indexed_model = metadata.get("embedding_model") if metadata else None
    if indexed_model and indexed_model != settings.embedding_model and status != "missing":
        status = "outdated"
        notes.append("Embedding model changed after the last KB rebuild.")

    if metadata is not None and status == "up_to_date":
        expected_curated = (curated_meta or {}).get("chunk_count")
        expected_official = (official_meta or {}).get("chunk_count")
        if (
            expected_curated is not None
            and curated["chunk_count"] is not None
            and expected_curated != curated["chunk_count"]
        ) or (
            expected_official is not None
            and official["chunk_count"] is not None
            and expected_official != official["chunk_count"]
        ):
            status = "outdated"
            notes.append("Indexed chunk counts differ from the saved KB rebuild metadata.")

    status_label = {
        "up_to_date": "Up to date",
        "outdated": "Outdated",
        "missing": "Missing",
    }[status]

    return {
        "status": status,
        "status_label": status_label,
        "reindex_required": status != "up_to_date",
        "raw_docs_count": len(snapshot["raw"]),
        "official_docs_count": len(snapshot["official"]),
        "embedding_model": _format_embedding_model_value(
            settings.embedding_model,
            indexed_model,
        ),
        "current_embedding_model": settings.embedding_model,
        "indexed_embedding_model": indexed_model,
        "last_rebuild_at": metadata.get("rebuilt_at") if metadata else None,
        "collections": {
            "curated": curated,
            "official": official,
        },
        "notes": notes,
    }


def rebuild_kb_index(
    *,
    raw_documents_dir: str | Path | None = None,
    official_docs_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Manually rebuild the curated and official-docs KB collections."""
    settings = get_settings()
    raw_dir = raw_documents_dir or settings.raw_documents_dir
    official_dir = official_docs_dir or settings.official_docs_dir
    persist_dir = persist_dir or settings.chroma_persist_dir

    client = get_chroma_client(persist_dir=str(persist_dir))
    delete_collection(DEFAULT_COLLECTION, client=client)
    delete_collection(settings.official_docs_collection, client=client)

    curated_chunks = run_ingestion(
        documents_dir=raw_dir,
        collection_name=DEFAULT_COLLECTION,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        persist_dir=str(persist_dir),
    )
    official_chunks = ingest_official_docs(
        directory=official_dir,
        collection_name=settings.official_docs_collection,
        persist_dir=str(persist_dir),
    )

    snapshot = build_source_snapshot(raw_dir, official_dir)
    metadata = {
        "version": 1,
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model,
        "snapshot": snapshot,
        "collections": {
            "curated": {
                "name": DEFAULT_COLLECTION,
                "chunk_count": curated_chunks,
                "source_count": len(snapshot["raw"]),
            },
            "official": {
                "name": settings.official_docs_collection,
                "chunk_count": official_chunks,
                "source_count": len(snapshot["official"]),
            },
        },
    }
    save_index_metadata(metadata, persist_dir)

    return get_kb_index_health(
        raw_documents_dir=raw_dir,
        official_docs_dir=official_dir,
        persist_dir=persist_dir,
    )
