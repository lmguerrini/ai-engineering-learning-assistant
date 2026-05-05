"""Official documentation KB loader and retrieval.

Provides separate loading, ingestion, and retrieval for official docs,
keeping them isolated from the curated Turing/course KB in data/raw/.
"""

from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.kb.chunker import chunk_documents
from src.kb.loader import Document, load_documents
from src.kb.vector_store import (
    add_documents,
    create_collection,
    get_chroma_client,
    similarity_search,
)

_SOURCE_TYPE = "official_docs"
_DOC_TYPE = "official_reference"

# Domain registry: maps filename prefix keywords to canonical domain names.
DOMAIN_REGISTRY: dict[str, str] = {
    "openai": "openai",
    "langchain": "langchain",
    "langgraph": "langgraph",
    "langsmith": "langsmith",
    "chroma": "chroma",
    "ragas": "ragas",
    "streamlit": "streamlit",
    "pydantic": "pydantic",
    "loguru": "loguru",
}

# Query keyword → domain mapping for domain-aware filtering.
_QUERY_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "openai": ["openai", "gpt", "structured output", "chat completion", "api key"],
    "langchain": ["langchain", "chain", "tools", "lcel", "runnable"],
    "langgraph": ["langgraph", "stategraph", "state graph", "node", "edge", "reducer", "conditional routing", "checkpoint"],
    "langsmith": ["langsmith", "tracing", "observability", "trace"],
    "chroma": ["chroma", "chromadb", "vector store", "vector database", "embedding store"],
    "ragas": ["ragas", "evaluation", "rag eval", "faithfulness", "answer relevancy"],
    "streamlit": ["streamlit", "st.", "sidebar", "ui component"],
    "pydantic": ["pydantic", "basemodel", "base model", "schema", "validation", "settings"],
    "loguru": ["loguru", "logging", "logger"],
}


def infer_domain(filename: str) -> str:
    """Infer domain from an official docs filename.

    Matches the first DOMAIN_REGISTRY key found in the lowered filename.
    Returns 'general' if no match.
    """
    lower = filename.lower()
    for key, domain in DOMAIN_REGISTRY.items():
        if key in lower:
            return domain
    return "general"


def detect_query_domains(query: str) -> list[str]:
    """Detect relevant domains from a query string using keyword heuristics.

    Returns a list of matched domain names, ordered by match priority.
    """
    lower = query.lower()
    matched: list[str] = []
    for domain, keywords in _QUERY_DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                matched.append(domain)
                break
    return matched


def load_official_docs(directory: str | Path | None = None) -> list[Document]:
    """Load official docs and tag them with enriched metadata.

    Each document receives:
        source_type = 'official_docs'
        doc_type = 'official_reference'
        domain = inferred from filename (e.g. 'langgraph', 'openai')

    Args:
        directory: Path to official docs directory.
            Defaults to the configured official_docs_dir.

    Returns:
        List of documents with enriched metadata.
    """
    if directory is None:
        directory = Path(get_settings().official_docs_dir)
    else:
        directory = Path(directory)

    docs = load_documents(directory)
    for doc in docs:
        doc.metadata["source_type"] = _SOURCE_TYPE
        doc.metadata["doc_type"] = _DOC_TYPE
        doc.metadata["domain"] = infer_domain(doc.metadata.get("filename", ""))
    logger.info("Loaded {} official docs from {}", len(docs), directory)
    return docs


def ingest_official_docs(
    directory: str | Path | None = None,
    collection_name: str | None = None,
    persist_dir: str | None = None,
) -> int:
    """Ingest official docs: load → chunk → embed → store.

    Args:
        directory: Path to official docs directory.
        collection_name: Chroma collection name for official docs.
        persist_dir: Chroma persistence directory.

    Returns:
        Number of chunks stored.
    """
    settings = get_settings()
    if collection_name is None:
        collection_name = settings.official_docs_collection
    if persist_dir is None:
        persist_dir = settings.chroma_persist_dir

    docs = load_official_docs(directory)
    if not docs:
        logger.warning("No official docs to ingest")
        return 0

    chunks = chunk_documents(docs)
    for chunk in chunks:
        chunk.metadata["source_type"] = _SOURCE_TYPE
        chunk.metadata["doc_type"] = _DOC_TYPE
        chunk.metadata.setdefault("domain", infer_domain(
            chunk.metadata.get("filename", "")
        ))

    client = get_chroma_client(persist_dir=persist_dir)
    collection = create_collection(name=collection_name, client=client)
    count = add_documents(chunks, collection=collection)
    logger.info(
        "Ingested {} official doc chunks into collection '{}'",
        count, collection_name,
    )
    return count


def retrieve_official_docs(
    query: str,
    top_k: int = 5,
    collection_name: str | None = None,
    persist_dir: str | None = None,
) -> list[Document]:
    """Retrieve relevant chunks from the official docs collection.

    Args:
        query: Search query text.
        top_k: Number of results to return.
        collection_name: Chroma collection name for official docs.
        persist_dir: Chroma persistence directory.

    Returns:
        List of relevant document chunks with metadata.
        Returns empty list on error or if collection is empty.
    """
    settings = get_settings()
    if collection_name is None:
        collection_name = settings.official_docs_collection
    if persist_dir is None:
        persist_dir = settings.chroma_persist_dir

    if not query.strip():
        logger.warning("Empty query for official docs retrieval")
        return []

    try:
        client = get_chroma_client(persist_dir=persist_dir)
        collection = create_collection(name=collection_name, client=client)
        # Fetch extra results when domain filtering is active, then rerank.
        query_domains = detect_query_domains(query)
        fetch_k = top_k * 3 if query_domains else top_k
        results = similarity_search(query=query, top_k=fetch_k, collection=collection)
        for doc in results:
            doc.metadata.setdefault("source_type", _SOURCE_TYPE)
            doc.metadata.setdefault("doc_type", _DOC_TYPE)
            doc.metadata.setdefault("domain", infer_domain(
                doc.metadata.get("filename", "")
            ))

        if query_domains:
            results = _rerank_by_domain(results, query_domains, top_k)
        else:
            results = results[:top_k]

        logger.info(
            "Retrieved {} official doc chunks for query: '{}' (domains={})",
            len(results), query[:80], query_domains or "any",
        )
        return results
    except Exception as e:
        logger.error("Official docs retrieval failed: {}", e)
        return []


def _rerank_by_domain(
    docs: list[Document],
    preferred_domains: list[str],
    top_k: int,
) -> list[Document]:
    """Re-rank documents so those matching preferred domains come first.

    Domain-matching docs are placed first (preserving original order),
    followed by non-matching docs.  Result is trimmed to top_k.
    """
    matched: list[Document] = []
    others: list[Document] = []
    for doc in docs:
        domain = doc.metadata.get("domain", "")
        if domain in preferred_domains:
            matched.append(doc)
        else:
            others.append(doc)
    return (matched + others)[:top_k]


def retrieve_with_fallback(
    query: str,
    curated_docs: list[Document],
    min_sources: int = 2,
    min_content_chars: int = 200,
    top_k: int = 4,
    collection_name: str | None = None,
    persist_dir: str | None = None,
) -> list[Document]:
    """Retrieve from official docs as fallback when curated KB is weak.

    Assesses curated docs quality first. If insufficient, queries
    official docs and merges results.

    Args:
        query: Search query text.
        curated_docs: Already-retrieved curated KB documents.
        min_sources: Minimum curated docs to consider sufficient.
        min_content_chars: Minimum total chars to consider sufficient.
        top_k: Number of official doc results to fetch on fallback.
        collection_name: Chroma collection name for official docs.
        persist_dir: Chroma persistence directory.

    Returns:
        Combined list of curated + official docs (curated first).
    """
    # Tag curated docs with source_type if not already set
    for doc in curated_docs:
        doc.metadata.setdefault("source_type", "curated_kb")

    total_chars = sum(len(d.content) for d in curated_docs)
    curated_sufficient = (
        len(curated_docs) >= min_sources and total_chars >= min_content_chars
    )

    if curated_sufficient:
        logger.debug(
            "Curated KB sufficient ({} docs, {} chars) — skipping official docs fallback",
            len(curated_docs), total_chars,
        )
        return curated_docs

    logger.info(
        "Curated KB insufficient ({} docs, {} chars) — querying official docs",
        len(curated_docs), total_chars,
    )
    official = retrieve_official_docs(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    combined = list(curated_docs) + official
    logger.info(
        "Combined {} curated + {} official docs",
        len(curated_docs), len(official),
    )
    return combined
