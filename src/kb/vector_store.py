"""Chroma vector store for the knowledge base.

Provides persistent vector storage with functions for creating collections,
adding documents, and performing similarity search.
"""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from loguru import logger

from src.config import get_settings
from src.kb.embeddings import get_embeddings
from src.kb.loader import Document

DEFAULT_COLLECTION = "knowledge_base"


def get_chroma_client(persist_dir: str | None = None) -> chromadb.ClientAPI:
    """Create a persistent Chroma client.

    Args:
        persist_dir: Directory for Chroma persistence.
            Defaults to config value.

    Returns:
        Chroma client instance.
    """
    if persist_dir is None:
        persist_dir = get_settings().chroma_persist_dir

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    logger.debug("Chroma client initialized (persist_dir={})", persist_dir)
    return client


def create_collection(
    name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
) -> Collection:
    """Get or create a Chroma collection.

    Args:
        name: Collection name.
        client: Chroma client. Created if not provided.

    Returns:
        Chroma collection.
    """
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Collection '{}' ready ({} documents)", name, collection.count())
    return collection


def get_collection(
    name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
) -> Collection | None:
    """Return an existing Chroma collection without creating it."""
    if client is None:
        client = get_chroma_client()

    try:
        collection = client.get_collection(name=name)
    except Exception as exc:
        logger.debug("Collection '{}' not available: {}", name, exc)
        return None

    logger.debug("Collection '{}' loaded ({} documents)", name, collection.count())
    return collection


def delete_collection(
    name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
) -> bool:
    """Delete a Chroma collection when it exists."""
    if client is None:
        client = get_chroma_client()

    collection = get_collection(name=name, client=client)
    if collection is None:
        return False

    client.delete_collection(name=name)
    logger.info("Deleted collection '{}'", name)
    return True


def add_documents(
    documents: list[Document],
    collection: Collection | None = None,
    batch_size: int = 50,
) -> int:
    """Add documents with embeddings to a Chroma collection.

    Args:
        documents: List of Document objects to add.
        collection: Chroma collection. Created if not provided.
        batch_size: Number of documents to embed per API call.

    Returns:
        Number of documents added.
    """
    if collection is None:
        collection = create_collection()

    if not documents:
        logger.warning("No documents to add")
        return 0

    total_added = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [doc.content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [
            f"{doc.metadata.get('filename', 'doc')}_{doc.metadata.get('chunk_index', j)}"
            for j, doc in enumerate(batch, start=i)
        ]

        embeddings = get_embeddings(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_added += len(batch)
        logger.debug("Added batch of {} documents (total: {})", len(batch), total_added)

    logger.info("Added {} documents to collection", total_added)
    return total_added


def similarity_search(
    query: str,
    top_k: int = 5,
    collection: Collection | None = None,
) -> list[Document]:
    """Search for similar documents using a query.

    Args:
        query: Search query text.
        top_k: Number of results to return.
        collection: Chroma collection. Created if not provided.

    Returns:
        List of matching documents sorted by relevance.
    """
    if collection is None:
        collection = create_collection()

    if collection.count() == 0:
        logger.warning("Collection is empty, no results to return")
        return []

    query_embedding = get_embeddings([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents: list[Document] = []
    if results["documents"] and results["documents"][0]:
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            metadata_with_score = {**metadata, "distance": distance}
            documents.append(
                Document(content=doc_text, metadata=metadata_with_score)
            )

    logger.debug("Similarity search returned {} results for query", len(documents))
    return documents
