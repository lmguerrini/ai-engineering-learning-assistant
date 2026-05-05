"""Document retrieval from the knowledge base.

Provides a clean interface for retrieving relevant chunks
from the vector store given a query.
"""

from loguru import logger

from src.kb.loader import Document
from src.kb.vector_store import create_collection, get_chroma_client, similarity_search


def retrieve_documents(
    query: str,
    top_k: int = 5,
    collection_name: str = "knowledge_base",
    persist_dir: str | None = None,
) -> list[Document]:
    """Retrieve relevant documents for a query.

    Args:
        query: Search query text.
        top_k: Number of results to return.
        collection_name: Name of the Chroma collection.
        persist_dir: Chroma persistence directory.

    Returns:
        List of relevant document chunks with metadata.
        Returns empty list if no results or collection is empty.
    """
    if not query.strip():
        logger.warning("Empty query provided, returning no results")
        return []

    try:
        client = get_chroma_client(persist_dir=persist_dir)
        collection = create_collection(name=collection_name, client=client)

        results = similarity_search(
            query=query,
            top_k=top_k,
            collection=collection,
        )

        logger.info(
            "Retrieved {} results for query: '{}'",
            len(results), query[:80],
        )
        return results

    except Exception as e:
        logger.error("Retrieval failed for query '{}': {}", query[:80], e)
        return []
