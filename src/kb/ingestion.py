"""End-to-end ingestion pipeline for the knowledge base.

Orchestrates: load documents → chunk → embed → store in Chroma.
"""

from pathlib import Path

from loguru import logger

from src.kb.chunker import chunk_documents
from src.kb.loader import load_documents
from src.kb.vector_store import add_documents, create_collection, get_chroma_client


def run_ingestion(
    documents_dir: str | Path | None = None,
    collection_name: str = "knowledge_base",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    persist_dir: str | None = None,
) -> int:
    """Run the full ingestion pipeline.

    Loads documents from disk, chunks them, generates embeddings,
    and stores everything in a Chroma collection.

    Args:
        documents_dir: Path to raw documents. Defaults to config value.
        collection_name: Name of the Chroma collection.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.
        persist_dir: Chroma persistence directory.

    Returns:
        Number of chunks stored.
    """
    logger.info("Starting ingestion pipeline")

    # Step 1: Load documents
    logger.info("Step 1/3: Loading documents")
    documents = load_documents(directory=documents_dir)
    if not documents:
        logger.warning("No documents found. Ingestion aborted.")
        return 0
    logger.info("Loaded {} documents", len(documents))

    # Step 2: Chunk documents
    logger.info("Step 2/3: Chunking documents")
    chunks = chunk_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not chunks:
        logger.warning("No chunks produced. Ingestion aborted.")
        return 0
    logger.info("Produced {} chunks", len(chunks))

    # Step 3: Embed and store
    logger.info("Step 3/3: Embedding and storing chunks")
    client = get_chroma_client(persist_dir=persist_dir)
    collection = create_collection(name=collection_name, client=client)
    added = add_documents(chunks, collection=collection)

    logger.info(
        "Ingestion complete: {} documents → {} chunks → {} stored",
        len(documents), len(chunks), added,
    )
    return added
