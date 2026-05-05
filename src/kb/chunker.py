"""Text chunking for the knowledge base.

Splits documents into smaller chunks with configurable size and overlap
for embedding and vector storage.
"""

from loguru import logger

from src.config import get_settings
from src.kb.loader import Document


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Number of overlapping characters between chunks.
            Defaults to config value.

    Returns:
        List of text chunks.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    logger.debug(
        "Split text ({} chars) into {} chunks (size={}, overlap={})",
        len(text), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split a document into smaller chunk documents.

    Each chunk inherits the original document's metadata plus a chunk_index.

    Args:
        document: The document to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunk documents with metadata.
    """
    text_chunks = chunk_text(document.content, chunk_size, chunk_overlap)

    chunk_docs: list[Document] = []
    for i, chunk in enumerate(text_chunks):
        metadata = {**document.metadata, "chunk_index": i}
        chunk_docs.append(Document(content=chunk, metadata=metadata))

    return chunk_docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split multiple documents into chunks.

    Args:
        documents: List of documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of all chunk documents.
    """
    all_chunks: list[Document] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))

    logger.info(
        "Chunked {} documents into {} chunks",
        len(documents), len(all_chunks),
    )
    return all_chunks
