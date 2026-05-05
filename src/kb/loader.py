"""Document loader for the knowledge base.

Loads .md and .txt files from the raw documents directory and returns
structured document objects with content and metadata.
"""

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings

SUPPORTED_EXTENSIONS = {".md", ".txt"}


class Document(BaseModel):
    """A loaded document with content and metadata."""

    content: str
    metadata: dict = Field(default_factory=dict)


def _infer_topic(filename: str) -> str:
    """Infer a topic name from the filename.

    Converts filenames like 'ai_agents.md' → 'AI Agents'.
    """
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()


def load_document(file_path: Path) -> Document:
    """Load a single document from a file path.

    Args:
        file_path: Path to a .md or .txt file.

    Returns:
        Document with content and metadata.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        logger.warning("Empty document: {}", file_path.name)

    metadata = {
        "filename": file_path.name,
        "source": str(file_path),
        "topic": _infer_topic(file_path.name),
    }

    logger.debug("Loaded document: {} ({} chars)", file_path.name, len(content))
    return Document(content=content, metadata=metadata)


def load_documents(directory: str | Path | None = None) -> list[Document]:
    """Load all supported documents from a directory.

    Args:
        directory: Path to the documents directory.
            Defaults to the configured raw_documents_dir.

    Returns:
        List of loaded documents.
    """
    if directory is None:
        directory = Path(get_settings().raw_documents_dir)
    else:
        directory = Path(directory)

    if not directory.exists():
        logger.warning("Documents directory not found: {}", directory)
        return []

    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported documents found in {}", directory)
        return []

    documents: list[Document] = []
    for file_path in files:
        try:
            doc = load_document(file_path)
            documents.append(doc)
        except Exception as e:
            logger.error("Failed to load {}: {}", file_path.name, e)

    logger.info("Loaded {} documents from {}", len(documents), directory)
    return documents
