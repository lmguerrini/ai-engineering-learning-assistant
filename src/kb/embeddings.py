"""OpenAI embeddings wrapper for the knowledge base.

Provides a clean interface for generating text embeddings with
configurable model and graceful error handling.
"""

from loguru import logger
from openai import OpenAI, OpenAIError

from src.config import get_settings


def get_embeddings(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name. Defaults to config value.

    Returns:
        List of embedding vectors.

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    settings = get_settings()
    model = model or settings.embedding_model

    if not texts:
        return []

    if not settings.openai_api_key:
        raise RuntimeError(
            "OpenAI API key is not configured. "
            "Set OPENAI_API_KEY in your .env file."
        )

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(input=texts, model=model)
        embeddings = [item.embedding for item in response.data]
        logger.debug(
            "Generated {} embeddings (model={}, dims={})",
            len(embeddings), model,
            len(embeddings[0]) if embeddings else 0,
        )
        return embeddings
    except OpenAIError as e:
        logger.error("OpenAI embedding error: {}", e)
        raise RuntimeError(f"Failed to generate embeddings: {e}") from e


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """Generate an embedding for a single text.

    Args:
        text: Text string to embed.
        model: Embedding model name.

    Returns:
        Embedding vector.
    """
    results = get_embeddings([text], model=model)
    return results[0]
