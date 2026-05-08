"""Application configuration using pydantic-settings."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""

    # LangSmith / LangChain tracing — accept both LANGCHAIN_* and LANGSMITH_* env vars
    langchain_tracing_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING",
            "langchain_tracing_v2", "langsmith_tracing",
        ),
    )
    langchain_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
            "langchain_api_key", "langsmith_api_key",
        ),
    )
    langchain_project: str = Field(
        default="ai-engineering-learning-assistant",
        validation_alias=AliasChoices(
            "LANGCHAIN_PROJECT", "LANGSMITH_PROJECT",
            "langchain_project", "langsmith_project",
        ),
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices(
            "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT",
            "langchain_endpoint", "langsmith_endpoint",
        ),
    )

    # App
    app_log_level: str = "DEBUG"
    app_default_model: str = "gpt-4o-mini"

    # Knowledge Base
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 50
    chroma_persist_dir: str = "data/chroma"
    raw_documents_dir: str = "data/raw"
    official_docs_dir: str = "data/official_docs"
    official_docs_collection: str = "official_docs"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
