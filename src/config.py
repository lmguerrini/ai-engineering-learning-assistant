"""Application configuration using pydantic-settings."""

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

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ai-engineering-learning-assistant"
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-engineering-learning-assistant"

    # App
    app_log_level: str = "DEBUG"
    app_default_model: str = "gpt-4o-mini"

    # Knowledge Base
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 50
    chroma_persist_dir: str = "data/chroma"
    raw_documents_dir: str = "data/raw"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
