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

    # App
    app_log_level: str = "DEBUG"
    app_default_model: str = "gpt-4o-mini"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
