"""Tests for configuration loading."""

from src.config import Settings, get_settings


def test_settings_defaults():
    """Settings should load with sensible defaults when no .env is present."""
    settings = Settings(
        _env_file=None,
        openai_api_key="",
    )
    assert settings.app_default_model == "gpt-4o-mini"
    assert settings.app_log_level == "DEBUG"
    assert settings.langchain_tracing_v2 is False


def test_get_settings_returns_settings():
    """get_settings() should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
