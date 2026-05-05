"""Lightweight observability utilities (LangSmith configuration, status reporting)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from loguru import logger

from src.config import get_settings


@dataclass
class TracingStatus:
    """Summary of the current LangSmith / tracing configuration."""

    enabled: bool = False
    project: str = ""
    endpoint: str = ""
    has_api_key: bool = False
    issues: list[str] = field(default_factory=list)


def configure_langsmith_tracing() -> TracingStatus:
    """Set LangSmith environment variables from app settings and return status.

    Safe to call even when LangSmith is not configured — returns a disabled
    status with no side effects.
    """
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load settings for tracing: {}", exc)
        return TracingStatus(issues=[f"Settings unavailable: {exc}"])

    project = settings.langchain_project or "ai-engineering-learning-assistant"
    endpoint = settings.langchain_endpoint or "https://api.smith.langchain.com"

    status = TracingStatus(project=project, endpoint=endpoint)

    if not settings.langchain_tracing_v2:
        status.issues.append("LANGCHAIN_TRACING_V2 is not enabled.")
        logger.debug("LangSmith tracing is disabled.")
        return status

    if not settings.langchain_api_key:
        status.issues.append("LANGCHAIN_API_KEY is missing.")
        logger.warning("LangSmith tracing enabled but API key is missing.")
        return status

    # Propagate to environment so LangChain/LangGraph picks them up
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

    status.enabled = True
    status.has_api_key = True
    logger.info("LangSmith tracing enabled for project '{}' at {}.", project, endpoint)
    return status


def get_tracing_status() -> TracingStatus:
    """Return current tracing status without modifying environment."""
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        return TracingStatus(issues=[f"Settings unavailable: {exc}"])

    enabled = settings.langchain_tracing_v2 and bool(settings.langchain_api_key)
    return TracingStatus(
        enabled=enabled,
        project=settings.langchain_project or "ai-engineering-learning-assistant",
        endpoint=settings.langchain_endpoint or "https://api.smith.langchain.com",
        has_api_key=bool(settings.langchain_api_key),
        issues=_collect_issues(settings),
    )


def _collect_issues(settings) -> list[str]:
    """Collect configuration issues for the tracing status report."""
    issues: list[str] = []
    if not settings.langchain_tracing_v2:
        issues.append("LANGCHAIN_TRACING_V2 is not enabled.")
    if not settings.langchain_api_key:
        issues.append("LANGCHAIN_API_KEY is missing.")
    return issues


def format_tracing_status(status: TracingStatus) -> dict:
    """Return a UI-safe dictionary describing the tracing status."""
    return {
        "tracing_enabled": status.enabled,
        "project": status.project,
        "endpoint": status.endpoint,
        "has_api_key": status.has_api_key,
        "issues": status.issues,
        "status_label": "✅ Active" if status.enabled else "❌ Disabled",
    }
