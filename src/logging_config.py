"""Loguru logging configuration."""

import sys

from loguru import logger

from src.config import get_settings


def setup_logging() -> None:
    """Configure loguru logger based on application settings."""
    settings = get_settings()

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.app_log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logger.info("Logging configured (level={})", settings.app_log_level.upper())
