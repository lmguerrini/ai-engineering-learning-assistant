"""Simple retry utility with exponential backoff for LLM / external calls.

Usage::

    from src.services.retry import with_retry

    result = with_retry(
        callable=my_llm_call,
        args=(prompt,),
        max_attempts=3,
        handled_exceptions=(openai.APIError,),
    )

Or as a decorator::

    @retry(max_attempts=3, handled_exceptions=(openai.APIError,))
    def my_llm_call(prompt: str) -> str:
        ...
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BASE_DELAY = 1.0  # seconds
_DEFAULT_BACKOFF_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


def with_retry(
    callable: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay: float = _DEFAULT_BASE_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    handled_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """Call *callable* with retry logic.

    Retries up to *max_attempts* times on exceptions listed in
    *handled_exceptions*.  Uses exponential backoff between attempts.
    The final exception is re-raised after all attempts are exhausted.
    """
    kwargs = kwargs or {}
    last_exc: BaseException | None = None
    fn_name = getattr(callable, "__name__", repr(callable))

    for attempt in range(1, max_attempts + 1):
        try:
            return callable(*args, **kwargs)
        except handled_exceptions as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.warning(
                    "Retry {}/{} for {} after error: {} — waiting {:.1f}s",
                    attempt,
                    max_attempts,
                    fn_name,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All {} attempts exhausted for {}: {}",
                    max_attempts,
                    fn_name,
                    exc,
                )

    # Re-raise the last exception so callers can handle it.
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay: float = _DEFAULT_BASE_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    handled_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[..., Any]:
    """Decorator version of :func:`with_retry`."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return with_retry(
                callable=fn,
                args=args,
                kwargs=kwargs,
                max_attempts=max_attempts,
                base_delay=base_delay,
                backoff_factor=backoff_factor,
                handled_exceptions=handled_exceptions,
            )

        return wrapper

    return decorator
