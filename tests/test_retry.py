"""Tests for the retry utility."""

from unittest.mock import MagicMock

import pytest

from src.services.retry import retry, with_retry


# ---------------------------------------------------------------------------
# with_retry (functional)
# ---------------------------------------------------------------------------


class TestWithRetry:
    """Tests for the with_retry helper function."""

    def test_succeeds_on_first_try(self):
        """No retries needed when the call succeeds immediately."""
        fn = MagicMock(return_value="ok")
        result = with_retry(callable=fn, max_attempts=3, base_delay=0)

        assert result == "ok"
        assert fn.call_count == 1

    def test_succeeds_after_transient_failure(self):
        """Retries and succeeds after an initial failure."""
        fn = MagicMock(side_effect=[ValueError("transient"), "ok"])
        result = with_retry(
            callable=fn,
            max_attempts=3,
            base_delay=0,
            handled_exceptions=(ValueError,),
        )

        assert result == "ok"
        assert fn.call_count == 2

    def test_fails_after_max_attempts(self):
        """Re-raises the last exception after all attempts are exhausted."""
        fn = MagicMock(side_effect=ValueError("persistent"))

        with pytest.raises(ValueError, match="persistent"):
            with_retry(
                callable=fn,
                max_attempts=3,
                base_delay=0,
                handled_exceptions=(ValueError,),
            )

        assert fn.call_count == 3

    def test_unhandled_exception_not_retried(self):
        """Exceptions not in handled_exceptions are raised immediately."""
        fn = MagicMock(side_effect=TypeError("bad type"))

        with pytest.raises(TypeError, match="bad type"):
            with_retry(
                callable=fn,
                max_attempts=3,
                base_delay=0,
                handled_exceptions=(ValueError,),
            )

        assert fn.call_count == 1

    def test_passes_args_and_kwargs(self):
        """Arguments and keyword arguments are forwarded correctly."""
        fn = MagicMock(return_value="result")
        with_retry(
            callable=fn,
            args=("a", "b"),
            kwargs={"key": "value"},
            max_attempts=1,
            base_delay=0,
        )

        fn.assert_called_once_with("a", "b", key="value")

    def test_single_attempt_raises_immediately(self):
        """With max_attempts=1, failure raises without retry."""
        fn = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError, match="fail"):
            with_retry(callable=fn, max_attempts=1, base_delay=0)

        assert fn.call_count == 1


# ---------------------------------------------------------------------------
# @retry decorator
# ---------------------------------------------------------------------------


class TestRetryDecorator:
    """Tests for the @retry decorator."""

    def test_decorator_success(self):
        """Decorated function succeeds normally."""
        @retry(max_attempts=2, base_delay=0)
        def good_fn():
            return 42

        assert good_fn() == 42

    def test_decorator_retries_and_succeeds(self):
        """Decorated function retries on failure then succeeds."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0, handled_exceptions=(ValueError,))
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient")
            return "recovered"

        assert flaky_fn() == "recovered"
        assert call_count == 2

    def test_decorator_exhausts_retries(self):
        """Decorated function raises after max attempts."""
        @retry(max_attempts=2, base_delay=0, handled_exceptions=(ValueError,))
        def bad_fn():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            bad_fn()
