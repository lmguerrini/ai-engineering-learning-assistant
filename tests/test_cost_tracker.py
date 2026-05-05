"""Tests for the token/cost tracking service."""

import pytest

from src.services.cost_tracker import aggregate_usage, build_usage_record, estimate_cost


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_known_model(self):
        """Cost for a known model uses its pricing."""
        cost = estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        # 1000/1000 * 0.00015 + 500/1000 * 0.0006 = 0.00015 + 0.0003 = 0.00045
        assert cost == pytest.approx(0.00045, abs=1e-6)

    def test_unknown_model_uses_default(self):
        """Unknown model falls back to default pricing."""
        cost = estimate_cost("unknown-model", prompt_tokens=1000, completion_tokens=1000)
        # 1000/1000 * 0.01 + 1000/1000 * 0.03 = 0.04
        assert cost == pytest.approx(0.04, abs=1e-6)

    def test_zero_tokens_returns_zero(self):
        """Zero tokens should return zero cost."""
        assert estimate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0) == 0.0

    def test_negative_tokens_returns_zero(self):
        """Negative tokens should return zero cost."""
        assert estimate_cost("gpt-4o", prompt_tokens=-10, completion_tokens=-5) == 0.0

    def test_partial_zero_tokens(self):
        """One token type is zero, the other is positive."""
        cost = estimate_cost("gpt-4o", prompt_tokens=0, completion_tokens=1000)
        # 0 + 1000/1000 * 0.015 = 0.015
        assert cost == pytest.approx(0.015, abs=1e-6)

    def test_gpt4_pricing(self):
        """GPT-4 pricing is applied correctly."""
        cost = estimate_cost("gpt-4", prompt_tokens=500, completion_tokens=500)
        # 500/1000 * 0.03 + 500/1000 * 0.06 = 0.015 + 0.03 = 0.045
        assert cost == pytest.approx(0.045, abs=1e-6)


# ---------------------------------------------------------------------------
# build_usage_record
# ---------------------------------------------------------------------------


class TestBuildUsageRecord:
    """Tests for build_usage_record function."""

    def test_with_valid_usage(self):
        """Builds a complete record from usage data."""
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        rec = build_usage_record("gpt-4o-mini", "learn_guide_generation", usage)

        assert rec["model"] == "gpt-4o-mini"
        assert rec["operation"] == "learn_guide_generation"
        assert rec["prompt_tokens"] == 100
        assert rec["completion_tokens"] == 50
        assert rec["total_tokens"] == 150
        assert rec["estimated_cost_usd"] > 0

    def test_with_none_usage(self):
        """None usage produces a zero-value record without crashing."""
        rec = build_usage_record("gpt-4o", "quiz_generation", None)

        assert rec["prompt_tokens"] == 0
        assert rec["completion_tokens"] == 0
        assert rec["total_tokens"] == 0
        assert rec["estimated_cost_usd"] == 0.0

    def test_with_empty_dict_usage(self):
        """Empty dict usage produces a zero-value record."""
        rec = build_usage_record("gpt-4o", "test_op", {})

        assert rec["prompt_tokens"] == 0
        assert rec["total_tokens"] == 0
        assert rec["estimated_cost_usd"] == 0.0

    def test_total_tokens_computed_if_missing(self):
        """If total_tokens is 0 but prompt+completion exist, compute it."""
        usage = {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 0}
        rec = build_usage_record("gpt-4o", "test_op", usage)

        assert rec["total_tokens"] == 300


# ---------------------------------------------------------------------------
# aggregate_usage
# ---------------------------------------------------------------------------


class TestAggregateUsage:
    """Tests for aggregate_usage function."""

    def test_single_record(self):
        """Aggregate a single record returns same values."""
        records = [
            build_usage_record("gpt-4o-mini", "learn", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
        ]
        agg = aggregate_usage(records)

        assert agg["prompt_tokens"] == 100
        assert agg["completion_tokens"] == 50
        assert agg["total_tokens"] == 150
        assert agg["estimated_cost_usd"] > 0
        assert len(agg["operations"]) == 1

    def test_multiple_records(self):
        """Aggregate multiple records sums values correctly."""
        records = [
            build_usage_record("gpt-4o-mini", "learn", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
            build_usage_record("gpt-4o-mini", "quiz", {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}),
        ]
        agg = aggregate_usage(records)

        assert agg["prompt_tokens"] == 300
        assert agg["completion_tokens"] == 150
        assert agg["total_tokens"] == 450
        assert len(agg["operations"]) == 2

    def test_empty_records(self):
        """Empty list returns zero aggregation."""
        agg = aggregate_usage([])

        assert agg["total_tokens"] == 0
        assert agg["estimated_cost_usd"] == 0.0
        assert agg["operations"] == []
