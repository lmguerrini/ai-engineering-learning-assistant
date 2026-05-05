"""Lightweight token/cost tracking for OpenAI LLM operations.

Provides cost estimation based on simple pricing constants,
usage record building, and aggregation utilities.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Pricing constants (USD per 1K tokens)
# Updated for common OpenAI models — extend as needed.
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}

# Fallback pricing when model is not found in the table.
_DEFAULT_PRICING = {"prompt": 0.01, "completion": 0.03}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the USD cost for a single LLM call.

    Returns 0.0 if token counts are zero or negative.
    """
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0.0

    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    cost = (
        max(prompt_tokens, 0) / 1000 * pricing["prompt"]
        + max(completion_tokens, 0) / 1000 * pricing["completion"]
    )
    return round(cost, 6)


def build_usage_record(
    model: str,
    operation: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a structured usage record from raw OpenAI usage data.

    If *usage* is ``None`` or empty, returns a zero-value record
    without crashing.
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

    estimated_cost = estimate_cost(model, prompt_tokens, completion_tokens)

    record: dict[str, Any] = {
        "model": model,
        "operation": operation,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }

    logger.debug(
        "Usage record: op={} model={} tokens={} cost=${:.6f}",
        operation,
        model,
        total_tokens,
        estimated_cost,
    )
    return record


def aggregate_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple usage records into a summary.

    Returns a dict with totals for tokens and cost, plus per-operation
    breakdown.
    """
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_cost = 0.0
    operations: list[dict[str, Any]] = []

    for rec in records:
        total_prompt += rec.get("prompt_tokens", 0)
        total_completion += rec.get("completion_tokens", 0)
        total_tokens += rec.get("total_tokens", 0)
        total_cost += rec.get("estimated_cost_usd", 0.0)
        operations.append({
            "operation": rec.get("operation", "unknown"),
            "total_tokens": rec.get("total_tokens", 0),
            "estimated_cost_usd": rec.get("estimated_cost_usd", 0.0),
        })

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "operations": operations,
    }
