"""USD price table per 1M tokens. Override via PRICING_JSON env if Azure rates differ."""
import json
import os

_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def price_table() -> dict[str, dict[str, float]]:
    override = os.getenv("PRICING_JSON")
    if not override:
        return _DEFAULT_PRICES
    return {**_DEFAULT_PRICES, **json.loads(override)}


def resolve_model_key(model: str) -> str | None:
    """Azure reports versioned names (gpt-4o-mini-2024-07-18); match the longest known prefix."""
    matches = [k for k in price_table() if model.startswith(k)]
    return max(matches, key=len) if matches else None


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = resolve_model_key(model)
    if key is None:
        return 0.0
    rates = price_table()[key]
    cost = prompt_tokens * rates["input"] + completion_tokens * rates["output"]
    return round(cost / 1_000_000, 8)
