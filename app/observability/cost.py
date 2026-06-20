"""Approximate token + cost accounting for a query.

Production RAG is a cost-sensitive workload, so every query reports how many
tokens it consumed and an estimated USD cost. Token counts use ``tiktoken``;
prices are a small, explicit table (USD per 1M tokens) that is easy to keep
current. Unknown models count tokens but report $0 rather than guessing.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# USD per 1,000,000 tokens: (input, output). Keep this list short and current.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}

_encoders: dict[str, object] = {}


def _get_encoder(model: str):
    if model in _encoders:
        return _encoders[model]
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    _encoders[model] = enc
    return enc


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Token count for ``text`` under ``model``'s encoding.

    Falls back to a generic encoding for unknown models, and to a rough
    chars/4 estimate if tiktoken is unavailable for any reason.
    """
    if not text:
        return 0
    try:
        return len(_get_encoder(model).encode(text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("token_count_fallback: %s", exc)
        return max(1, len(text) // 4)


def _price_for(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    # prefix match so versioned ids (gpt-4o-2024-08-06) resolve to gpt-4o
    for key, price in PRICES.items():
        if model and model.startswith(key):
            return price
    return (0.0, 0.0)


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    inp, out = _price_for(model)
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


def usage_for(prompt_text: str, answer_text: str, model: str) -> dict:
    """Token + cost summary for one generation call."""
    input_tokens = count_tokens(prompt_text, model)
    output_tokens = count_tokens(answer_text, model)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(estimate_cost_usd(input_tokens, output_tokens, model), 6),
        "model": model,
    }
