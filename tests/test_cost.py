from app.observability.cost import (
    count_tokens,
    estimate_cost_usd,
    usage_for,
)


def test_count_tokens_nonempty():
    assert count_tokens("hello world", "gpt-4o") > 0


def test_count_tokens_empty_is_zero():
    assert count_tokens("", "gpt-4o") == 0


def test_count_tokens_unknown_model_still_counts():
    # falls back to a generic encoding rather than crashing
    assert count_tokens("hello world", "some-unknown-model") > 0


def test_estimate_cost_gpt4o():
    # 1000 in @ $2.50/1M + 500 out @ $10.00/1M = 0.0025 + 0.0050
    assert round(estimate_cost_usd(1000, 500, "gpt-4o"), 6) == 0.0075


def test_price_prefix_matches_versioned_model_id():
    # versioned ids like gpt-4o-2024-08-06 should match the gpt-4o price
    assert estimate_cost_usd(1_000_000, 0, "gpt-4o-2024-08-06") == 2.50


def test_unknown_model_is_zero_cost():
    assert estimate_cost_usd(1000, 1000, "mystery-model") == 0.0


def test_usage_for_shape_and_values():
    u = usage_for("some prompt text here", "a generated answer", "gpt-4o")
    assert set(u) == {"input_tokens", "output_tokens", "cost_usd", "model"}
    assert u["input_tokens"] > 0
    assert u["output_tokens"] > 0
    assert u["cost_usd"] >= 0
    assert u["model"] == "gpt-4o"
