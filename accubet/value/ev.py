"""Core expected-value math. Tiny, pure, and unit-tested — the foundation everything else
builds on.

* implied probability = 1 / decimal_odds
* EV               = true_prob * decimal_odds - 1
* value %          = true_prob - implied_prob
"""

from __future__ import annotations


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def expected_value(true_prob: float, decimal_odds: float) -> float:
    """EV per unit staked. Positive means +EV."""
    return true_prob * decimal_odds - 1.0


def value_pct(true_prob: float, decimal_odds: float) -> float:
    """How much our probability exceeds the price's implied probability."""
    return true_prob - implied_probability(decimal_odds)


def is_value(true_prob: float, decimal_odds: float, *, min_ev: float) -> bool:
    return expected_value(true_prob, decimal_odds) > min_ev
