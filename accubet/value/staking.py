"""Staking — fractional Kelly with hard caps.

Full Kelly maximizes long-run growth but assumes you know the true probability exactly.
You don't, so we always use a *fraction* (0.25x default) and cap exposure per event.
"""

from __future__ import annotations


def kelly_fraction(true_prob: float, decimal_odds: float) -> float:
    """Full-Kelly fraction of bankroll. Clamped at 0 (never stake on -EV)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (true_prob * b - (1.0 - true_prob)) / b  # == (p*odds - 1) / (odds - 1)
    return max(0.0, f)


def fractional_kelly_stake(
    true_prob: float,
    decimal_odds: float,
    bankroll: float,
    *,
    fraction: float = 0.25,
    max_fraction_per_event: float = 0.03,
) -> float:
    """Recommended stake = fractional Kelly, capped at max exposure per event."""
    f = kelly_fraction(true_prob, decimal_odds) * fraction
    f = min(f, max_fraction_per_event)
    return round(f * bankroll, 2)
