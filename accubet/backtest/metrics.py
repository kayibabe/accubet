"""Betting performance metrics — pure functions over sequences of numbers.

All functions are stateless and have no external dependencies beyond the stdlib
``statistics`` module, so they can be unit-tested without a database.
"""

from __future__ import annotations

import statistics
from typing import Sequence


def roi(pnl_list: Sequence[float], stake_list: Sequence[float]) -> float | None:
    """Return on investment: total P&L / total staked. None when nothing was staked."""
    total_staked = sum(stake_list)
    return sum(pnl_list) / total_staked if total_staked else None


def win_rate(results: Sequence[str]) -> float | None:
    """Win fraction over decided bets (excludes voids). None when no decided bets."""
    decided = [r for r in results if r in ("win", "loss")]
    return (sum(1 for r in decided if r == "win") / len(decided)) if decided else None


def sharpe_ratio(
    pnl_per_bet: Sequence[float],
    stake_per_bet: Sequence[float],
) -> float | None:
    """Per-bet Sharpe: mean(unit_return) / stdev(unit_return).

    unit_return = pnl / stake.  Requires at least 2 bets with non-zero stake.
    """
    pairs = [(p, s) for p, s in zip(pnl_per_bet, stake_per_bet) if s > 0]
    if len(pairs) < 2:
        return None
    returns = [p / s for p, s in pairs]
    sd = statistics.stdev(returns)
    return statistics.mean(returns) / sd if sd else None


def max_drawdown(pnl_series: Sequence[float]) -> float:
    """Maximum peak-to-trough drop in cumulative P&L.

    A value of 0 means the bankroll never dropped below its running peak.
    """
    cumulative = peak = max_dd = 0.0
    for p in pnl_series:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def clv_mean(clv_list: Sequence[float | None]) -> float | None:
    """Average CLV across bets that have a CLV recorded."""
    valid = [c for c in clv_list if c is not None]
    return statistics.mean(valid) if valid else None


def clv_positive_pct(clv_list: Sequence[float | None]) -> float | None:
    """Fraction of CLV-tracked bets where we beat the closing line."""
    valid = [c for c in clv_list if c is not None]
    if not valid:
        return None
    return sum(1 for c in valid if c > 0) / len(valid)
