"""Closing Line Value (CLV) — the platform's north-star edge metric.

CLV compares the price we took against the market's closing price. Beating the close
consistently (over 100+ bets) is stronger evidence of genuine edge than win rate.

Full CLV needs closing odds captured right before kickoff; that capture job lands in a
later phase. This module provides the math now so tracked bets can be scored as soon as
closing prices exist.
"""

from __future__ import annotations


def clv(taken_odds: float, closing_odds: float) -> float:
    """Fractional CLV. Positive means we beat the close.

    With decimal odds, a *higher* price we took vs a *lower* closing price is good value,
    so CLV is expressed on implied-probability terms: (p_close - p_taken) / p_taken.
    """
    if taken_odds <= 1.0 or closing_odds <= 1.0:
        return 0.0
    p_taken = 1.0 / taken_odds
    p_close = 1.0 / closing_odds
    return (p_close - p_taken) / p_taken


def beat_close(taken_odds: float, closing_odds: float) -> bool:
    return clv(taken_odds, closing_odds) > 0
