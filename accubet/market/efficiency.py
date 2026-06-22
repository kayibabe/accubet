"""Market efficiency / inefficiency scoring.

* **Inefficiency score** — how far a given price's implied probability sits from the fair
  consensus probability, scaled by liquidity (more books agreeing = more trustworthy edge).
  This is the signal the value engine acts on; large positive score = exploitable gap.
* **Market confidence (0-100)** — how much to trust the consensus itself, from the number
  of agreeing books and the tightness of the overround.
"""

from __future__ import annotations

from accubet.value.ev import implied_probability


def inefficiency_score(fair_prob: float, price: float, n_books: int) -> float:
    """Signed edge in probability terms, liquidity-weighted.

    Positive => the price is too generous vs fair (value for the bettor).
    """
    edge = fair_prob - implied_probability(price)
    liquidity_factor = min(1.0, n_books / 5.0)  # saturates at 5 books
    return edge * liquidity_factor


def market_confidence(n_books: int, overround: float | None) -> float:
    """0-100 score: more books and a tighter margin => higher confidence in the fair price."""
    book_component = min(1.0, n_books / 6.0)  # 6+ books = full marks
    if overround is None or overround <= 1.0:
        margin_component = 1.0
    else:
        # 1.02 overround (2% margin) ~ excellent; 1.12 (12%) ~ poor.
        margin_component = max(0.0, 1.0 - (overround - 1.0) / 0.12)
    return round(100.0 * (0.6 * book_component + 0.4 * margin_component), 1)
