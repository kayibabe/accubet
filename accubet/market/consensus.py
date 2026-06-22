"""Consensus market probabilities with the bookmaker margin (vig) removed.

Workflow per match:
1. Take the latest price each world bookmaker offers for every (market, selection, line).
2. Aggregate across books → mean "consensus" odds and best available odds.
3. Within each mutually-exclusive outcome group, strip the overround so probabilities sum
   to 1 → the **fair** market probability. This is our baseline "true probability" until
   the internal models come online (graceful degradation: market weight = 100% for now).
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.value.ev import implied_probability
from accubet.storage.models import Consensus, OddsSnapshot


# Outcome groups that form a complete, mutually-exclusive set (vig removal valid).
# over_under is handled dynamically per line. double_chance is intentionally excluded
# (its outcomes overlap, so naive normalization is wrong).
STATIC_GROUPS = {
    "match_winner": ["home", "draw", "away"],
    "btts": ["yes", "no"],
    "draw_no_bet": ["home", "away"],
}


def remove_margin(odds_by_selection: dict[str, float]) -> tuple[dict[str, float], float]:
    """Return (fair_probabilities, overround) using proportional normalization.

    fair_prob_i = implied_i / sum(implied). Overround > 1.0 is the book's margin.
    """
    implied = {sel: implied_probability(o) for sel, o in odds_by_selection.items()}
    overround = sum(implied.values())
    if overround <= 0:
        return {sel: 0.0 for sel in odds_by_selection}, overround
    fair = {sel: imp / overround for sel, imp in implied.items()}
    return fair, overround


def _latest_book_prices(session: Session, match_id: int, source: str = "apifootball"):
    """Return {(market, selection, line): {book: price}} using the latest snapshot per book."""
    rows = session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match_id, OddsSnapshot.source == source)
        .order_by(OddsSnapshot.captured_at.asc())
    ).scalars().all()

    latest: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r.market, r.selection, r.line)
        # ascending order means later rows overwrite earlier → keeps the latest price.
        latest[key][r.bookmaker] = r.price
    return latest


def build_consensus(session: Session, match_id: int, *, min_books: int = 1) -> list[Consensus]:
    """Compute, persist, and return consensus rows for one match."""
    latest = _latest_book_prices(session, match_id)
    if not latest:
        return []

    # Aggregate across books for each (market, selection, line).
    agg: dict[tuple, dict] = {}
    for (market, selection, line), book_prices in latest.items():
        prices = list(book_prices.values())
        if not prices:
            continue
        agg[(market, selection, line)] = {
            "mean_odds": fmean(prices),
            "best_odds": max(prices),
            "n_books": len(prices),
        }

    # Clear any prior consensus for this match (recompute fresh).
    for existing in session.execute(
        select(Consensus).where(Consensus.match_id == match_id)
    ).scalars().all():
        session.delete(existing)
    session.flush()

    results: list[Consensus] = []

    def _emit_group(market: str, selections: list[str], line):
        present = {
            sel: agg[(market, sel, line)]
            for sel in selections
            if (market, sel, line) in agg
        }
        if len(present) != len(selections):
            return  # incomplete group → can't remove vig reliably
        n_books = min(v["n_books"] for v in present.values())
        if n_books < min_books:
            return
        fair, overround = remove_margin({s: v["mean_odds"] for s, v in present.items()})
        for sel, p in present.items():
            row = Consensus(
                match_id=match_id,
                market=market,
                selection=sel,
                line=line,
                fair_prob=fair[sel],
                consensus_odds=(1.0 / fair[sel]) if fair[sel] > 0 else 0.0,
                best_odds=p["best_odds"],
                n_books=p["n_books"],
                overround=overround,
            )
            session.add(row)
            results.append(row)

    # Static groups.
    for market, selections in STATIC_GROUPS.items():
        _emit_group(market, selections, None)

    # Over/Under: one group per distinct line.
    ou_lines = {line for (m, _sel, line) in agg if m == "over_under" and line is not None}
    for line in ou_lines:
        _emit_group("over_under", ["over", "under"], line)

    session.flush()
    return results


def build_all_consensus(session: Session, match_ids: list[int], *, min_books: int = 1) -> int:
    total = 0
    for mid in match_ids:
        total += len(build_consensus(session, mid, min_books=min_books))
    return total
