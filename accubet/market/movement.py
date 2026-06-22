"""Odds movement analysis (opening vs current) — drift / steam detection.

Phase 1 scaffold: works off whatever snapshots we've captured so far. As ingestion runs
repeatedly pre-kickoff, more snapshots accumulate and these signals sharpen.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.storage.models import OddsSnapshot


@dataclass
class Movement:
    opening_odds: float
    current_odds: float

    @property
    def drift_pct(self) -> float:
        """(current - opening) / opening. Negative = price shortened (money came in)."""
        if self.opening_odds <= 0:
            return 0.0
        return (self.current_odds - self.opening_odds) / self.opening_odds

    @property
    def is_steam(self) -> bool:
        """Sharp shortening of >5% — money/information moving onto this selection."""
        return self.drift_pct <= -0.05


def selection_movement(
    session: Session, match_id: int, market: str, selection: str, line: float | None
) -> Movement | None:
    """Detect price movement by comparing each world-book's first vs latest snapshot.

    Groups snapshots by bookmaker so that a market with many books each captured once
    (which is typical after a single ingest run) returns None rather than a spurious
    drift reading.  Returns a Movement whose prices are the average across all books
    that have been captured at least twice.
    """
    rows = session.execute(
        select(OddsSnapshot)
        .where(
            OddsSnapshot.match_id == match_id,
            OddsSnapshot.source == "apifootball",   # world-book prices only
            OddsSnapshot.market == market,
            OddsSnapshot.selection == selection,
            OddsSnapshot.line == line,
        )
        .order_by(OddsSnapshot.captured_at.asc())
    ).scalars().all()

    by_book: dict[str, list[float]] = {}
    for r in rows:
        by_book.setdefault(r.bookmaker, []).append(r.price)

    # Only bookmakers with ≥2 captures have actual price history.
    moving = [(prices[0], prices[-1]) for prices in by_book.values() if len(prices) >= 2]
    if not moving:
        return None

    opening = sum(p[0] for p in moving) / len(moving)
    current = sum(p[1] for p in moving) / len(moving)
    return Movement(opening_odds=opening, current_odds=current)
