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
    rows = session.execute(
        select(OddsSnapshot)
        .where(
            OddsSnapshot.match_id == match_id,
            OddsSnapshot.market == market,
            OddsSnapshot.selection == selection,
            OddsSnapshot.line == line,
        )
        .order_by(OddsSnapshot.captured_at.asc())
    ).scalars().all()
    if not rows:
        return None
    return Movement(opening_odds=rows[0].price, current_odds=rows[-1].price)
