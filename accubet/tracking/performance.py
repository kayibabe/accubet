"""Performance reporting off the auto-tracked paper bets.

Answers the two questions the tracker exists for: *which markets are working* and *how is
the system doing overall* — by win rate, ROI/yield, and P&L, judged only on settled bets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.storage.models import TrackedBet


@dataclass
class PerfRow:
    label: str
    total: int = 0
    pending: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    staked: float = 0.0   # settled stake only
    pnl: float = 0.0

    @property
    def settled(self) -> int:
        return self.wins + self.losses + self.voids

    @property
    def decided(self) -> int:
        return self.wins + self.losses  # excludes voids

    @property
    def win_rate(self) -> float | None:
        return (self.wins / self.decided) if self.decided else None

    @property
    def roi(self) -> float | None:
        return (self.pnl / self.staked) if self.staked else None


def _accumulate(row: PerfRow, tb: TrackedBet) -> None:
    row.total += 1
    if not tb.settled:
        row.pending += 1
        return
    if tb.result == "win":
        row.wins += 1
    elif tb.result == "loss":
        row.losses += 1
    else:
        row.voids += 1
    row.staked += tb.stake or 0.0
    row.pnl += tb.pnl or 0.0


def report(session: Session) -> dict:
    bets = list(session.execute(select(TrackedBet)).scalars())

    overall = PerfRow("OVERALL")
    by_market: dict[str, PerfRow] = defaultdict(lambda: PerfRow(""))
    by_kind: dict[str, PerfRow] = defaultdict(lambda: PerfRow(""))

    for tb in bets:
        _accumulate(overall, tb)
        m = by_market[tb.market]
        m.label = tb.market
        _accumulate(m, tb)
        k = by_kind[tb.kind]
        k.label = tb.kind
        _accumulate(k, tb)

    return {
        "overall": overall,
        "by_market": sorted(by_market.values(), key=lambda r: r.total, reverse=True),
        "by_kind": sorted(by_kind.values(), key=lambda r: r.total, reverse=True),
    }
