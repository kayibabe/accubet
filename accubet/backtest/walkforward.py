"""Walk-forward backtesting over the settled tracked-bet history.

Slides a rolling window of ``window_days`` across [start, end) in steps of
``step_days`` (defaults to the same width, i.e. non-overlapping windows), computes
a full set of performance metrics for each slice, and returns a list of
``WindowResult`` objects so the caller can spot trends across time.

Usage::

    from accubet.backtest.walkforward import run_walkforward, overall_metrics

    windows = run_walkforward(session, start_date, end_date, window_days=30)
    summary = overall_metrics(session, start_date, end_date)
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.backtest import metrics as m
from accubet.storage.models import TrackedBet


@dataclass
class WindowResult:
    period_start: date
    period_end: date
    n_bets: int
    staked: float
    pnl: float
    roi: float | None
    win_rate: float | None
    sharpe: float | None
    max_dd: float
    clv_mean: float | None
    clv_positive_pct: float | None
    n_with_clv: int


def months_ago(d: date, n: int) -> date:
    """Return the date ``n`` calendar months before ``d``, clamped to month boundaries."""
    mo = d.month - n
    y = d.year
    while mo <= 0:
        mo += 12
        y -= 1
    last_day = calendar.monthrange(y, mo)[1]
    return d.replace(year=y, month=mo, day=min(d.day, last_day))


def _dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def _settled_in_range(session: Session, start: datetime, end: datetime) -> list[TrackedBet]:
    return list(session.execute(
        select(TrackedBet)
        .where(TrackedBet.settled.is_(True))
        .where(TrackedBet.placed_at >= start)
        .where(TrackedBet.placed_at < end)
        .order_by(TrackedBet.placed_at.asc())
    ).scalars().all())


def _window_result(bets: list[TrackedBet], start: date, end: date) -> WindowResult:
    pnl_list = [b.pnl or 0.0 for b in bets]
    stake_list = [b.stake or 0.0 for b in bets]
    result_list = [b.result for b in bets]
    clv_list = [b.clv for b in bets]
    n_with_clv = sum(1 for b in bets if b.clv is not None)

    return WindowResult(
        period_start=start,
        period_end=end,
        n_bets=len(bets),
        staked=sum(stake_list),
        pnl=sum(pnl_list),
        roi=m.roi(pnl_list, stake_list),
        win_rate=m.win_rate(result_list),
        sharpe=m.sharpe_ratio(pnl_list, stake_list),
        max_dd=m.max_drawdown(pnl_list),
        clv_mean=m.clv_mean(clv_list),
        clv_positive_pct=m.clv_positive_pct(clv_list),
        n_with_clv=n_with_clv,
    )


def run_walkforward(
    session: Session,
    start: date,
    end: date,
    window_days: int = 30,
    step_days: int | None = None,
) -> list[WindowResult]:
    """Slide a window of *window_days* across [start, end) and return per-window metrics."""
    step = step_days if step_days is not None else window_days
    results: list[WindowResult] = []
    cursor = start
    while cursor < end:
        w_end = min(cursor + timedelta(days=window_days), end)
        bets = _settled_in_range(session, _dt(cursor), _dt(w_end))
        results.append(_window_result(bets, cursor, w_end))
        cursor += timedelta(days=step)
    return results


def overall_metrics(session: Session, start: date, end: date) -> WindowResult:
    """Single summary across the entire [start, end) range."""
    bets = _settled_in_range(session, _dt(start), _dt(end))
    return _window_result(bets, start, end)
