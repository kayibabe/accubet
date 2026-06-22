"""Tests for the Phase 5 backtest metrics and walk-forward engine."""

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from accubet.backtest import metrics as m
from accubet.backtest.walkforward import WindowResult, months_ago, overall_metrics, run_walkforward
from accubet.storage.models import Base, TrackedBet


# ---------------------------------------------------------------------------
# metrics.py — pure function tests
# ---------------------------------------------------------------------------

def test_roi_basic():
    assert m.roi([15.0, -10.0], [10.0, 10.0]) == pytest.approx(0.25)


def test_roi_no_stake():
    assert m.roi([], []) is None


def test_win_rate_mixed():
    assert m.win_rate(["win", "loss", "win", "void"]) == pytest.approx(2 / 3)


def test_win_rate_voids_only():
    assert m.win_rate(["void", "void"]) is None


def test_win_rate_empty():
    assert m.win_rate([]) is None


def test_sharpe_positive_edge():
    pnl = [10.0, 5.0, 8.0, 12.0, 6.0]
    stakes = [10.0] * 5
    s = m.sharpe_ratio(pnl, stakes)
    assert s is not None and s > 0


def test_sharpe_too_few_bets():
    assert m.sharpe_ratio([5.0], [10.0]) is None


def test_max_drawdown_flat_wins():
    # never drops below peak — drawdown is 0
    assert m.max_drawdown([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_max_drawdown_all_losses():
    # cumulative goes -10, -20, -30; peak stays 0; dd grows each step
    assert m.max_drawdown([-10.0, -10.0, -10.0]) == pytest.approx(30.0)


def test_max_drawdown_recovery():
    # +10 -> peak 10, then -20 -> dd=20, then +15 -> cum=5, dd=5 → max 20
    assert m.max_drawdown([10.0, -20.0, 15.0]) == pytest.approx(20.0)


def test_max_drawdown_empty():
    assert m.max_drawdown([]) == pytest.approx(0.0)


def test_clv_mean_ignores_none():
    assert m.clv_mean([0.05, None, 0.10, None]) == pytest.approx(0.075)


def test_clv_mean_all_none():
    assert m.clv_mean([None, None]) is None


def test_clv_positive_pct():
    assert m.clv_positive_pct([0.05, -0.02, 0.10, None]) == pytest.approx(2 / 3)


def test_clv_positive_pct_all_none():
    assert m.clv_positive_pct([None]) is None


# ---------------------------------------------------------------------------
# months_ago helper
# ---------------------------------------------------------------------------

def test_months_ago_basic():
    assert months_ago(date(2025, 6, 15), 3) == date(2025, 3, 15)


def test_months_ago_crosses_year():
    assert months_ago(date(2025, 2, 28), 3) == date(2024, 11, 28)


def test_months_ago_clamps_to_month_end():
    # March 31 minus 1 month = Feb 28 (not Feb 31)
    assert months_ago(date(2025, 3, 31), 1) == date(2025, 2, 28)


# ---------------------------------------------------------------------------
# walkforward / overall_metrics — integration tests with in-memory SQLite
# ---------------------------------------------------------------------------

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _bet(session, placed: datetime, result: str, pnl: float, stake: float = 10.0,
         clv: float | None = None):
    tb = TrackedBet(
        kind="single", market="match_winner", selection="home",
        odds=2.0, stake=stake, settled=True, result=result,
        pnl=pnl, placed_at=placed, clv=clv,
    )
    session.add(tb)
    session.flush()
    return tb


def test_overall_metrics_empty(session):
    result = overall_metrics(session, date(2025, 1, 1), date(2025, 6, 1))
    assert result.n_bets == 0
    assert result.roi is None
    assert result.max_dd == pytest.approx(0.0)


def test_overall_metrics_win_and_loss(session):
    _bet(session, datetime(2025, 3, 1), "win", 15.0, clv=0.05)
    _bet(session, datetime(2025, 3, 15), "loss", -10.0, clv=-0.02)

    result = overall_metrics(session, date(2025, 1, 1), date(2025, 6, 1))
    assert result.n_bets == 2
    assert result.pnl == pytest.approx(5.0)
    assert result.staked == pytest.approx(20.0)
    assert result.roi == pytest.approx(0.25)
    assert result.win_rate == pytest.approx(0.5)
    assert result.n_with_clv == 2
    assert result.clv_mean == pytest.approx(0.015)


def test_overall_metrics_excludes_unsettled(session):
    _bet(session, datetime(2025, 3, 1), "win", 15.0)
    pending = TrackedBet(kind="single", market="match_winner", selection="home",
                         odds=2.0, stake=10.0, settled=False, result="pending", pnl=None,
                         placed_at=datetime(2025, 3, 5))
    session.add(pending)
    session.flush()

    result = overall_metrics(session, date(2025, 1, 1), date(2025, 6, 1))
    assert result.n_bets == 1  # pending excluded


def test_walkforward_windows(session):
    # place two bets in Jan and two in Feb; window=31 days, so two windows
    _bet(session, datetime(2025, 1, 5), "win", 10.0)
    _bet(session, datetime(2025, 1, 20), "loss", -10.0)
    _bet(session, datetime(2025, 2, 5), "win", 15.0)
    _bet(session, datetime(2025, 2, 20), "win", 8.0)

    windows = run_walkforward(session, date(2025, 1, 1), date(2025, 3, 5), window_days=31)
    assert len(windows) >= 2

    jan = windows[0]
    assert jan.n_bets == 2
    assert jan.pnl == pytest.approx(0.0)

    feb = windows[1]
    assert feb.n_bets == 2
    assert feb.pnl == pytest.approx(23.0)


def test_walkforward_out_of_range_bets_ignored(session):
    _bet(session, datetime(2024, 12, 31), "win", 10.0)   # before range
    _bet(session, datetime(2025, 6, 1), "win", 10.0)     # at end boundary (excluded)
    _bet(session, datetime(2025, 3, 1), "loss", -5.0)    # inside range

    result = overall_metrics(session, date(2025, 1, 1), date(2025, 6, 1))
    assert result.n_bets == 1
