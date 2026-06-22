"""Tests for grading, accumulator building, settlement, and performance."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from accubet.config import get_config
from accubet.storage.models import Base, Match, OddsSnapshot, Result, Team, TrackedBet
from accubet.tracking.performance import report
from accubet.tracking.tracked_bets import log_singles, settle, settle_selection
from accubet.value.accumulator import build_accumulators


def opp(mid, market, selection, price, prob, line=None, source="betway", passes=True):
    return SimpleNamespace(
        match_id=mid, home=f"Home{mid}", away=f"Away{mid}", market=market,
        selection=selection, line=line, price=price, fair_prob=prob,
        price_source=source, ev=prob * price - 1, confidence=0.8, _passes=passes,
    )


# --- grading ---------------------------------------------------------------

@pytest.mark.parametrize("market,selection,line,hg,ag,expected", [
    ("match_winner", "home", None, 2, 1, "win"),
    ("match_winner", "away", None, 2, 1, "loss"),
    ("match_winner", "draw", None, 1, 1, "win"),
    ("btts", "yes", None, 1, 1, "win"),
    ("btts", "yes", None, 1, 0, "loss"),
    ("btts", "no", None, 1, 0, "win"),
    ("over_under", "over", 2.5, 2, 1, "win"),
    ("over_under", "under", 2.5, 1, 1, "win"),
    ("over_under", "over", 2.0, 1, 1, "void"),     # push
    ("double_chance", "1x", None, 1, 1, "win"),
    ("double_chance", "x2", None, 2, 1, "loss"),
])
def test_settle_selection(market, selection, line, hg, ag, expected):
    assert settle_selection(market, selection, line, hg, ag) == expected


# --- accumulators ----------------------------------------------------------

def test_value_tier_lands_in_band():
    cfg = get_config()
    opps = [opp(1, "match_winner", "home", 2.0, 0.6), opp(2, "btts", "yes", 2.3, 0.5)]
    tickets = build_accumulators(opps, cfg)
    v = tickets["value"]
    assert v is not None
    assert 3.0 <= v.combined_odds <= 5.0
    assert len({leg.match_id for leg in v.legs}) == len(v.legs)  # no two legs same match


def test_banker_tier_high_strike_low_odds():
    cfg = get_config()
    opps = [opp(3, "match_winner", "home", 1.30, 0.80, passes=False),
            opp(4, "match_winner", "home", 1.35, 0.78, passes=False)]
    tickets = build_accumulators(opps, cfg)
    b = tickets["banker"]
    assert b is not None
    assert 1.3 <= b.combined_odds <= 1.8
    assert b.combined_prob > 0.5


# --- settlement + reporting ------------------------------------------------

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_settle_and_report(session):
    home, away = Team(api_team_id=1, name="A"), Team(api_team_id=2, name="B")
    session.add_all([home, away])
    session.flush()
    m = Match(api_fixture_id=9001, home_team_id=home.id, away_team_id=away.id, status="FT")
    session.add(m)
    session.flush()
    session.add(Result(match_id=m.id, home_goals=1, away_goals=1))
    # one winning BTTS-yes single (1-1 => yes), one losing match_winner-home single
    session.add(TrackedBet(kind="single", match_id=m.id, market="btts", selection="yes",
                           odds=2.50, stake=10.0))
    session.add(TrackedBet(kind="single", match_id=m.id, market="match_winner", selection="home",
                           odds=1.80, stake=10.0))
    session.flush()

    out = settle(session, get_config())
    assert out["settled"] == 2

    rep = report(session)
    o = rep["overall"]
    assert o.wins == 1 and o.losses == 1
    assert o.staked == pytest.approx(20.0)
    # win pnl = 10*(2.5-1)=15 ; loss = -10 ; net +5 ; roi = 5/20 = 0.25
    assert o.pnl == pytest.approx(5.0)
    assert o.roi == pytest.approx(0.25)


def test_settle_populates_clv(session):
    from datetime import datetime
    home, away = Team(api_team_id=20, name="C"), Team(api_team_id=21, name="D")
    session.add_all([home, away])
    session.flush()
    kickoff = datetime(2025, 4, 10, 15, 0)
    m = Match(api_fixture_id=9002, home_team_id=home.id, away_team_id=away.id,
              status="FT", kickoff=kickoff)
    session.add(m)
    session.flush()
    session.add(Result(match_id=m.id, home_goals=2, away_goals=0))
    # closing world-book price for home win, captured 15 min before kickoff
    session.add(OddsSnapshot(
        match_id=m.id, source="apifootball", bookmaker="bet365",
        market="match_winner", selection="home", price=1.90,
        captured_at=datetime(2025, 4, 10, 14, 45),
    ))
    # we tracked a bet at 2.10 (better than the 1.90 close → positive CLV)
    session.add(TrackedBet(
        kind="single", match_id=m.id, market="match_winner", selection="home",
        odds=2.10, stake=10.0, settled=False, result="pending",
    ))
    session.flush()

    out = settle(session, get_config())
    assert out["settled"] == 1

    tb = session.execute(select(TrackedBet)).scalar_one()
    assert tb.closing_odds == pytest.approx(1.90)
    # p_close=1/1.9≈0.5263, p_taken=1/2.1≈0.4762  →  CLV=(0.5263-0.4762)/0.4762 > 0
    assert tb.clv is not None and tb.clv > 0


def test_settle_no_snapshot_leaves_clv_null(session):
    from datetime import datetime
    home, away = Team(api_team_id=30, name="E"), Team(api_team_id=31, name="F")
    session.add_all([home, away])
    session.flush()
    m = Match(api_fixture_id=9003, home_team_id=home.id, away_team_id=away.id, status="FT")
    session.add(m)
    session.flush()
    session.add(Result(match_id=m.id, home_goals=1, away_goals=0))
    session.add(TrackedBet(
        kind="single", match_id=m.id, market="match_winner", selection="home",
        odds=2.00, stake=10.0, settled=False, result="pending",
    ))
    session.flush()

    settle(session, get_config())

    tb = session.execute(select(TrackedBet)).scalar_one()
    assert tb.settled is True
    assert tb.clv is None   # no snapshot → CLV stays null, not an error


def test_log_singles_dedupes(session):
    cfg = get_config()
    opps = [opp(1, "btts", "yes", 2.65, 0.47)]
    assert log_singles(session, cfg, opps) == 1
    assert log_singles(session, cfg, opps) == 0  # same selection not logged twice
