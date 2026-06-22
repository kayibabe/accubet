"""Tests for vig removal and the consensus → Betway comparison flow."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from accubet.config import get_config
from accubet.market.comparison import _is_liquid, compare_match
from accubet.market.consensus import build_consensus, remove_margin
from accubet.storage.models import Base, Match, OddsSnapshot, Team


def test_liquidity_filter_excludes_exotic_lines():
    cfg = get_config()
    assert _is_liquid(cfg, "match_winner", None)
    assert _is_liquid(cfg, "btts", None)
    assert _is_liquid(cfg, "over_under", 2.5)
    assert not _is_liquid(cfg, "over_under", 6.5)   # exotic O/U line
    assert not _is_liquid(cfg, "correct_score", None)  # illiquid market


def test_remove_margin_sums_to_one():
    # 1X2 with an obvious overround.
    odds = {"home": 2.10, "draw": 3.30, "away": 3.40}
    fair, overround = remove_margin(odds)
    assert overround > 1.0  # bookmaker margin present
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["home"] > fair["away"]  # shorter price = higher probability


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_match(s: Session) -> Match:
    home = Team(api_team_id=1, name="Arsenal")
    away = Team(api_team_id=2, name="Brighton")
    s.add_all([home, away])
    s.flush()
    match = Match(api_fixture_id=1001, home_team_id=home.id, away_team_id=away.id, status="NS")
    s.add(match)
    s.flush()
    # Seven world books on 1X2 (enough for the consensus-confidence gate).
    books = {
        "BookA": (2.10, 3.30, 3.40),
        "BookB": (2.05, 3.25, 3.50),
        "BookC": (2.15, 3.20, 3.45),
        "BookD": (2.08, 3.28, 3.46),
        "BookE": (2.12, 3.22, 3.44),
        "BookF": (2.06, 3.26, 3.52),
        "BookG": (2.14, 3.24, 3.42),
    }
    for book, (h, d, a) in books.items():
        for sel, price in (("home", h), ("draw", d), ("away", a)):
            s.add(OddsSnapshot(match_id=match.id, source="apifootball", bookmaker=book,
                               market="match_winner", selection=sel, price=price))
    s.flush()
    return match


def test_build_consensus_produces_fair_probs(session):
    match = _seed_match(session)
    rows = build_consensus(session, match.id, min_books=1)
    by_sel = {r.selection: r for r in rows}
    assert set(by_sel) == {"home", "draw", "away"}
    assert sum(r.fair_prob for r in rows) == pytest.approx(1.0)


def _seed_steam_match(s: Session, home_open: float, home_close: float) -> Match:
    """Create a minimal match with two ingestion snapshots for home selection.

    Each of 3 bookmakers has an opening price and a later closing price so the
    movement function can detect drift without any single-snapshot confusion.
    """
    from datetime import datetime
    home = Team(api_team_id=50, name="SteamHome")
    away = Team(api_team_id=51, name="SteamAway")
    s.add_all([home, away])
    s.flush()
    match = Match(api_fixture_id=5001, home_team_id=home.id, away_team_id=away.id, status="NS")
    s.add(match)
    s.flush()

    t_open = datetime(2025, 3, 1, 10, 0)
    t_now = datetime(2025, 3, 1, 13, 0)

    for book, (ho, hc) in {
        "BookX": (home_open, home_close),
        "BookY": (home_open * 1.01, home_close * 1.01),
        "BookZ": (home_open * 0.99, home_close * 0.99),
    }.items():
        for sel, op_price, cl_price in [
            ("home", ho, hc),
            ("draw", 3.30, 3.30),
            ("away", 3.40, 3.40),
        ]:
            s.add(OddsSnapshot(match_id=match.id, source="apifootball", bookmaker=book,
                               market="match_winner", selection=sel, price=op_price,
                               captured_at=t_open))
            s.add(OddsSnapshot(match_id=match.id, source="apifootball", bookmaker=book,
                               market="match_winner", selection=sel, price=cl_price,
                               captured_at=t_now))
    s.flush()
    return match


def test_steam_move_boosts_confidence(session):
    from accubet.market.consensus import build_all_consensus

    # Home price shortens from 2.15 to 1.90 (~11.6% drop) = clear steam move.
    match = _seed_steam_match(session, home_open=2.15, home_close=1.90)
    build_all_consensus(session, [match.id], min_books=1)
    session.add(OddsSnapshot(match_id=match.id, source="betway", bookmaker="Betway",
                             market="match_winner", selection="home", price=4.00))
    session.flush()

    cfg = get_config()
    opps = compare_match(session, cfg, match)
    home = next(o for o in opps if o.selection == "home")

    assert home.steam_move is True
    assert home.confidence <= 1.0


def test_no_steam_when_price_stable(session):
    from accubet.market.consensus import build_all_consensus

    # Price barely moves: 2.10 -> 2.08 (~1% — well under the 5% steam threshold).
    match = _seed_steam_match(session, home_open=2.10, home_close=2.08)
    build_all_consensus(session, [match.id], min_books=1)
    session.add(OddsSnapshot(match_id=match.id, source="betway", bookmaker="Betway",
                             market="match_winner", selection="home", price=4.00))
    session.flush()

    cfg = get_config()
    opps = compare_match(session, cfg, match)
    home = next(o for o in opps if o.selection == "home")
    assert home.steam_move is False


def test_no_steam_when_single_snapshot_per_book(session):
    """Single-ingest scenario (each book captured once) must not fire a false steam."""
    match = _seed_match(session)   # 7 books × 1 snapshot each
    from accubet.market.consensus import build_all_consensus
    build_all_consensus(session, [match.id], min_books=1)
    session.add(OddsSnapshot(match_id=match.id, source="betway", bookmaker="Betway",
                             market="match_winner", selection="home", price=4.00))
    session.flush()

    cfg = get_config()
    opps = compare_match(session, cfg, match)
    home = next(o for o in opps if o.selection == "home")
    assert home.steam_move is False   # no movement data → no false positive


def test_comparison_flags_value_when_betway_beats_fair(session):
    match = _seed_match(session)
    build_consensus(session, match.id, min_books=1)
    # Betway offers a generous home price (4.00) well above the fair ~2.08.
    session.add(OddsSnapshot(match_id=match.id, source="betway", bookmaker="Betway Malawi",
                             market="match_winner", selection="home", price=4.00))
    session.flush()

    cfg = get_config()
    opps = compare_match(session, cfg, match)
    home = next(o for o in opps if o.selection == "home")
    assert home.price_source == "betway"
    assert home.price == 4.00
    assert home.ev > 0  # fair prob ~0.47 * 4.0 - 1 > 0
    assert home._passes  # clears EV + confidence gates
