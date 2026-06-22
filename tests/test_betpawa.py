"""Tests for the BetPawa Malawi connector."""

import pytest

from accubet.ingestion.scrapers.betpawa import BetPawaMalawiConnector, _tokens


# --- token helper -----------------------------------------------------------

def test_tokens_strips_stopwords_and_accents():
    assert "arsenal" in _tokens("Arsenal FC")
    assert "fc" not in _tokens("Arsenal FC")
    assert "manchester" in _tokens("Manchester United")
    assert "united" not in _tokens("Manchester United")


def test_tokens_handles_accented_chars():
    assert "strasbourg" in _tokens("RC Strasbourg")


# --- _extract_events --------------------------------------------------------

def test_extract_events_bare_list():
    data = [{"id": 1}, {"id": 2}]
    assert BetPawaMalawiConnector._extract_events(data) == data


def test_extract_events_nested_dict():
    data = {"data": {"events": [{"id": 1}]}}
    assert BetPawaMalawiConnector._extract_events(data) == [{"id": 1}]


def test_extract_events_top_level_events():
    data = {"events": [{"id": 10}]}
    assert BetPawaMalawiConnector._extract_events(data) == [{"id": 10}]


def test_extract_events_unknown_shape_returns_empty():
    assert BetPawaMalawiConnector._extract_events({"foo": "bar"}) == []


# --- _team_name -------------------------------------------------------------

def test_team_name_flat_string():
    ev = {"homeTeam": "Arsenal", "awayTeam": "Chelsea"}
    assert BetPawaMalawiConnector._team_name(ev, "home") == "Arsenal"
    assert BetPawaMalawiConnector._team_name(ev, "away") == "Chelsea"


def test_team_name_nested_dict():
    ev = {"homeTeam": {"name": "Man City"}, "awayTeam": {"name": "Liverpool"}}
    assert BetPawaMalawiConnector._team_name(ev, "home") == "Man City"


def test_team_name_from_name_field():
    ev = {"name": "Arsenal v Chelsea"}
    assert BetPawaMalawiConnector._team_name(ev, "home") == "Arsenal"
    assert BetPawaMalawiConnector._team_name(ev, "away") == "Chelsea"


# --- _map_outcome -----------------------------------------------------------

def test_map_outcome_1x2_numeric():
    assert BetPawaMalawiConnector._map_outcome("match_winner", "1", "Arsenal", "Chelsea", None) == ("home", None)
    assert BetPawaMalawiConnector._map_outcome("match_winner", "X", "Arsenal", "Chelsea", None) == ("draw", None)
    assert BetPawaMalawiConnector._map_outcome("match_winner", "2", "Arsenal", "Chelsea", None) == ("away", None)


def test_map_outcome_1x2_by_team_name():
    assert BetPawaMalawiConnector._map_outcome("match_winner", "arsenal", "Arsenal", "Chelsea", None) == ("home", None)
    assert BetPawaMalawiConnector._map_outcome("match_winner", "chelsea", "Arsenal", "Chelsea", None) == ("away", None)


def test_map_outcome_btts():
    assert BetPawaMalawiConnector._map_outcome("btts", "yes", "", "", None) == ("yes", None)
    assert BetPawaMalawiConnector._map_outcome("btts", "GG", "", "", None) == ("yes", None)
    assert BetPawaMalawiConnector._map_outcome("btts", "no", "", "", None) == ("no", None)


def test_map_outcome_over_under():
    sel, line = BetPawaMalawiConnector._map_outcome("over_under", "Over", "", "", 2.5)
    assert sel == "over" and line == pytest.approx(2.5)
    sel, line = BetPawaMalawiConnector._map_outcome("over_under", "Under", "", "", 2.5)
    assert sel == "under" and line == pytest.approx(2.5)


def test_map_outcome_unknown_returns_none():
    assert BetPawaMalawiConnector._map_outcome("match_winner", "??", "A", "B", None) == (None, None)


# --- _parse_event -----------------------------------------------------------

def _make_event(home="Arsenal", away="Chelsea"):
    return {
        "homeTeam": home,
        "awayTeam": away,
        "startTime": 1_700_000_000,
        "markets": [
            {
                "name": "1X2",
                "outcomes": [
                    {"name": "1", "odds": 2.10},
                    {"name": "X", "odds": 3.30},
                    {"name": "2", "odds": 3.40},
                ],
            },
            {
                "name": "Both Teams To Score",
                "outcomes": [
                    {"name": "yes", "odds": 1.75},
                    {"name": "no", "odds": 2.05},
                ],
            },
        ],
    }


def test_parse_event_returns_rows(tmp_path, monkeypatch):
    from accubet.config import AppConfig, Secrets
    cfg = AppConfig(secrets=Secrets())
    connector = BetPawaMalawiConnector(cfg)
    rows = connector._parse_event(_make_event())
    assert len(rows) == 5  # 3 match_winner + 2 btts
    markets = {r["market"] for r in rows}
    assert "match_winner" in markets
    assert "btts" in markets


def test_parse_event_filters_bad_prices(tmp_path):
    from accubet.config import AppConfig, Secrets
    cfg = AppConfig(secrets=Secrets())
    connector = BetPawaMalawiConnector(cfg)
    ev = _make_event()
    ev["markets"][0]["outcomes"][0]["odds"] = 0.9  # sub-1 price — should be dropped
    rows = connector._parse_event(ev)
    home_rows = [r for r in rows if r["market"] == "match_winner" and r["selection"] == "home"]
    assert len(home_rows) == 0


# --- source_key ------------------------------------------------------------

def test_source_key():
    from accubet.config import AppConfig, Secrets
    cfg = AppConfig(secrets=Secrets())
    connector = BetPawaMalawiConnector(cfg)
    assert connector.source_key == "betpawa"
    assert connector.name == "BetPawa Malawi"
