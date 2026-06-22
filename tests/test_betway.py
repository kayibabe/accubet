"""Tests for the Betway connector's feed-join and team-matching logic (no network)."""

from accubet.config import get_config
from accubet.ingestion.scrapers.betway import BetwayMalawiConnector, _tokens


def test_team_token_matching_handles_name_variants():
    assert _tokens("IR Iran") & _tokens("Iran")
    assert _tokens("Cape Verde Islands") & _tokens("Cape Verde")
    assert not (_tokens("Belgium") & _tokens("Uruguay"))


def test_build_index_joins_prices_to_labeled_outcomes():
    data = {
        "events": [{
            "eventId": 1, "name": "Belgium vs. IR Iran",
            "homeTeam": "Belgium", "awayTeam": "IR Iran",
            "expectedStartEpoch": 1782068400,
        }],
        "markets": [
            {"marketId": 11, "eventId": 1, "name": "[Win/Draw/Win]"},
            {"marketId": 12, "eventId": 1, "name": "[Both Teams To Score]"},
            {"marketId": 13, "eventId": 1, "name": "[Over/Under]"},
        ],
        "outcomes": [
            {"outcomeId": 111, "marketId": 11, "eventId": 1, "name": "Belgium"},
            {"outcomeId": 112, "marketId": 11, "eventId": 1, "name": "Draw"},
            {"outcomeId": 113, "marketId": 11, "eventId": 1, "name": "IR Iran"},
            {"outcomeId": 121, "marketId": 12, "eventId": 1, "name": "Yes"},
            {"outcomeId": 122, "marketId": 12, "eventId": 1, "name": "No"},
            {"outcomeId": 131, "marketId": 13, "eventId": 1, "name": "Over", "sbv": 2.5},
            {"outcomeId": 132, "marketId": 13, "eventId": 1, "name": "Under", "sbv": 2.5},
        ],
        "prices": [
            {"outcomeId": 111, "priceDecimal": 1.45},
            {"outcomeId": 112, "priceDecimal": 3.9},
            {"outcomeId": 113, "priceDecimal": 8.0},
            {"outcomeId": 121, "priceDecimal": 2.65},
            {"outcomeId": 122, "priceDecimal": 1.47},
            {"outcomeId": 131, "priceDecimal": 1.85},
            {"outcomeId": 132, "priceDecimal": 2.11},
        ],
    }
    conn = BetwayMalawiConnector(get_config())
    idx = conn._build_index(data)
    assert len(idx) == 1
    rows = {(r["market"], r["selection"], r["line"]): r["price"] for r in idx[0]["rows"]}
    assert rows[("match_winner", "home", None)] == 1.45
    assert rows[("match_winner", "away", None)] == 8.0
    assert rows[("btts", "yes", None)] == 2.65
    assert rows[("over_under", "over", 2.5)] == 1.85
