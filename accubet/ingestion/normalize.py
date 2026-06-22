"""Normalize API-Football payloads into AccuBet's common schema.

Bookmaker market/selection names vary; everything is mapped to a small canonical
vocabulary so models and the value engine never see provider-specific strings:

* markets:    match_winner, double_chance, draw_no_bet, over_under, btts
* selections: home / draw / away · yes / no · over / under · 1x / 12 / x2
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator


# --- fixtures ---------------------------------------------------------------

def parse_kickoff(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_fixture(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten one /fixtures response element into our match fields."""
    fx = item.get("fixture", {})
    lg = item.get("league", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})
    venue = (fx.get("venue") or {}).get("name") or ""
    status = (fx.get("status") or {}).get("short") or "NS"
    return {
        "fixture_id": fx.get("id"),
        "kickoff": parse_kickoff(fx.get("date")),
        "venue": venue,
        "status": status,
        "league_id": lg.get("id"),
        "league_name": lg.get("name") or "",
        "country": lg.get("country") or "",
        "season": lg.get("season"),
        "home_id": (teams.get("home") or {}).get("id"),
        "home_name": (teams.get("home") or {}).get("name") or "",
        "away_id": (teams.get("away") or {}).get("id"),
        "away_name": (teams.get("away") or {}).get("name") or "",
        "home_goals": (goals or {}).get("home"),
        "away_goals": (goals or {}).get("away"),
    }


# --- odds -------------------------------------------------------------------

def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_selection(market: str, raw: str) -> tuple[str | None, float | None]:
    """Return (canonical_selection, line) for a raw bookmaker value, or (None, None)."""
    v = (raw or "").strip().lower()
    if market == "match_winner":
        return {"home": "home", "draw": "draw", "away": "away"}.get(v), None
    if market == "draw_no_bet":
        return {"home": "home", "away": "away"}.get(v), None
    if market == "btts":
        return {"yes": "yes", "no": "no"}.get(v), None
    if market == "double_chance":
        return {
            "home/draw": "1x", "1x": "1x",
            "home/away": "12", "12": "12",
            "draw/away": "x2", "x2": "x2",
        }.get(v), None
    if market == "over_under":
        parts = v.split()
        if len(parts) == 2 and parts[0] in ("over", "under"):
            return parts[0], _f(parts[1])
        return None, None
    return None, None


# Canonical market keyed by lowercased bookmaker bet name.
_BET_NAME_TO_MARKET = {
    "match winner": "match_winner",
    "double chance": "double_chance",
    "draw no bet": "draw_no_bet",
    "goals over/under": "over_under",
    "over/under": "over_under",
    "both teams score": "btts",
    "both teams to score": "btts",
}


def iter_normalized_odds(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield normalized odds rows from one /odds response element.

    Each row: ``{bookmaker, market, selection, line, price}``.
    """
    for bm in item.get("bookmakers", []) or []:
        book = bm.get("name") or ""
        for bet in bm.get("bets", []) or []:
            market = _BET_NAME_TO_MARKET.get((bet.get("name") or "").strip().lower())
            if market is None:
                continue
            for val in bet.get("values", []) or []:
                price = _f(val.get("odd"))
                selection, line = _map_selection(market, val.get("value", ""))
                if selection is None or price is None or price <= 1.0:
                    continue
                yield {
                    "bookmaker": book,
                    "market": market,
                    "selection": selection,
                    "line": line,
                    "price": price,
                }
