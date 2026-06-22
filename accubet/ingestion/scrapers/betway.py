"""Betway Malawi connector — live odds via the public sportsbook REST feed.

Betway's site (betway.mw) is a Nuxt SPA, but its bet-book is served by a plain JSON
endpoint we can call directly with ``requests`` (no browser needed at runtime):

    GET https://www.betway.mw/sportsapi/br/v1/BetBook/Highlights/
        ?countryCode=MW&sportId=soccer&Skip=0&Take=..&cultureCode=en-US
        &boostedOnly=false&marketTypes=[Win/Draw/Win]&marketTypes=[Double Chance]
        &marketTypes=[Both Teams To Score]&marketTypes=[Over/Under]

The response is a relational snapshot — parallel ``events / markets / outcomes / prices``
arrays — which we join (price→outcome→market→event) into normalized rows and match to
API-Football fixtures by team name + kickoff.

A hand-entered override at ``data/manual_odds/betway/<fixture_id>.json`` still wins, for
testing or when the live feed is unavailable. Endpoints/markup can change without notice
(their ToS applies); this connector is isolated so the rest of the engine is unaffected.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from accubet.config import PROJECT_ROOT, AppConfig
from accubet.ingestion.scrapers.base import LocalBookConnector, ManualOddsConnector
from accubet.logging_setup import get_logger

log = get_logger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_BETBOOK_URL = "https://www.betway.mw/sportsapi/br/v1/BetBook/Highlights/"
_MARKET_TYPES = ["[Win/Draw/Win]", "[Double Chance]", "[Both Teams To Score]", "[Over/Under]"]

# Bet-book market name (lowercased) -> canonical market.
_MARKET_MAP = {
    "[win/draw/win]": "match_winner",
    "[both teams to score]": "btts",
    "[double chance]": "double_chance",
    "[over/under]": "over_under",
    "[total goals]": "over_under",
}

_STOP = {"fc", "sc", "afc", "cf", "ac", "ss", "if", "sk", "bk", "club", "the", "de"}


def _tokens(name: str) -> set[str]:
    """Normalized significant tokens of a team name (accent/punct-insensitive)."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return {t for t in s.split() if t and t not in _STOP}


class BetwayMalawiConnector(LocalBookConnector):
    name = "Betway Malawi"

    def __init__(self, cfg: AppConfig, cache_ttl_seconds: int = 900):
        self.cfg = cfg
        self._manual = ManualOddsConnector(name=self.name, slug="betway")
        self._cache_path = PROJECT_ROOT / "data" / "cache" / "betway_highlights.json"
        self._cache_ttl = cache_ttl_seconds
        self._events: list[dict] | None = None  # built index of betway events + rows

    # -- public -------------------------------------------------------------
    def fetch_match_odds(self, match: Any) -> list[dict]:
        # 1) hand-entered override always wins.
        manual = self._manual.fetch_match_odds(match)
        if manual:
            return manual

        # 2) live feed, matched by team names (+ kickoff sanity check).
        index = self._ensure_index()
        if not index:
            return []
        best = self._match_event(match, index)
        return best["rows"] if best else []

    # -- live feed ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _http_get(self) -> dict:
        params = [
            ("countryCode", "MW"), ("sportId", "soccer"), ("Skip", "0"), ("Take", "60"),
            ("cultureCode", "en-US"), ("isEsport", "false"), ("boostedOnly", "false"),
        ] + [("marketTypes", mt) for mt in _MARKET_TYPES]
        headers = {"User-Agent": _UA, "Referer": "https://www.betway.mw/sport/soccer",
                   "Accept": "application/json, text/plain, */*"}
        resp = requests.get(_BETBOOK_URL, params=params, headers=headers, timeout=25)
        resp.raise_for_status()
        return resp.json()

    def _load_betbook(self) -> dict | None:
        # short-lived disk cache to avoid hammering the feed on repeated runs
        try:
            if self._cache_path.exists():
                cached = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if time.time() - cached.get("fetched_at", 0) < self._cache_ttl:
                    return cached["payload"]
        except (OSError, json.JSONDecodeError):
            pass
        try:
            payload = self._http_get()
        except Exception as exc:  # network/feed problems must not break ingestion
            log.warning("Betway feed fetch failed: %s", exc)
            return None
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps({"fetched_at": time.time(), "payload": payload}), encoding="utf-8"
            )
        except OSError:
            pass
        return payload

    def _ensure_index(self) -> list[dict]:
        if self._events is not None:
            return self._events
        data = self._load_betbook()
        self._events = self._build_index(data) if data else []
        log.info("Betway live feed: %d events indexed.", len(self._events))
        return self._events

    def _build_index(self, data: dict) -> list[dict]:
        events = {e["eventId"]: e for e in data.get("events", [])}
        markets = {m["marketId"]: m for m in data.get("markets", [])}
        prices = {p["outcomeId"]: p.get("priceDecimal") for p in data.get("prices", [])}

        rows_by_event: dict[Any, list[dict]] = {}
        for o in data.get("outcomes", []):
            price = prices.get(o.get("outcomeId"))
            market = markets.get(o.get("marketId"))
            event = events.get(o.get("eventId"))
            if not price or price <= 1.0 or market is None or event is None:
                continue
            canonical = _MARKET_MAP.get((market.get("name") or "").strip().lower())
            if canonical is None:
                continue
            selection, line = self._map_outcome(
                canonical, o.get("name", ""), event.get("homeTeam", ""),
                event.get("awayTeam", ""), o.get("sbv"), o.get("handicap"),
            )
            if selection is None:
                continue
            rows_by_event.setdefault(o["eventId"], []).append(
                {"bookmaker": self.name, "market": canonical, "selection": selection,
                 "line": line, "price": float(price)}
            )

        index = []
        for eid, rows in rows_by_event.items():
            ev = events[eid]
            index.append({
                "home_tokens": _tokens(ev.get("homeTeam", "")),
                "away_tokens": _tokens(ev.get("awayTeam", "")),
                "start_epoch": ev.get("expectedStartEpoch"),
                "name": ev.get("name", ""),
                "rows": rows,
            })
        return index

    @staticmethod
    def _map_outcome(market, name, home, away, sbv, handicap):
        n = (name or "").strip().lower()
        if market == "match_winner":
            if n == (home or "").lower():
                return "home", None
            if n == (away or "").lower():
                return "away", None
            if n == "draw":
                return "draw", None
            return None, None
        if market == "btts":
            return {"yes": "yes", "no": "no"}.get(n), None
        if market == "double_chance":
            hl, al = (home or "").lower(), (away or "").lower()
            if n in (f"{hl} or draw", "1 or x", "1 or draw"):
                return "1x", None
            if n in (f"{hl} or {al}", "1 or 2"):
                return "12", None
            if n in (f"draw or {al}", "x or 2", "draw or 2"):
                return "x2", None
            return None, None
        if market == "over_under":
            line = sbv if sbv not in (None, 0) else handicap
            try:
                line = float(line) if line is not None else None
            except (TypeError, ValueError):
                line = None
            if n.startswith("over"):
                return "over", line
            if n.startswith("under"):
                return "under", line
            return None, None
        return None, None

    # -- matching -----------------------------------------------------------
    def _match_event(self, match: Any, index: list[dict]) -> dict | None:
        home = _tokens(match.home_team.name if match.home_team else "")
        away = _tokens(match.away_team.name if match.away_team else "")
        if not home or not away:
            return None

        kickoff_epoch = None
        if match.kickoff:
            ko = match.kickoff if match.kickoff.tzinfo else match.kickoff.replace(tzinfo=timezone.utc)
            kickoff_epoch = ko.timestamp()

        best, best_score = None, 0
        for ev in index:
            h = len(home & ev["home_tokens"])
            a = len(away & ev["away_tokens"])
            if h == 0 or a == 0:
                continue  # both sides must share at least one token, same orientation
            # kickoff sanity: within ~2 days if both known
            if kickoff_epoch and ev["start_epoch"]:
                if abs(kickoff_epoch - ev["start_epoch"]) > 2 * 86400:
                    continue
            score = h + a
            if score > best_score:
                best, best_score = ev, score
        return best
