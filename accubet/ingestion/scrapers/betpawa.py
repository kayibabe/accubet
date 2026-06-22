"""BetPawa Malawi connector — live odds via the public sportsbook JSON feed.

BetPawa (betpawa.mw) serves its fixture list from a plain JSON REST endpoint
that can be called directly with requests — no browser automation needed.

Endpoint discovery:
    Open betpawa.mw in Chrome -> DevTools -> Network -> filter XHR/Fetch ->
    click "Sports" / "Football" and look for a request returning a JSON array
    of events.  Copy the URL and update _HIGHLIGHTS_URL below if it differs.

Currently mapped to the Sportradar-backed feed structure that BetPawa Africa
sites share.  The parser is intentionally lenient so minor response shape
changes don't hard-crash ingestion.

A hand-entered override at ``data/manual_odds/betpawa/<fixture_id>.json``
always wins, same as the Betway connector.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import timezone
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

# Primary highlights endpoint — update if BetPawa rotates their API URL.
# Discovered via DevTools Network tab on betpawa.mw.
_HIGHLIGHTS_URL = "https://www.betpawa.mw/api/v1/events/highlights"
_EVENTS_URL = "https://www.betpawa.mw/api/v1/sports/1/events"

# Canonical market name mapping from BetPawa's market identifiers.
_MARKET_MAP: dict[str, str] = {
    # 1X2 / Match Winner variants
    "1x2": "match_winner",
    "match_odds": "match_winner",
    "match winner": "match_winner",
    "win draw win": "match_winner",
    "full time result": "match_winner",
    # BTTS
    "both teams to score": "btts",
    "both teams score": "btts",
    "btts": "btts",
    # Over/Under
    "over/under": "over_under",
    "total goals": "over_under",
    "goals over/under": "over_under",
    # Double chance
    "double chance": "double_chance",
}

# Tokens that carry no name signal (same stop list as Betway for consistency).
_STOP = {"fc", "sc", "afc", "cf", "ac", "ss", "if", "sk", "bk", "club", "the", "de", "united"}


def _tokens(name: str) -> set[str]:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return {t for t in s.split() if t and t not in _STOP}


class BetPawaMalawiConnector(LocalBookConnector):
    """Fetch live BetPawa Malawi odds and map them to AccuBet's canonical format."""

    name = "BetPawa Malawi"
    source_key = "betpawa"

    def __init__(self, cfg: AppConfig, cache_ttl_seconds: int = 900):
        self.cfg = cfg
        self._manual = ManualOddsConnector(name=self.name, slug="betpawa")
        self._cache_path = PROJECT_ROOT / "data" / "cache" / "betpawa_highlights.json"
        self._cache_ttl = cache_ttl_seconds
        self._events: list[dict] | None = None

    # -- public API ---------------------------------------------------------

    def fetch_match_odds(self, match: Any) -> list[dict]:
        # Manual override wins.
        manual = self._manual.fetch_match_odds(match)
        if manual:
            return manual

        index = self._ensure_index()
        if not index:
            return []
        best = self._match_event(match, index)
        return best["rows"] if best else []

    # -- HTTP ---------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _http_get(self, url: str) -> dict | list:
        headers = {
            "User-Agent": _UA,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.betpawa.mw/",
            "Origin": "https://www.betpawa.mw",
        }
        params = {"sportId": 1, "count": 100, "page": 1}
        resp = requests.get(url, params=params, headers=headers, timeout=25)
        resp.raise_for_status()
        return resp.json()

    def _load_feed(self) -> dict | list | None:
        # Short-lived disk cache so repeated ingest runs don't hammer the feed.
        try:
            if self._cache_path.exists():
                cached = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if time.time() - cached.get("fetched_at", 0) < self._cache_ttl:
                    return cached["payload"]
        except (OSError, json.JSONDecodeError):
            pass

        payload = None
        for url in (_HIGHLIGHTS_URL, _EVENTS_URL):
            try:
                payload = self._http_get(url)
                if payload:
                    break
            except Exception as exc:
                log.warning("BetPawa feed fetch failed (%s): %s", url, exc)

        if payload is None:
            return None

        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps({"fetched_at": time.time(), "payload": payload}),
                encoding="utf-8",
            )
        except OSError:
            pass
        return payload

    # -- index building -----------------------------------------------------

    def _ensure_index(self) -> list[dict]:
        if self._events is not None:
            return self._events
        data = self._load_feed()
        self._events = self._build_index(data) if data else []
        log.info("BetPawa live feed: %d events indexed.", len(self._events))
        return self._events

    def _build_index(self, data: dict | list) -> list[dict]:
        """Parse the feed payload into a flat index of {home_tokens, away_tokens, rows}."""
        # BetPawa feeds can be wrapped as {"data": {"events": [...]}} or {"events": [...]}
        # or a bare list.  Walk down to the list of event dicts.
        events_raw = self._extract_events(data)
        if not events_raw:
            log.warning("BetPawa: no events found in feed payload.")
            return []

        index = []
        for ev in events_raw:
            rows = self._parse_event(ev)
            if not rows:
                continue
            home = self._team_name(ev, "home")
            away = self._team_name(ev, "away")
            start = self._start_epoch(ev)
            index.append({
                "home_tokens": _tokens(home),
                "away_tokens": _tokens(away),
                "start_epoch": start,
                "name": f"{home} v {away}",
                "rows": rows,
            })
        return index

    @staticmethod
    def _extract_events(data) -> list:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # {"data": {"events": [...]}}  or  {"events": [...]}  or  {"response": [...]}
            for key in ("data", "result", "response"):
                inner = data.get(key)
                if isinstance(inner, dict):
                    data = inner
                    break
            for key in ("events", "items", "fixtures", "matches"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
        return []

    @staticmethod
    def _team_name(ev: dict, side: str) -> str:
        """Extract team name regardless of whether the feed uses nested or flat keys."""
        # Nested: {"homeTeam": {"name": "..."}} or flat: {"homeTeam": "..."}
        for key in (f"{side}Team", f"{side}_team", side):
            val = ev.get(key)
            if isinstance(val, dict):
                return val.get("name") or val.get("teamName") or ""
            if isinstance(val, str):
                return val
        # Some feeds use "name" = "Home v Away"
        name_field = ev.get("name") or ev.get("eventName") or ""
        parts = re.split(r"\s+[vV][sS]?\s+|\s+-\s+", name_field, maxsplit=1)
        return parts[0 if side == "home" else 1].strip() if len(parts) == 2 else ""

    @staticmethod
    def _start_epoch(ev: dict) -> float | None:
        for key in ("startTime", "start_time", "kickOff", "kickoff",
                    "expectedStartEpoch", "openDate", "date"):
            val = ev.get(key)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                # epoch in ms vs s heuristic
                return val / 1000 if val > 1e10 else float(val)
            if isinstance(val, str):
                try:
                    from datetime import datetime as _dt
                    return _dt.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    pass
        return None

    def _parse_event(self, ev: dict) -> list[dict]:
        """Extract normalized odds rows from one event dict."""
        home = self._team_name(ev, "home")
        away = self._team_name(ev, "away")
        rows: list[dict] = []

        markets_raw = ev.get("markets") or ev.get("marketGroups") or []
        for mkt in markets_raw:
            mkt_name = (
                mkt.get("name") or mkt.get("marketName") or mkt.get("type") or ""
            ).strip().lower()
            canonical = _MARKET_MAP.get(mkt_name)
            if canonical is None:
                continue

            outcomes = mkt.get("outcomes") or mkt.get("selections") or []
            for o in outcomes:
                sel, line = self._map_outcome(
                    canonical,
                    o.get("name") or o.get("label") or "",
                    home, away,
                    o.get("line") or o.get("handicap") or mkt.get("line"),
                )
                if sel is None:
                    continue
                price = o.get("odds") or o.get("price") or o.get("priceDecimal")
                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue
                if price <= 1.0:
                    continue
                rows.append({
                    "bookmaker": self.name,
                    "market": canonical,
                    "selection": sel,
                    "line": line,
                    "price": price,
                })
        return rows

    @staticmethod
    def _map_outcome(
        market: str, name: str, home: str, away: str, raw_line: Any
    ) -> tuple[str | None, float | None]:
        n = name.strip().lower()
        hl, al = (home or "").strip().lower(), (away or "").strip().lower()

        if market == "match_winner":
            # BetPawa uses "1", "X", "2" or full team names
            if n in ("1", "home", hl):
                return "home", None
            if n in ("x", "draw", "the draw"):
                return "draw", None
            if n in ("2", "away", al):
                return "away", None
            return None, None

        if market == "btts":
            if n in ("yes", "gg"):
                return "yes", None
            if n in ("no", "ng"):
                return "no", None
            return None, None

        if market == "double_chance":
            if n in ("1x", "1 or x", f"{hl} or draw"):
                return "1x", None
            if n in ("12", "1 or 2", f"{hl} or {al}"):
                return "12", None
            if n in ("x2", "x or 2", f"draw or {al}"):
                return "x2", None
            return None, None

        if market == "over_under":
            try:
                line = float(raw_line) if raw_line not in (None, 0, "0") else None
            except (TypeError, ValueError):
                line = None
            if n.startswith("over") or n == "o":
                return "over", line
            if n.startswith("under") or n == "u":
                return "under", line
            return None, None

        return None, None

    # -- event matching -----------------------------------------------------

    def _match_event(self, match: Any, index: list[dict]) -> dict | None:
        home = _tokens(match.home_team.name if match.home_team else "")
        away = _tokens(match.away_team.name if match.away_team else "")
        if not home or not away:
            return None

        kickoff_epoch = None
        if match.kickoff:
            ko = (
                match.kickoff
                if match.kickoff.tzinfo
                else match.kickoff.replace(tzinfo=timezone.utc)
            )
            kickoff_epoch = ko.timestamp()

        best, best_score = None, 0
        for ev in index:
            h = len(home & ev["home_tokens"])
            a = len(away & ev["away_tokens"])
            if h == 0 or a == 0:
                continue
            if kickoff_epoch and ev["start_epoch"]:
                if abs(kickoff_epoch - ev["start_epoch"]) > 2 * 86400:
                    continue
            score = h + a
            if score > best_score:
                best, best_score = ev, score
        return best
