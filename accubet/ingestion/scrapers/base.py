"""Local bookmaker connector interface.

Every local book (Betway now; BetPawa/PremierBet later) implements
:class:`LocalBookConnector.fetch_match_odds`, returning the *normalized* odds rows
``{bookmaker, market, selection, line, price}`` the rest of the engine understands.

A :class:`ManualOddsConnector` is provided so the comparison engine works end-to-end
today by reading odds you paste into ``data/manual_odds/<book>/<fixture_id>.json`` — no
live scraping required to validate the pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from accubet.config import PROJECT_ROOT
from accubet.logging_setup import get_logger

log = get_logger(__name__)


class LocalBookConnector(ABC):
    name: str = "local"
    source_key: str = "local"  # stored as OddsSnapshot.source; must be unique per book

    @abstractmethod
    def fetch_match_odds(self, match: Any) -> list[dict]:
        """Return normalized odds rows for a single match (may be empty)."""
        raise NotImplementedError


class ManualOddsConnector(LocalBookConnector):
    """Reads odds from JSON files: ``data/manual_odds/<slug>/<fixture_id>.json``.

    File contents = a JSON list of ``{market, selection, line, price}`` (bookmaker is
    filled from ``name``). Lets us drive Betway-vs-fair comparison from hand-entered prices
    before/instead of live scraping.
    """

    def __init__(self, name: str = "Betway Malawi", slug: str = "betway"):
        self.name = name
        self.dir = PROJECT_ROOT / "data" / "manual_odds" / slug

    def fetch_match_odds(self, match: Any) -> list[dict]:
        path = self.dir / f"{match.api_fixture_id}.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read manual odds %s: %s", path, exc)
            return []
        rows = []
        for r in data:
            price = r.get("price")
            if not price:
                continue
            rows.append(
                {
                    "bookmaker": self.name,
                    "market": r.get("market"),
                    "selection": r.get("selection"),
                    "line": r.get("line"),
                    "price": float(price),
                }
            )
        return rows
