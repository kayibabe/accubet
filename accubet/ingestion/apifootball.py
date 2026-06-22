"""API-Football (api-sports.io) HTTP client — cache-first and quota-guarded.

Every call first checks the local cache; only a miss (or stale entry) touches the network,
and only a real network call is counted against the daily quota. The free tier also rate
limits to ~10 req/min, which we respect with a small inter-call sleep.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from accubet.config import AppConfig
from accubet.ingestion import cache, quota
from accubet.logging_setup import get_logger

log = get_logger(__name__)


class ApiFootballError(RuntimeError):
    pass


class ApiFootballClient:
    """Thin client returning the parsed JSON ``response`` list for an endpoint."""

    def __init__(self, cfg: AppConfig, session: Session):
        self.cfg = cfg
        self.session = session
        self.base_url = cfg.secrets.apifootball_base_url.rstrip("/")
        self._headers = {"x-apisports-key": cfg.secrets.apifootball_key}
        self._min_interval = 60.0 / max(1, cfg.apifootball.rate_limit_per_minute)
        self._last_call = 0.0
        # Set on each get(): True when the last response was served from cache (no quota).
        self.last_from_cache = False

    # -- low-level ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _http_get(self, endpoint: str, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, headers=self._headers, params=params, timeout=30)
        if resp.status_code >= 500:
            raise requests.ConnectionError(f"{resp.status_code} from {endpoint}")
        resp.raise_for_status()
        return resp.json()

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    # -- core fetch (cache + quota) ----------------------------------------
    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        essential: bool = False,
        ttl_minutes: int | None = None,
        force: bool = False,
    ) -> dict:
        """Return the full API JSON for ``endpoint`` (with ``response``/``errors`` keys).

        Served from cache when fresh; otherwise a quota-guarded network call that is then
        cached and counted.
        """
        params = params or {}
        ttl = ttl_minutes if ttl_minutes is not None else self.cfg.apifootball.cache_ttl_minutes
        self.last_from_cache = False

        if not force:
            cached = cache.get_fresh(self.session, endpoint, params, ttl)
            if cached is not None:
                log.debug("cache hit: %s %s", endpoint, params)
                self.last_from_cache = True
                return cached

        quota.guard(self.session, self.cfg, endpoint, essential=essential)
        if not self.cfg.secrets.apifootball_key:
            raise ApiFootballError("APIFOOTBALL_KEY is not set — add it to your .env file.")

        self._respect_rate_limit()
        log.info("API call: %s %s", endpoint, params)
        data = self._http_get(endpoint, params)
        quota.record_request(self.session, endpoint, success=True)

        errors = data.get("errors")
        if errors:
            # API-Football reports auth/quota issues here, not via HTTP status.
            log.warning("API-Football errors for %s: %s", endpoint, errors)

        cache.set_entry(self.session, endpoint, params, data)
        return data

    @staticmethod
    def response(data: dict) -> list[dict]:
        return data.get("response", []) or []

    # -- typed helpers ------------------------------------------------------
    def fixtures_by_date(self, date_str: str, *, force: bool = False) -> list[dict]:
        """All fixtures on a date (one request) — filtered to configured leagues locally."""
        data = self.get(
            "fixtures",
            {"date": date_str},
            essential=True,           # the daily slate is the one call we always allow
            ttl_minutes=self.cfg.apifootball.cache_ttl_minutes,
            force=force,
        )
        return self.response(data)

    def odds_for_fixture(self, fixture_id: int) -> list[dict]:
        """Odds (all books) for a single fixture — one request, cached per fixture."""
        data = self.get("odds", {"fixture": fixture_id}, essential=False)
        return self.response(data)

    def results_by_date(self, date_str: str, *, force: bool = False) -> list[dict]:
        data = self.get("fixtures", {"date": date_str}, essential=True, force=force)
        return self.response(data)
