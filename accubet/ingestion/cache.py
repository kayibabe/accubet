"""Cache-first storage for API responses.

Every API payload is persisted in ``api_cache`` keyed by (endpoint, params). Reads are
served from cache until the entry is older than the TTL. Combined with the quota guard and
the fixtures *diff-before-pull* check in the client, re-running ``ingest`` on an unchanged
slate makes zero network calls.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.storage.models import ApiCache


def _canonical(params: dict[str, Any] | None) -> str:
    """Stable JSON for hashing (sorted keys, no whitespace)."""
    return json.dumps(params or {}, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(endpoint: str, params: dict[str, Any] | None) -> tuple[str, str]:
    """Return ``(params_json, params_hash)`` for an endpoint+params pair."""
    params_json = _canonical(params)
    digest = hashlib.sha256(f"{endpoint}|{params_json}".encode()).hexdigest()
    return params_json, digest


def get_entry(session: Session, endpoint: str, params: dict[str, Any] | None) -> ApiCache | None:
    _, digest = cache_key(endpoint, params)
    stmt = select(ApiCache).where(
        ApiCache.endpoint == endpoint, ApiCache.params_hash == digest
    )
    return session.execute(stmt).scalar_one_or_none()


def get_fresh(
    session: Session,
    endpoint: str,
    params: dict[str, Any] | None,
    ttl_minutes: int,
) -> Any | None:
    """Return the cached payload if present and within TTL, else ``None``."""
    entry = get_entry(session, endpoint, params)
    if entry is None:
        return None
    age = datetime.utcnow() - entry.fetched_at
    if age > timedelta(minutes=ttl_minutes):
        return None
    return entry.payload


def set_entry(
    session: Session,
    endpoint: str,
    params: dict[str, Any] | None,
    payload: Any,
) -> None:
    """Insert or update the cache entry for an endpoint+params pair."""
    params_json, digest = cache_key(endpoint, params)
    entry = get_entry(session, endpoint, params)
    if entry is None:
        entry = ApiCache(
            endpoint=endpoint,
            params_hash=digest,
            params_json=params_json,
            payload=payload,
            fetched_at=datetime.utcnow(),
        )
        session.add(entry)
    else:
        entry.payload = payload
        entry.params_json = params_json
        entry.fetched_at = datetime.utcnow()
    session.flush()
