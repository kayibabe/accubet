"""Daily API quota guard.

API-Football's free tier allows 100 requests/day. We count every *real* call in
``request_log`` and refuse non-essential calls once we cross the soft stop (default 90),
leaving headroom so a scan never dies mid-run on a hard 429.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from accubet.config import AppConfig
from accubet.logging_setup import get_logger
from accubet.storage.models import RequestLog

log = get_logger(__name__)


class QuotaExceeded(RuntimeError):
    """Raised when a call would exceed the configured daily request budget."""


def requests_used_today(session: Session, day: date | None = None) -> int:
    day = day or date.today()
    stmt = select(func.count()).select_from(RequestLog).where(RequestLog.day == day)
    return int(session.execute(stmt).scalar_one())


def remaining(session: Session, cfg: AppConfig, day: date | None = None) -> int:
    return max(0, cfg.apifootball.daily_request_limit - requests_used_today(session, day))


def can_request(session: Session, cfg: AppConfig, *, essential: bool = False) -> bool:
    """Whether another call is allowed.

    ``essential`` calls (e.g. the daily fixtures diff) may use the full daily limit;
    everything else stops at the soft cap to preserve headroom.
    """
    used = requests_used_today(session)
    ceiling = (
        cfg.apifootball.daily_request_limit if essential else cfg.apifootball.quota_soft_stop
    )
    return used < ceiling


def record_request(session: Session, endpoint: str, *, success: bool = True) -> None:
    session.add(RequestLog(day=date.today(), endpoint=endpoint, success=success))
    session.flush()


def guard(session: Session, cfg: AppConfig, endpoint: str, *, essential: bool = False) -> None:
    """Raise :class:`QuotaExceeded` if a call to ``endpoint`` is not permitted."""
    if not can_request(session, cfg, essential=essential):
        used = requests_used_today(session)
        raise QuotaExceeded(
            f"Daily quota guard hit ({used} used) — refusing "
            f"{'essential ' if essential else ''}call to {endpoint!r}."
        )
