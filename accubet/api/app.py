"""AccuBet FastAPI application.

Exposes JSON endpoints consumed by the dashboard SPA:

    GET /api/health       — liveness check
    GET /api/scan         — value opportunities, optionally filtered by kickoff date
    GET /api/matches      — all matches for a date with status + results
    GET /api/bets         — tracked paper-bet history with optional filters
    GET /api/performance  — overall + per-market performance summary
"""

from __future__ import annotations

from datetime import date as date_type, datetime
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func as sqlfunc, select

from accubet.config import get_config
from accubet.market.comparison import scan as scan_value
from accubet.market.consensus import build_all_consensus
from accubet.models.predictor import run_predictions
from accubet.storage.db import init_db, session_scope
from accubet.storage.models import Consensus, Match, Result, TrackedBet
from accubet.tracking.performance import report as perf_report

_STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AccuBet", version="0.1.0", docs_url="/api/docs")

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> date_type | None:
    """Parse YYYY-MM-DD string to date; return None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def _match_ids_for_date(session, target_date: date_type | None = None) -> list[int]:
    """Return match IDs that have consensus data.

    If *target_date* is given, return all matches for that calendar day regardless
    of status (so FT matches appear when browsing past dates).
    If no date is given, default to today's matches.
    """
    if target_date is None:
        target_date = date_type.today()

    day_start = datetime(target_date.year, target_date.month, target_date.day)
    day_end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)

    return list(session.execute(
        select(Match.id)
        .join(Consensus, Consensus.match_id == Match.id)
        .where(Match.kickoff >= day_start, Match.kickoff <= day_end)
        .distinct()
    ).scalars().all())


def _load_results(session, match_ids: list[int]) -> dict[int, dict]:
    """Return {match_id: {home_goals, away_goals}} for matches that have results."""
    if not match_ids:
        return {}
    return {
        r.match_id: {"home_goals": r.home_goals, "away_goals": r.away_goals}
        for r in session.execute(
            select(Result).where(Result.match_id.in_(match_ids))
        ).scalars()
    }


def _load_statuses(session, match_ids: list[int]) -> dict[int, str]:
    """Return {match_id: status_string} for a set of match IDs."""
    if not match_ids:
        return {}
    return {
        m.id: m.status
        for m in session.execute(
            select(Match).where(Match.id.in_(match_ids))
        ).scalars()
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": "accubet"}


@app.get("/api/scan")
def scan(
    top: int = Query(50, ge=1, le=200),
    gates_only: bool = Query(False),
    date: str | None = Query(None),
) -> JSONResponse:
    """Return value opportunities ranked by EV for a given kickoff date (default: today)."""
    cfg = get_config()
    target_date = _parse_date(date)

    with session_scope() as session:
        match_ids = _match_ids_for_date(session, target_date)
        if not match_ids:
            return JSONResponse({"opportunities": [], "meta": {"total": 0, "passing": 0}})

        opps = scan_value(session, cfg, match_ids)

        statuses = _load_statuses(session, match_ids)
        results  = _load_results(session, match_ids)

        shown = [o for o in opps if o._passes] if gates_only else opps
        shown = shown[:top]

        items = []
        for o in shown:
            items.append({
                "match_id":     o.match_id,
                "match":        f"{o.home} v {o.away}",
                "kickoff":      o.kickoff,
                "market":       o.market,
                "selection":    o.selection,
                "line":         o.line,
                "fair_prob":    round(o.fair_prob, 4),
                "price":        o.price,
                "price_source": o.price_source,
                "ev":           round(o.ev, 4),
                "value_pct":    round(o.value_pct, 4),
                "confidence":   round(o.confidence, 4),
                "n_books":      o.n_books,
                "steam_move":   o.steam_move,
                "passes":       o._passes,
                "match_status": statuses.get(o.match_id, "NS"),
                "result":       results.get(o.match_id),
            })

        return JSONResponse({
            "opportunities": items,
            "meta": {
                "total":   len(opps),
                "passing": sum(o._passes for o in opps),
            },
        })


@app.get("/api/matches")
def matches_for_date(date: str | None = Query(None)) -> JSONResponse:
    """Return all matches for a kickoff date with status, results, and signal counts."""
    target_date = _parse_date(date) or date_type.today()

    with session_scope() as session:
        day_start = datetime(target_date.year, target_date.month, target_date.day)
        day_end   = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)

        match_list = list(session.execute(
            select(Match)
            .where(Match.kickoff >= day_start, Match.kickoff <= day_end)
            .order_by(Match.kickoff)
        ).scalars())

        if not match_list:
            return JSONResponse({"date": target_date.isoformat(), "matches": [], "total": 0})

        match_ids = [m.id for m in match_list]

        results_map = _load_results(session, match_ids)

        sig_counts: dict[int, int] = {}
        for mid, cnt in session.execute(
            select(Consensus.match_id, sqlfunc.count(Consensus.id))
            .where(Consensus.match_id.in_(match_ids))
            .group_by(Consensus.match_id)
        ):
            sig_counts[mid] = cnt

        items = []
        for m in match_list:
            r = results_map.get(m.id)
            items.append({
                "id":            m.id,
                "home":          m.home_team.name if m.home_team else "?",
                "away":          m.away_team.name if m.away_team else "?",
                "kickoff":       m.kickoff.isoformat() if m.kickoff else None,
                "status":        m.status,
                "result":        r,
                "signals_count": sig_counts.get(m.id, 0),
            })

        return JSONResponse({
            "date":    target_date.isoformat(),
            "matches": items,
            "total":   len(items),
        })


@app.get("/api/bets")
def bets(
    limit: int = Query(100, ge=1, le=500),
    settled_only: bool = Query(False),
    market: str | None = Query(None),
) -> JSONResponse:
    """Return tracked paper bets, most recent first."""
    with session_scope() as session:
        q = select(TrackedBet).order_by(TrackedBet.placed_at.desc()).limit(limit)
        if settled_only:
            q = q.where(TrackedBet.settled == True)   # noqa: E712
        if market:
            q = q.where(TrackedBet.market == market)
        rows = list(session.execute(q).scalars())

        items = []
        for b in rows:
            items.append({
                "id":            b.id,
                "kind":          b.kind,
                "market":        b.market,
                "selection":     b.selection,
                "line":          b.line,
                "odds":          b.odds,
                "ev":            b.ev,
                "stake":         b.stake,
                "placed_at":     b.placed_at.isoformat() if b.placed_at else None,
                "settled":       b.settled,
                "result":        b.result,
                "pnl":           b.pnl,
                "clv":           b.clv,
                "predicted_prob": b.predicted_prob,
            })
        return JSONResponse({"bets": items, "count": len(items)})


@app.get("/api/system")
def system_info() -> JSONResponse:
    """Returns API quota usage and the tail of the pipeline cron log."""
    cfg = get_config()
    from accubet.ingestion.quota import remaining, requests_used_today
    with session_scope() as session:
        used = requests_used_today(session)
        rem  = remaining(session, cfg)

    cron_log_path = Path("/data/cron.log")
    cron_tail = ""
    if cron_log_path.exists():
        try:
            with open(cron_log_path, "r", errors="replace") as f:
                lines = f.readlines()
                cron_tail = "".join(lines[-40:])
        except Exception:
            cron_tail = "(could not read log)"

    return JSONResponse({
        "quota": {
            "used":      used,
            "limit":     cfg.apifootball.daily_request_limit,
            "remaining": rem,
        },
        "cron_log": cron_tail,
    })


@app.get("/api/performance")
def performance() -> JSONResponse:
    """Return overall + per-market performance from settled paper bets."""
    with session_scope() as session:
        rep = perf_report(session)

    def _row(r) -> dict:
        return {
            "label":    r.label,
            "total":    r.total,
            "settled":  r.settled,
            "wins":     r.wins,
            "losses":   r.losses,
            "voids":    r.voids,
            "pending":  r.pending,
            "staked":   round(r.staked, 2),
            "pnl":      round(r.pnl, 2),
            "roi":      round(r.roi, 4) if r.roi is not None else None,
            "win_rate": round(r.win_rate, 4) if r.win_rate is not None else None,
        }

    return JSONResponse({
        "overall":   _row(rep["overall"]),
        "by_market": [_row(r) for r in rep["by_market"]],
        "by_kind":   [_row(r) for r in rep["by_kind"]],
    })
