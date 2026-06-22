"""AccuBet FastAPI application.

Exposes three JSON endpoints consumed by the dashboard SPA:

    GET /api/health       — liveness check
    GET /api/scan         — upcoming value opportunities (gate-passers first)
    GET /api/bets         — tracked paper-bet history with optional filters
    GET /api/performance  — overall + per-market performance summary

The static dashboard (index.html) is served from accubet/static/ and is
the default route at GET /.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from accubet.config import get_config
from accubet.market.comparison import scan as scan_value
from accubet.market.consensus import build_all_consensus
from accubet.models.predictor import run_predictions
from accubet.storage.db import init_db, session_scope
from accubet.storage.models import Consensus, Match, TrackedBet
from accubet.tracking.performance import report as perf_report

_STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AccuBet", version="0.1.0", docs_url="/api/docs")

# Serve static files (dashboard) — must come AFTER route definitions.
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _upcoming_match_ids(session) -> list[int]:
    return list(session.execute(
        select(Match.id)
        .join(Consensus, Consensus.match_id == Match.id)
        .where(Match.status.in_(("NS", "TBD", "PST")))
        .distinct()
    ).scalars().all())


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def dashboard() -> FileResponse:
    """Serve the single-page dashboard."""
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": "accubet"}


@app.get("/api/scan")
def scan(
    top: int = Query(50, ge=1, le=200),
    gates_only: bool = Query(False),
) -> JSONResponse:
    """Return upcoming value opportunities ranked by EV."""
    cfg = get_config()
    with session_scope() as session:
        match_ids = _upcoming_match_ids(session)
        if not match_ids:
            return JSONResponse({"opportunities": [], "meta": {"total": 0, "passing": 0}})

        opps = scan_value(session, cfg, match_ids)
        shown = [o for o in opps if o._passes] if gates_only else opps
        shown = shown[:top]

        items = []
        for o in shown:
            items.append({
                "match": f"{o.home} v {o.away}",
                "kickoff": o.kickoff,
                "market": o.market,
                "selection": o.selection,
                "line": o.line,
                "fair_prob": round(o.fair_prob, 4),
                "price": o.price,
                "price_source": o.price_source,
                "ev": round(o.ev, 4),
                "value_pct": round(o.value_pct, 4),
                "confidence": round(o.confidence, 4),
                "n_books": o.n_books,
                "steam_move": o.steam_move,
                "passes": o._passes,
            })

        return JSONResponse({
            "opportunities": items,
            "meta": {
                "total": len(opps),
                "passing": sum(o._passes for o in opps),
            },
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
                "id": b.id,
                "kind": b.kind,
                "market": b.market,
                "selection": b.selection,
                "line": b.line,
                "odds": b.odds,
                "ev": b.ev,
                "stake": b.stake,
                "placed_at": b.placed_at.isoformat() if b.placed_at else None,
                "settled": b.settled,
                "result": b.result,
                "pnl": b.pnl,
                "clv": b.clv,
                "predicted_prob": b.predicted_prob,
            })
        return JSONResponse({"bets": items, "count": len(items)})


@app.get("/api/performance")
def performance() -> JSONResponse:
    """Return overall + per-market performance from settled paper bets."""
    with session_scope() as session:
        rep = perf_report(session)

    def _row(r) -> dict:
        return {
            "label": r.label,
            "total": r.total,
            "settled": r.settled,
            "wins": r.wins,
            "losses": r.losses,
            "voids": r.voids,
            "pending": r.pending,
            "staked": round(r.staked, 2),
            "pnl": round(r.pnl, 2),
            "roi": round(r.roi, 4) if r.roi is not None else None,
            "win_rate": round(r.win_rate, 4) if r.win_rate is not None else None,
        }

    return JSONResponse({
        "overall": _row(rep["overall"]),
        "by_market": [_row(r) for r in rep["by_market"]],
        "by_kind": [_row(r) for r in rep["by_kind"]],
    })
