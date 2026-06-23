"""Auto-tracking of PAPER bets and their settlement.

Each scan logs the top-N EV singles and both accumulator tiers as paper bets (flat notional
stake, no real money). Once the matches finish, :func:`settle` grades them against stored
results so the performance module can report how the system is actually doing.
"""

from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.config import AppConfig
from accubet.logging_setup import get_logger
from accubet.market.clv import clv as _clv
from accubet.storage.models import Accumulator, AccumulatorLeg, Match, OddsSnapshot, Result, TrackedBet
from accubet.value.accumulator import AccaTicket

log = get_logger(__name__)


# --- grading ---------------------------------------------------------------

def settle_selection(market: str, selection: str, line: float | None,
                     home_goals: int | None, away_goals: int | None) -> str:
    """Grade one selection given a final score → 'win' | 'loss' | 'void' | 'pending'."""
    if home_goals is None or away_goals is None:
        return "pending"
    hg, ag = home_goals, away_goals
    if market == "match_winner":
        res = "home" if hg > ag else "away" if ag > hg else "draw"
        return "win" if selection == res else "loss"
    if market == "btts":
        yes = hg > 0 and ag > 0
        return "win" if (selection == "yes") == yes else "loss"
    if market == "over_under":
        if line is None:
            return "void"
        total = hg + ag
        if total == line:
            return "void"  # push on integer lines
        return "win" if (selection == "over") == (total > line) else "loss"
    if market == "draw_no_bet":
        if hg == ag:
            return "void"
        res = "home" if hg > ag else "away"
        return "win" if selection == res else "loss"
    if market == "double_chance":
        res = "home" if hg > ag else "away" if ag > hg else "draw"
        cover = {"1x": {"home", "draw"}, "12": {"home", "away"}, "x2": {"draw", "away"}}
        return "win" if res in cover.get(selection, set()) else "loss"
    return "void"


def pnl_for(result: str, stake: float, odds: float) -> float:
    if result == "win":
        return round(stake * (odds - 1.0), 2)
    if result == "loss":
        return -stake
    return 0.0  # void / push


# --- logging ---------------------------------------------------------------

def log_singles(session: Session, cfg: AppConfig, opps: list) -> int:
    """Log the highest-probability gate-passing signal per match as a flat-stake paper bet."""
    # From gate-passing signals, keep only the highest-probability signal per match
    best: dict[int, object] = {}
    for o in opps:
        if not o._passes:
            continue
        if o.match_id not in best or o.fair_prob > best[o.match_id].fair_prob:
            best[o.match_id] = o

    stored = 0
    for o in sorted(best.values(), key=lambda o: o.fair_prob, reverse=True):
        dup = session.execute(
            select(TrackedBet.id).where(
                TrackedBet.kind == "single",
                TrackedBet.match_id == o.match_id,
                TrackedBet.market == o.market,
                TrackedBet.selection == o.selection,
                TrackedBet.line == o.line,
            )
        ).first()
        if dup:
            continue
        session.add(TrackedBet(
            kind="single", match_id=o.match_id, market=o.market, selection=o.selection,
            line=o.line, book=o.price_source, odds=o.price, ev=o.ev,
            confidence=o.confidence, stake=cfg.tracking.paper_stake,
            predicted_prob=o.fair_prob,
        ))
        stored += 1
    session.flush()
    return stored


def _signature(legs) -> str:
    return "|".join(sorted(f"{leg.match_id}:{leg.market}:{leg.selection}:{leg.line}" for leg in legs))


def _acca_exists(session: Session, tier: str, signature: str) -> bool:
    for acc in session.execute(select(Accumulator).where(Accumulator.tier == tier)).scalars():
        if _signature(acc.legs) == signature:
            return True
    return False


def log_accumulators(session: Session, cfg: AppConfig, tickets: dict[str, AccaTicket | None]) -> int:
    stored = 0
    for ticket in tickets.values():
        if ticket is None:
            continue
        if _acca_exists(session, ticket.tier, _signature(ticket.legs)):
            continue
        acc = Accumulator(
            tier=ticket.tier, mode=ticket.mode, combined_odds=ticket.combined_odds,
            combined_prob=ticket.combined_prob, expected_return=ticket.ev,
            risk_rating=ticket.risk_rating,
        )
        session.add(acc)
        session.flush()
        for leg in ticket.legs:
            session.add(AccumulatorLeg(
                accumulator_id=acc.id, match_id=leg.match_id, market=leg.market,
                selection=leg.selection, line=leg.line, odds=leg.odds, prob=leg.prob,
            ))
        session.add(TrackedBet(
            kind="accumulator", ref_id=acc.id, market=f"acca:{ticket.tier}",
            odds=ticket.combined_odds, ev=ticket.ev, stake=cfg.tracking.paper_stake,
        ))
        stored += 1
    session.flush()
    return stored


# --- closing-line value helpers --------------------------------------------

def _closing_price(
    session: Session,
    match_id: int,
    market: str,
    selection: str,
    line: float | None,
    kickoff,
) -> float | None:
    """Last world-book price snapshot at or before kickoff (closing-line proxy).

    Falls back to the most recent snapshot if none exist before kickoff (e.g. the
    free-tier API only refreshed odds once, after the match started).
    """
    base = (
        select(OddsSnapshot.price)
        .where(
            OddsSnapshot.match_id == match_id,
            OddsSnapshot.source == "apifootball",
            OddsSnapshot.market == market,
            OddsSnapshot.selection == selection,
            OddsSnapshot.line == line,
        )
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(1)
    )
    if kickoff is not None:
        price = session.execute(
            base.where(OddsSnapshot.captured_at <= kickoff)
        ).scalar_one_or_none()
        if price is not None:
            return price
    return session.execute(base).scalar_one_or_none()


# --- settlement ------------------------------------------------------------

def settle(session: Session, cfg: AppConfig) -> dict[str, int]:
    """Grade all unsettled tracked bets that now have results, and populate CLV."""
    results = {r.match_id: r for r in session.execute(select(Result)).scalars()}
    pending = list(session.execute(
        select(TrackedBet).where(TrackedBet.settled == False)  # noqa: E712
    ).scalars())

    # pre-load kickoff times so we can look up the pre-kickoff closing price
    single_match_ids = {tb.match_id for tb in pending if tb.kind == "single" and tb.match_id}
    matches: dict[int, Match] = {}
    if single_match_ids:
        matches = {
            m.id: m for m in session.execute(
                select(Match).where(Match.id.in_(single_match_ids))
            ).scalars()
        }

    settled = 0
    for tb in pending:
        if tb.kind == "single":
            r = results.get(tb.match_id)
            if r is None:
                continue
            outcome = settle_selection(tb.market, tb.selection, tb.line, r.home_goals, r.away_goals)
            if outcome == "pending":
                continue
            tb.result = outcome
            tb.pnl = pnl_for(outcome, tb.stake, tb.odds)
            tb.settled = True
            # closing line value — requires at least one world-book snapshot
            m = matches.get(tb.match_id)
            closing = _closing_price(
                session, tb.match_id, tb.market, tb.selection, tb.line,
                m.kickoff if m else None,
            )
            if closing is not None and closing > 1.0:
                tb.closing_odds = closing
                tb.clv = _clv(tb.odds, closing)
            settled += 1
        elif tb.kind == "accumulator":
            acc = session.get(Accumulator, tb.ref_id)
            if acc is None:
                continue
            leg_results = []
            for leg in acc.legs:
                r = results.get(leg.match_id)
                if r is None:
                    leg_results.append("pending")
                    break
                leg_results.append(
                    settle_selection(leg.market, leg.selection, leg.line, r.home_goals, r.away_goals)
                )
            if "pending" in leg_results:
                continue
            if "loss" in leg_results:
                tb.result, tb.pnl = "loss", -tb.stake
            else:
                eff = math.prod(
                    leg.odds for leg, res in zip(acc.legs, leg_results) if res == "win"
                )
                tb.result, tb.pnl = "win", round(tb.stake * (eff - 1.0), 2)
            tb.settled = True
            settled += 1
    session.flush()
    return {"settled": settled}
