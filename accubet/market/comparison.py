"""Betway-vs-fair comparison — the Phase 1 value signal.

For each selection we have a fair (vig-removed) consensus probability from world books.
We compare it against the price we could actually take:
* **Betway Malawi** when we have its price (the book you'd really bet at); otherwise
* the **best available** world-book price (so global-league value still surfaces).

A value opportunity is where the takeable price implies a *lower* probability than our
fair estimate — i.e. positive EV. Until the internal models come online, fair consensus
*is* our true probability (market weight = 100%, graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.config import AppConfig
from accubet.market.efficiency import inefficiency_score, market_confidence
from accubet.storage.models import Consensus, Match, OddsSnapshot, Prediction, ValueBet
from accubet.value.ev import expected_value, implied_probability, value_pct


@dataclass
class ValueOpportunity:
    match_id: int
    home: str
    away: str
    kickoff: str
    market: str
    selection: str
    line: float | None
    fair_prob: float
    price: float            # the price we'd take
    price_source: str       # "betway" | "best"
    betway_odds: float | None
    best_odds: float | None
    ev: float
    value_pct: float
    confidence: float       # 0-1
    n_books: int
    prob_source: str = "market"   # "ensemble" | "market"

    @property
    def passes(self) -> bool:  # filled against gates by the caller
        return self._passes

    _passes: bool = False


def _betway_prices(session: Session, match_id: int) -> dict[tuple, float]:
    rows = session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match_id, OddsSnapshot.source == "betway")
        .order_by(OddsSnapshot.captured_at.asc())
    ).scalars().all()
    prices: dict[tuple, float] = {}
    for r in rows:
        prices[(r.market, r.selection, r.line)] = r.price  # latest wins (asc order)
    return prices


def _is_liquid(cfg: AppConfig, market: str, line: float | None) -> bool:
    """Whether a (market, line) is in the liquid set the scan considers by default."""
    if market not in cfg.scan.markets:
        return False
    if market == "over_under" and line not in cfg.scan.ou_lines:
        return False
    return True


def compare_match(
    session: Session, cfg: AppConfig, match: Match, *, restrict: bool = True
) -> list[ValueOpportunity]:
    consensus_rows = session.execute(
        select(Consensus).where(Consensus.match_id == match.id)
    ).scalars().all()
    if not consensus_rows:
        return []

    betway = _betway_prices(session, match.id)
    preds = {
        (p.market, p.selection, p.line): p
        for p in session.execute(
            select(Prediction).where(
                Prediction.match_id == match.id, Prediction.model == "ensemble"
            )
        ).scalars().all()
    }
    home = match.home_team.name if match.home_team else "?"
    away = match.away_team.name if match.away_team else "?"
    kickoff = match.kickoff.isoformat() if match.kickoff else ""

    opps: list[ValueOpportunity] = []
    for c in consensus_rows:
        if restrict and not _is_liquid(cfg, c.market, c.line):
            continue
        betway_odds = betway.get((c.market, c.selection, c.line))
        price = betway_odds if betway_odds else c.best_odds
        if not price or price <= 1.0:
            continue

        # true probability = ensemble (market + models) when available, else market consensus.
        pred = preds.get((c.market, c.selection, c.line))
        if pred is not None:
            true_prob = pred.prob
            conf = pred.confidence if pred.confidence is not None else 0.6
            prob_source = "ensemble"
        else:
            true_prob = c.fair_prob
            conf = market_confidence(c.n_books, c.overround) / 100.0
            prob_source = "market"

        opp = ValueOpportunity(
            match_id=match.id, home=home, away=away, kickoff=kickoff,
            market=c.market, selection=c.selection, line=c.line,
            fair_prob=true_prob, price=price,
            price_source="betway" if betway_odds else "best",
            betway_odds=betway_odds, best_odds=c.best_odds,
            ev=expected_value(true_prob, price),
            value_pct=value_pct(true_prob, price),
            confidence=conf, n_books=c.n_books, prob_source=prob_source,
        )
        # Value gate: a genuine flag needs an INDEPENDENT price (Betway) beating our true
        # probability, sufficient EV, a trustworthy consensus, and enough model/market
        # confidence. "best"-sourced rows are shown for info but never pass the gate
        # (best-of-N vs consensus-of-N just measures the market's own spread).
        opp._passes = (
            opp.price_source == "betway"
            and opp.ev > cfg.value.min_ev
            and c.n_books >= cfg.value.min_books_for_consensus
            and conf >= cfg.value.min_confidence
        )
        opps.append(opp)
    return opps


def scan(
    session: Session, cfg: AppConfig, match_ids: list[int], *, restrict: bool = True
) -> list[ValueOpportunity]:
    matches = session.execute(
        select(Match).where(Match.id.in_(match_ids))
    ).scalars().all()
    opps: list[ValueOpportunity] = []
    for m in matches:
        opps.extend(compare_match(session, cfg, m, restrict=restrict))
    opps.sort(key=lambda o: o.ev, reverse=True)
    return opps


def persist_value_bets(session: Session, opps: list[ValueOpportunity]) -> int:
    """Persist gate-passing opportunities as ValueBet rows. Returns count stored."""
    stored = 0
    for o in opps:
        if not o._passes:
            continue
        session.add(
            ValueBet(
                match_id=o.match_id,
                market=o.market,
                selection=o.selection,
                line=o.line,
                book=o.price_source,
                odds=o.price,
                true_prob=o.fair_prob,
                implied_prob=implied_probability(o.price),
                ev=o.ev,
                value_pct=o.value_pct,
                confidence=o.confidence,
                passed_gates=True,
            )
        )
        stored += 1
    session.flush()
    return stored
