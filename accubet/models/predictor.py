"""Prediction orchestration: build per-competition models from history, run the ensemble
for matches, and persist ensemble predictions.

Models are trained per competition (a team's strength only means something within its
league). For each target match we blend the market consensus with the Poisson, Glicko-2,
and form models; matches whose competition has no usable history fall back to market only
(graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.config import AppConfig
from accubet.logging_setup import get_logger
from accubet.models import form as form_mod
from accubet.models import goals_poisson as poisson
from accubet.models.ensemble import (
    ensemble, market_to_groups, onex2_to_groups, poisson_to_groups,
)
from accubet.models.ml import fit_ml, predict_ml
from accubet.models.ratings_glicko import fit_ratings, ratings_to_1x2
from accubet.storage.models import Consensus, Match, Prediction, Result

log = get_logger(__name__)


@dataclass
class CompetitionModels:
    strengths: object | None
    ratings: dict
    matches_chrono: list[tuple]
    ml_model: object | None = None

    def internal_groups(self, home_id: int, away_id: int, ou_lines: tuple[float, ...]) -> dict:
        goals = poisson.predict(self.strengths, home_id, away_id, ou_lines) if self.strengths else None
        glicko = ratings_to_1x2(self.ratings, home_id, away_id) if self.ratings else None
        hp = form_mod.points_per_game(self.matches_chrono, home_id)
        ap = form_mod.points_per_game(self.matches_chrono, away_id)
        ml = predict_ml(self.ml_model, home_id, away_id)
        return {
            "goals": poisson_to_groups(goals),
            "glicko": onex2_to_groups(glicko),
            "form": onex2_to_groups(form_mod.form_to_1x2(hp, ap)),
            "ml": onex2_to_groups(ml),
        }


def build_competition_models(session: Session, competition_id: int | None) -> CompetitionModels:
    chrono: list[tuple] = []
    if competition_id is not None:
        rows = session.execute(
            select(Match, Result)
            .join(Result, Result.match_id == Match.id)
            .where(Match.competition_id == competition_id)
            .order_by(Match.kickoff.asc())
        ).all()
        chrono = [
            (m.home_team_id, m.away_team_id, r.home_goals, r.away_goals)
            for m, r in rows
            if None not in (m.home_team_id, m.away_team_id, r.home_goals, r.away_goals)
        ]
    return CompetitionModels(
        strengths=poisson.fit_strengths(chrono),
        ratings=fit_ratings(chrono),
        matches_chrono=chrono,
        ml_model=fit_ml(chrono),
    )


def _weights(cfg: AppConfig) -> dict[str, float]:
    mw = cfg.model_weights
    return {"market": mw.market, "goals": mw.goals, "glicko": mw.glicko, "form": mw.form, "ml": mw.ml}


def predict_match(session: Session, cfg: AppConfig, match: Match, comp: CompetitionModels,
                  ou_lines: tuple[float, ...]) -> int:
    consensus = session.execute(
        select(Consensus).where(Consensus.match_id == match.id)
    ).scalars().all()
    if not consensus:
        return 0

    models_by_name = {"market": market_to_groups(consensus)}
    if match.home_team_id and match.away_team_id:
        models_by_name.update(comp.internal_groups(match.home_team_id, match.away_team_id, ou_lines))

    preds = ensemble(models_by_name, _weights(cfg))

    for old in session.execute(
        select(Prediction).where(Prediction.match_id == match.id, Prediction.model == "ensemble")
    ).scalars().all():
        session.delete(old)
    session.flush()

    n = 0
    for (market, line), gp in preds.items():
        # Only persist where an internal model actually contributed (n_models > 1).
        # Market-only groups are left to the market fallback in the value engine, so
        # data-sparse matches behave exactly as the pure-market Phase 1 did.
        if gp.n_models <= 1:
            continue
        for sel, p in gp.dist.items():
            session.add(Prediction(
                match_id=match.id, market=market, selection=sel, line=line,
                model="ensemble", prob=p, confidence=gp.confidence,
            ))
            n += 1
    session.flush()
    return n


def run_predictions(session: Session, cfg: AppConfig, match_ids: list[int]) -> int:
    ou_lines = tuple(cfg.scan.ou_lines)
    matches = session.execute(select(Match).where(Match.id.in_(match_ids))).scalars().all()
    comp_cache: dict[int | None, CompetitionModels] = {}
    total = 0
    for m in matches:
        if m.competition_id not in comp_cache:
            comp_cache[m.competition_id] = build_competition_models(session, m.competition_id)
        total += predict_match(session, cfg, m, comp_cache[m.competition_id], ou_lines)
    return total
