"""Integration test: history -> ensemble predictions -> value engine uses them."""

import itertools
from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from accubet.config import get_config
from accubet.market.comparison import compare_match
from accubet.market.consensus import build_consensus
from accubet.models.predictor import run_predictions
from accubet.storage.models import (
    Base, Competition, Match, OddsSnapshot, Prediction, Result, Team,
)

_LVL = {1: 2.4, 2: 1.8, 3: 1.3, 4: 0.9}  # team 1 strongest


def test_predict_then_compare_uses_ensemble():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cfg = get_config()
    with Session(engine) as s:
        comp = Competition(api_league_id=39, name="EPL", scope="global")
        s.add(comp)
        s.flush()
        team = {}
        for i in range(1, 5):
            t = Team(api_team_id=i, name=f"T{i}")
            s.add(t)
            s.flush()
            team[i] = t.id

        # seed a finished history (team 1 strong, team 4 weak)
        fid = 1000
        for _ in range(4):
            for h, a in itertools.permutations([1, 2, 3, 4], 2):
                fid += 1
                m = Match(api_fixture_id=fid, competition_id=comp.id, home_team_id=team[h],
                          away_team_id=team[a], status="FT", kickoff=datetime(2024, 1, 1))
                s.add(m)
                s.flush()
                s.add(Result(match_id=m.id, home_goals=max(0, round(_LVL[h])),
                             away_goals=max(0, round(_LVL[a] * 0.7))))
        s.flush()

        # an upcoming match: strong T1 (home) vs weak T4 (away), with a market consensus
        up = Match(api_fixture_id=9999, competition_id=comp.id, home_team_id=team[1],
                   away_team_id=team[4], status="NS", kickoff=datetime(2026, 8, 1))
        s.add(up)
        s.flush()
        market = {
            ("match_winner", "home", None): 1.5, ("match_winner", "draw", None): 4.5,
            ("match_winner", "away", None): 7.0, ("btts", "yes", None): 2.0,
            ("btts", "no", None): 1.7, ("over_under", "over", 2.5): 1.8,
            ("over_under", "under", 2.5): 2.0,
        }
        for b in range(7):
            for (mk, sel, ln), price in market.items():
                s.add(OddsSnapshot(match_id=up.id, source="apifootball", bookmaker=f"B{b}",
                                   market=mk, selection=sel, line=ln, price=price))
        s.flush()
        build_consensus(s, up.id, min_books=1)

        # run the ensemble
        assert run_predictions(s, cfg, [up.id]) > 0
        preds = {
            (p.market, p.selection, p.line): p
            for p in s.execute(select(Prediction).where(Prediction.model == "ensemble")).scalars()
        }
        assert ("match_winner", "home", None) in preds
        # models strongly favour T1 at home -> ensemble home prob exceeds the ~0.62 market fair
        assert preds[("match_winner", "home", None)].prob > 0.62

        # the value engine must now read the ensemble (not raw consensus)
        opps = compare_match(s, cfg, up)
        assert any(o.prob_source == "ensemble" for o in opps)
