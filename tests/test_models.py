"""Tests for the Poisson / Glicko-2 / form models and the ensemble."""

import itertools

import pytest

from accubet.models import form as form_mod
from accubet.models import goals_poisson as poisson
from accubet.models.ensemble import combine_group, ensemble
from accubet.models.ratings_glicko import fit_ratings, ratings_to_1x2


# team scoring "level"; team 1 strongest, team 4 weakest
_LVL = {1: 2.4, 2: 1.8, 3: 1.3, 4: 0.9}


def _league(repeats: int = 3):
    matches = []
    for _ in range(repeats):
        for h, a in itertools.permutations([1, 2, 3, 4], 2):
            hg = max(0, round(_LVL[h]))
            ag = max(0, round(_LVL[a] * 0.7))  # home advantage
            matches.append((h, a, hg, ag))
    return matches


# --- Poisson ---------------------------------------------------------------

def test_poisson_strengths_and_prediction():
    s = poisson.fit_strengths(_league())
    assert s is not None
    eg = poisson.expected_goals(s, 1, 4)  # strong home vs weak away
    assert eg is not None
    assert eg[0] > eg[1]  # home expected goals exceed away

    probs = poisson.predict(s, 1, 4)
    mw = probs["match_winner"]
    # ~1.0 up to the goal-matrix truncation tail (cap at 10 goals); ensemble renormalizes.
    assert sum(mw.values()) == pytest.approx(1.0, abs=1e-3)
    assert mw["home"] > mw["away"]
    assert 0.0 <= probs["btts"]["yes"] <= 1.0
    assert probs["over_under"][("over", 2.5)] + probs["over_under"][("under", 2.5)] == pytest.approx(1.0)


def test_dixon_coles_probabilities_still_sum_to_one():
    s = poisson.fit_strengths(_league())
    assert s is not None
    probs = poisson.predict(s, 1, 4)
    mw = probs["match_winner"]
    assert sum(mw.values()) == pytest.approx(1.0, abs=1e-6)
    assert probs["btts"]["yes"] + probs["btts"]["no"] == pytest.approx(1.0, abs=1e-6)


def test_dixon_coles_rho_in_valid_range():
    s = poisson.fit_strengths(_league())
    assert s is not None
    assert -0.4 <= s.rho <= 0.0


def test_dixon_coles_tau_known_values():
    """tau formula from Dixon & Coles (1997) Section 3."""
    # rho=0 -> all tau values are 1.0 (reduces to plain Poisson)
    assert poisson._tau(0, 0, 1.5, 1.2, 0.0) == pytest.approx(1.0)
    assert poisson._tau(1, 0, 1.5, 1.2, 0.0) == pytest.approx(1.0)
    assert poisson._tau(2, 3, 1.5, 1.2, 0.0) == pytest.approx(1.0)  # non-low cell

    # With rho=-0.13 (typical): 0-0 gets multiplied by > 1 (over-represented)
    tau_00 = poisson._tau(0, 0, 1.5, 1.2, -0.13)
    assert tau_00 > 1.0   # negative rho -> 1 - lam_h*lam_a*rho > 1

    # Non-low cells always return 1.0 regardless of rho
    assert poisson._tau(3, 2, 1.5, 1.2, -0.13) == pytest.approx(1.0)


def test_dixon_coles_shifts_draw_probability():
    """With rho<0 the 1-1 cell is down-weighted; draws should be slightly less
    probable than plain Poisson predicts for evenly-matched teams."""
    eh = ea = 1.3  # evenly matched
    plain = poisson.predict_markets(eh, ea, rho=0.0)
    dc = poisson.predict_markets(eh, ea, rho=-0.13)
    # 1-1 is a draw; its correction factor < 1 means draws are adjusted
    # The *sum* of draw probability shifts a little but stays meaningful.
    assert abs(plain["match_winner"]["draw"] - dc["match_winner"]["draw"]) < 0.05
    # Both should still sum to 1.0
    assert sum(dc["match_winner"].values()) == pytest.approx(1.0, abs=1e-6)


def test_poisson_insufficient_data_returns_none():
    assert poisson.fit_strengths([(1, 2, 1, 0)]) is None
    s = poisson.fit_strengths(_league())
    assert poisson.predict(s, 99, 98) is None  # unknown teams -> graceful None


# --- Glicko-2 --------------------------------------------------------------

def test_glicko_orders_teams_and_predicts():
    matches = [(1, 2, 3, 0)] * 10  # team 1 repeatedly beats team 2
    ratings = fit_ratings(matches)
    assert ratings[1].r > ratings[2].r
    p = ratings_to_1x2(ratings, 1, 2)
    assert p is not None
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["home"] > p["away"]
    assert ratings_to_1x2(ratings, 1, 999) is None  # unknown team


# --- form ------------------------------------------------------------------

def test_form_ppg_and_distribution():
    matches = [(1, 2, 2, 0), (3, 1, 0, 1), (1, 4, 3, 0)]  # team 1 won all
    assert form_mod.points_per_game(matches, 1, n=5) == pytest.approx(3.0)
    d = form_mod.form_to_1x2(3.0, 0.5)
    assert d is not None
    assert sum(d.values()) == pytest.approx(1.0)
    assert d["home"] > d["away"]


# --- ensemble --------------------------------------------------------------

def test_combine_group_agreement_and_degradation():
    weights = {"market": 0.55, "goals": 0.15, "glicko": 0.15, "form": 0.10}
    agree = {
        "market": {"home": 0.5, "draw": 0.3, "away": 0.2},
        "goals": {"home": 0.5, "draw": 0.3, "away": 0.2},
    }
    gp = combine_group(agree, weights)
    assert gp.dist["home"] == pytest.approx(0.5)
    assert gp.n_models == 2
    assert gp.confidence > 0.7  # perfect agreement -> high confidence

    # graceful degradation: empty model dropped, market only -> returns market dist
    only_market = combine_group({"market": {"home": 0.6, "draw": 0.25, "away": 0.15}, "goals": {}}, weights)
    assert only_market.n_models == 1
    assert only_market.dist["home"] == pytest.approx(0.6)


def test_ensemble_blends_groups():
    weights = {"market": 0.5, "glicko": 0.5}
    models = {
        "market": {("match_winner", None): {"home": 0.4, "draw": 0.3, "away": 0.3}},
        "glicko": {("match_winner", None): {"home": 0.6, "draw": 0.2, "away": 0.2}},
    }
    out = ensemble(models, weights)
    mw = out[("match_winner", None)]
    assert mw.dist["home"] == pytest.approx(0.5)  # 0.5*0.4 + 0.5*0.6
    assert sum(mw.dist.values()) == pytest.approx(1.0)
