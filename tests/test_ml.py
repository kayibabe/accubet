"""Tests for the ML model module (xgboost-backed, graceful fallback when absent)."""

import pytest

from accubet.models.ml import (
    FEATURE_NAMES,
    MLModel,
    _features,
    _outcome,
    _team_stats,
    fit_ml,
    predict_ml,
)

xgb = pytest.importorskip("xgboost", reason="xgboost not installed ([ml] extra required)")


# ---------------------------------------------------------------------------
# Pure-Python helpers — no xgboost dep
# ---------------------------------------------------------------------------

def _history(n: int, home_id: int = 1, away_id: int = 2) -> list[tuple]:
    """Alternating wins for each team, n matches total."""
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append((home_id, away_id, 2, 0))
        elif i % 3 == 1:
            rows.append((away_id, home_id, 0, 1))
        else:
            rows.append((home_id, away_id, 1, 1))
    return rows


def test_outcome_encoding():
    assert _outcome(2, 0) == 0   # home win
    assert _outcome(1, 1) == 1   # draw
    assert _outcome(0, 2) == 2   # away win


def test_team_stats_returns_none_unknown_team():
    history = [(1, 2, 2, 0), (1, 2, 1, 1)]
    assert _team_stats(history, 99) is None


def test_team_stats_basic():
    history = [
        (1, 2, 2, 0),  # team 1 wins at home (3pts, gs=2, gc=0)
        (2, 1, 0, 1),  # team 1 wins away (3pts, gs=1, gc=0)
        (1, 2, 1, 1),  # draw (1pt, gs=1, gc=1)
    ]
    s = _team_stats(history, 1)
    assert s is not None
    assert s["ppg"] == pytest.approx((3 + 3 + 1) / 3)
    assert s["gspg"] == pytest.approx((2 + 1 + 1) / 3)
    assert s["gcpg"] == pytest.approx((0 + 0 + 1) / 3)


def test_team_stats_respects_window():
    history = _history(20, home_id=1, away_id=2)
    s = _team_stats(history, 1, n=3)
    assert s is not None
    # Only the last 3 appearances of team 1 should be counted.
    assert s["ppg"] is not None


def test_features_returns_9_values():
    history = _history(10)
    f = _features(history, 1, 2)
    assert f is not None
    assert len(f) == len(FEATURE_NAMES) == 9


def test_features_returns_none_for_unknown_team():
    history = _history(10, home_id=1, away_id=2)
    assert _features(history, 1, 99) is None
    assert _features(history, 99, 2) is None


def test_predict_ml_none_model():
    assert predict_ml(None, 1, 2) is None


def test_predict_ml_unknown_team():
    history = _history(100)
    model = fit_ml(history)
    assert model is not None
    assert predict_ml(model, 1, 999) is None


# ---------------------------------------------------------------------------
# fit_ml — requires xgboost
# ---------------------------------------------------------------------------

def test_fit_ml_insufficient_data_returns_none():
    history = _history(10)  # far fewer than _MIN_SAMPLES training rows
    assert fit_ml(history) is None


def _large_history(n_matches: int = 200) -> list[tuple]:
    """Varied results across 10 teams so features are informative."""
    import random
    rng = random.Random(0)
    teams = list(range(1, 11))
    rows = []
    for _ in range(n_matches):
        h, a = rng.sample(teams, 2)
        hg = rng.randint(0, 4)
        ag = rng.randint(0, 3)
        rows.append((h, a, hg, ag))
    return rows


def test_fit_ml_returns_model():
    history = _large_history(250)
    model = fit_ml(history)
    assert model is not None
    assert isinstance(model, MLModel)
    assert model.clf is not None
    assert len(model.all_matches) == 250


def test_predict_ml_probabilities_sum_to_one():
    history = _large_history(250)
    model = fit_ml(history)
    assert model is not None
    dist = predict_ml(model, 1, 2)
    assert dist is not None
    assert set(dist.keys()) == {"home", "draw", "away"}
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= v <= 1.0 for v in dist.values())


def test_predict_ml_reasonable_home_advantage():
    """Team 1 always wins at home in training → model should favour home."""
    rows = [(1, 2, 3, 0)] * 200   # team 1 always wins 3-0 at home
    model = fit_ml(rows)
    if model is None:
        pytest.skip("not enough varied training data")
    dist = predict_ml(model, 1, 2)
    assert dist is not None
    assert dist["home"] > dist["away"]
