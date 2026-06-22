"""Gradient-boosted classifier for the ML slot of the ensemble (weight ~10%).

Trains an XGBoostClassifier on historical match data using temporally-correct
rolling features — each training row sees only matches that occurred *before* it,
eliminating lookahead bias.  Returns ``None`` if xgboost is not installed (it lives
in the ``[ml]`` optional extra) or if there is insufficient training data, so the
ensemble degrades gracefully to the remaining three models.

Interface mirrors the other models in this package:
  fit_ml(matches_chrono)          → MLModel | None
  predict_ml(model, home, away)   → {"home": p, "draw": p, "away": p} | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:  # pragma: no cover
    _XGB = False

_MIN_SAMPLES = 80   # minimum training rows before we trust the classifier
_N_FORM = 6         # rolling window size for pre-match features

FEATURE_NAMES = [
    "home_ppg",           # home team points-per-game (last _N_FORM matches)
    "away_ppg",
    "ppg_diff",           # home_ppg - away_ppg
    "home_gspg",          # home team goals scored per game
    "home_gcpg",          # home team goals conceded per game
    "away_gspg",
    "away_gcpg",
    "home_attack_edge",   # home_gspg - away_gcpg  (how much home attack > away defence)
    "away_attack_edge",   # away_gspg - home_gcpg
]

# Outcome encoding: 0 = home win, 1 = draw, 2 = away win
_HOME, _DRAW, _AWAY = 0, 1, 2


@dataclass
class MLModel:
    clf: Any                          # fitted XGBClassifier
    all_matches: list[tuple]          # full history, used for prediction-time features
    team_ids: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _team_stats(history: list[tuple], team_id: int, n: int = _N_FORM) -> dict | None:
    """Rolling stats for *team_id* across the most recent *n* matches in *history*."""
    pts: list[int] = []
    gs: list[int] = []   # goals scored
    gc: list[int] = []   # goals conceded
    for h, a, hg, ag in reversed(history):
        if h == team_id:
            pts.append(3 if hg > ag else 1 if hg == ag else 0)
            gs.append(hg)
            gc.append(ag)
        elif a == team_id:
            pts.append(3 if ag > hg else 1 if hg == ag else 0)
            gs.append(ag)
            gc.append(hg)
        if len(pts) >= n:
            break
    if not pts:
        return None
    return {
        "ppg": sum(pts) / len(pts),
        "gspg": sum(gs) / len(gs),
        "gcpg": sum(gc) / len(gc),
    }


def _features(history: list[tuple], home_id: int, away_id: int) -> list[float] | None:
    """Build the 9-feature vector for a (home, away) fixture from *history*."""
    hs = _team_stats(history, home_id)
    aw = _team_stats(history, away_id)
    if hs is None or aw is None:
        return None
    return [
        hs["ppg"],
        aw["ppg"],
        hs["ppg"] - aw["ppg"],
        hs["gspg"],
        hs["gcpg"],
        aw["gspg"],
        aw["gcpg"],
        hs["gspg"] - aw["gcpg"],   # home attack advantage
        aw["gspg"] - hs["gcpg"],   # away attack advantage
    ]


def _outcome(hg: int, ag: int) -> int:
    return _HOME if hg > ag else _DRAW if hg == ag else _AWAY


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_ml(matches_chrono: list[tuple]) -> MLModel | None:
    """Train XGBoost on historical matches using temporally-correct rolling features.

    Returns ``None`` if xgboost is unavailable or there is too little data.
    """
    if not _XGB:
        return None  # pragma: no cover

    X: list[list[float]] = []
    y: list[int] = []
    for i in range(1, len(matches_chrono)):
        h, a, hg, ag = matches_chrono[i]
        if None in (h, a, hg, ag):
            continue
        feats = _features(matches_chrono[:i], h, a)
        if feats is None:
            continue
        X.append(feats)
        y.append(_outcome(hg, ag))

    if len(X) < _MIN_SAMPLES:
        return None

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X, y)

    team_ids = {row[0] for row in matches_chrono} | {row[1] for row in matches_chrono}
    return MLModel(clf=clf, all_matches=list(matches_chrono), team_ids=team_ids)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_ml(model: MLModel | None, home_id: int, away_id: int) -> dict[str, float] | None:
    """Return {home, draw, away} probability dict, or None when no model / no features."""
    if model is None:
        return None
    if home_id not in model.team_ids or away_id not in model.team_ids:
        return None
    feats = _features(model.all_matches, home_id, away_id)
    if feats is None:
        return None
    probs = model.clf.predict_proba([feats])[0]
    # classes_ is sorted; map back in case not all three outcomes appear in training
    labels = list(model.clf.classes_)
    def _p(cls: int) -> float:
        return float(probs[labels.index(cls)]) if cls in labels else 0.0
    return {"home": _p(_HOME), "draw": _p(_DRAW), "away": _p(_AWAY)}
