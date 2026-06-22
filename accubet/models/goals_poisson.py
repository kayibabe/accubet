"""Poisson goals model with Dixon-Coles low-score correction.

A ratio-based (no-optimizer) Poisson estimates each team's home/away attack and defence
strength relative to the league, derives expected goals, then builds the score matrix to
read off 1X2 / Over-Under / BTTS probabilities.

The Dixon-Coles (1997) correction multiplies the cells for scorelines (0-0), (1-0),
(0-1), (1-1) by a factor tau(i, j, lam_h, lam_a, rho) that accounts for the empirically
observed over-frequency of low-scoring draws and under-frequency of 1-goal home wins
relative to the independent Poisson assumption. rho is estimated from the training data
as the log-linear correction that best fits the low-score frequency distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_MAX_GOALS = 10
_MIN_GAMES = 3   # per team, per home/away split, before we trust its strength
_RHO_DEFAULT = -0.13   # typical negative correlation for low-scoring cells


@dataclass
class Strengths:
    avg_home: float
    avg_away: float
    rho: float = _RHO_DEFAULT   # Dixon-Coles low-score correlation parameter
    home_attack: dict[int, float] = field(default_factory=dict)
    home_defense: dict[int, float] = field(default_factory=dict)
    away_attack: dict[int, float] = field(default_factory=dict)
    away_defense: dict[int, float] = field(default_factory=dict)

    def has(self, home_id: int, away_id: int) -> bool:
        return (
            home_id in self.home_attack and home_id in self.home_defense
            and away_id in self.away_attack and away_id in self.away_defense
        )


def _tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles correction factor for the (i, j) score cell.

    Only the four low-scoring cells are adjusted; all others return 1.0.
    """
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _estimate_rho(matches: list[tuple], avg_home: float, avg_away: float) -> float:
    """Estimate rho from the empirical vs Poisson frequency of the four low-score cells.

    Uses a simple closed-form moment estimator rather than numerical optimisation so
    that no scipy dependency is introduced in the base install.
    """
    n = len(matches)
    if n < 30:
        return _RHO_DEFAULT

    c00 = sum(1 for _, _, hg, ag in matches if hg == 0 and ag == 0) / n
    c10 = sum(1 for _, _, hg, ag in matches if hg == 1 and ag == 0) / n
    c01 = sum(1 for _, _, hg, ag in matches if hg == 0 and ag == 1) / n
    c11 = sum(1 for _, _, hg, ag in matches if hg == 1 and ag == 1) / n

    p00 = math.exp(-(avg_home + avg_away))
    p10 = avg_home * math.exp(-(avg_home + avg_away))
    p01 = avg_away * math.exp(-(avg_home + avg_away))
    p11 = avg_home * avg_away * math.exp(-(avg_home + avg_away))

    # Each cell gives an estimate of rho; average them for stability.
    estimates = []
    denom00 = p00 * avg_home * avg_away
    if denom00 > 1e-9:
        estimates.append((c00 - p00) / denom00)
    if p01 > 1e-9:
        estimates.append((p01 - c01) / (p01 * avg_home))
    if p10 > 1e-9:
        estimates.append((p10 - c10) / (p10 * avg_away))
    denom11 = p11
    if denom11 > 1e-9:
        estimates.append((p11 - c11) / denom11)

    if not estimates:
        return _RHO_DEFAULT
    rho = sum(estimates) / len(estimates)
    # Clamp to a range where tau stays positive for typical lambda values.
    return max(-0.4, min(0.0, rho))


def fit_strengths(matches: list[tuple[int, int, int, int]]) -> Strengths | None:
    """Fit strengths from finished matches: list of (home_id, away_id, home_goals, away_goals)."""
    matches = [m for m in matches if None not in m]
    n = len(matches)
    if n < 20:
        return None  # not enough to estimate a league baseline

    avg_home = sum(m[2] for m in matches) / n
    avg_away = sum(m[3] for m in matches) / n
    if avg_home <= 0 or avg_away <= 0:
        return None

    hs: dict[int, int] = {}
    hc: dict[int, int] = {}
    hp: dict[int, int] = {}
    as_: dict[int, int] = {}
    ac: dict[int, int] = {}
    ap: dict[int, int] = {}
    for h, a, hg, ag in matches:
        hs[h] = hs.get(h, 0) + hg
        hc[h] = hc.get(h, 0) + ag
        hp[h] = hp.get(h, 0) + 1
        as_[a] = as_.get(a, 0) + ag
        ac[a] = ac.get(a, 0) + hg
        ap[a] = ap.get(a, 0) + 1

    rho = _estimate_rho(matches, avg_home, avg_away)
    s = Strengths(avg_home=avg_home, avg_away=avg_away, rho=rho)
    for t, games in hp.items():
        if games >= _MIN_GAMES:
            s.home_attack[t] = (hs[t] / games) / avg_home
            s.home_defense[t] = (hc[t] / games) / avg_away
    for t, games in ap.items():
        if games >= _MIN_GAMES:
            s.away_attack[t] = (as_[t] / games) / avg_away
            s.away_defense[t] = (ac[t] / games) / avg_home
    return s


def expected_goals(s: Strengths, home_id: int, away_id: int) -> tuple[float, float] | None:
    if not s.has(home_id, away_id):
        return None
    eh = s.home_attack[home_id] * s.away_defense[away_id] * s.avg_home
    ea = s.away_attack[away_id] * s.home_defense[home_id] * s.avg_away
    # guard against degenerate zeros
    return max(eh, 0.05), max(ea, 0.05)


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _goal_dist(lam: float) -> list[float]:
    return [poisson_pmf(k, lam) for k in range(_MAX_GOALS + 1)]


def predict_markets(
    eh: float, ea: float,
    ou_lines: tuple[float, ...] = (2.5,),
    rho: float = _RHO_DEFAULT,
) -> dict[str, dict]:
    """Build market probabilities from expected goals via the Dixon-Coles score matrix."""
    home = _goal_dist(eh)
    away = _goal_dist(ea)

    p_home = p_draw = p_away = 0.0
    p_btts_yes = 0.0
    over = {ln: 0.0 for ln in ou_lines}
    total = 0.0

    raw: list[tuple[int, int, float]] = []
    for i in range(_MAX_GOALS + 1):
        for j in range(_MAX_GOALS + 1):
            p = home[i] * away[j] * _tau(i, j, eh, ea, rho)
            raw.append((i, j, p))
            total += p

    for i, j, p in raw:
        p /= total   # renormalise after tau adjustments
        if i > j:
            p_home += p
        elif i == j:
            p_draw += p
        else:
            p_away += p
        if i >= 1 and j >= 1:
            p_btts_yes += p
        for ln in ou_lines:
            if i + j > ln:
                over[ln] += p

    out: dict[str, dict] = {
        "match_winner": {"home": p_home, "draw": p_draw, "away": p_away},
        "btts": {"yes": p_btts_yes, "no": 1.0 - p_btts_yes},
    }
    for ln in ou_lines:
        out.setdefault("over_under", {})[("over", ln)] = over[ln]
        out["over_under"][("under", ln)] = 1.0 - over[ln]
    return out


def predict(s: Strengths, home_id: int, away_id: int,
            ou_lines: tuple[float, ...] = (2.5,)) -> dict[str, dict] | None:
    eg = expected_goals(s, home_id, away_id)
    if eg is None:
        return None
    return predict_markets(eg[0], eg[1], ou_lines, rho=s.rho)
