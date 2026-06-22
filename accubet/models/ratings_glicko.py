"""Glicko-2 rating system + a rating->1X2 mapping.

Glicko-2 tracks each team's rating, rating deviation (uncertainty) and volatility, which
handles new/promoted/inconsistent teams better than plain Elo. We process finished matches
chronologically (one match per update), then map the pre-match rating gap to home/draw/away
probabilities with a home-advantage term and a closeness-driven draw model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SCALE = 173.7178
_PI2 = math.pi ** 2


@dataclass
class Rating:
    r: float = 1500.0
    rd: float = 350.0
    sigma: float = 0.06


class Glicko2:
    def __init__(self, tau: float = 0.5):
        self.tau = tau

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / _PI2)

    def _rate_one(self, p: Rating, opp: Rating, s: float) -> Rating:
        mu = (p.r - 1500.0) / _SCALE
        phi = p.rd / _SCALE
        sigma = p.sigma
        mu_j = (opp.r - 1500.0) / _SCALE
        phi_j = opp.rd / _SCALE

        g = self._g(phi_j)
        E = 1.0 / (1.0 + math.exp(-g * (mu - mu_j)))
        E = min(max(E, 1e-9), 1 - 1e-9)
        v = 1.0 / (g ** 2 * E * (1 - E))
        delta = v * g * (s - E)

        a = math.log(sigma ** 2)
        tau = self.tau

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (delta ** 2 - phi ** 2 - v - ex)
            den = 2.0 * (phi ** 2 + v + ex) ** 2
            return num / den - (x - a) / tau ** 2

        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau

        fa, fb = f(A), f(B)
        while abs(B - A) > 1e-6:
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa /= 2.0
            B, fb = C, fc
        sigma_p = math.exp(A / 2.0)

        phi_star = math.sqrt(phi ** 2 + sigma_p ** 2)
        phi_p = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
        mu_p = mu + phi_p ** 2 * g * (s - E)
        return Rating(r=_SCALE * mu_p + 1500.0, rd=_SCALE * phi_p, sigma=sigma_p)

    def rate_match(self, home: Rating, away: Rating, score_home: float) -> tuple[Rating, Rating]:
        """Return updated (home, away) ratings. score_home in {1.0, 0.5, 0.0}."""
        new_home = self._rate_one(home, away, score_home)
        new_away = self._rate_one(away, home, 1.0 - score_home)
        return new_home, new_away


def fit_ratings(matches_chrono: list[tuple[int, int, int, int]], tau: float = 0.5) -> dict[int, Rating]:
    """Process chronologically-ordered (home_id, away_id, home_goals, away_goals)."""
    glicko = Glicko2(tau=tau)
    ratings: dict[int, Rating] = {}
    for h, a, hg, ag in matches_chrono:
        if None in (h, a, hg, ag):
            continue
        rh = ratings.get(h, Rating())
        ra = ratings.get(a, Rating())
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        ratings[h], ratings[a] = glicko.rate_match(rh, ra, score)
    return ratings


def ratings_to_1x2(
    ratings: dict[int, Rating], home_id: int, away_id: int,
    *, home_adv: float = 60.0, draw_max: float = 0.30,
) -> dict[str, float] | None:
    """Map a rating gap to 1X2 probabilities (None if either team is unrated)."""
    if home_id not in ratings or away_id not in ratings:
        return None
    d = (ratings[home_id].r + home_adv) - ratings[away_id].r
    e = 1.0 / (1.0 + 10.0 ** (-d / 400.0))           # 2-way: prob home outperforms away
    p_draw = draw_max * (1.0 - abs(2.0 * e - 1.0))    # draws likeliest when evenly matched
    return {
        "home": e * (1.0 - p_draw),
        "draw": p_draw,
        "away": (1.0 - e) * (1.0 - p_draw),
    }
