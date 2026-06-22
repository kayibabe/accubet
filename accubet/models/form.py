"""Recent-form model.

A light signal from each team's last-N results (points per game). Converted to a 1X2
distribution via the points-per-game gap, with the same closeness-driven draw model used
by the ratings model. It only carries ~10% ensemble weight — a tilt, not a primary view.
"""

from __future__ import annotations


def points_per_game(
    matches_chrono: list[tuple[int, int, int, int]], team_id: int, n: int = 5
) -> float | None:
    """PPG over a team's most recent ``n`` matches (None if it has played none)."""
    pts: list[int] = []
    for h, a, hg, ag in reversed(matches_chrono):  # most recent first
        if team_id == h:
            pts.append(3 if hg > ag else 1 if hg == ag else 0)
        elif team_id == a:
            pts.append(3 if ag > hg else 1 if hg == ag else 0)
        if len(pts) >= n:
            break
    if not pts:
        return None
    return sum(pts) / len(pts)


def form_to_1x2(home_ppg: float | None, away_ppg: float | None,
                *, home_adv_ppg: float = 0.35, draw_max: float = 0.30) -> dict[str, float] | None:
    if home_ppg is None or away_ppg is None:
        return None
    # PPG in [0,3]; map the (home-advantaged) gap to a 2-way probability via a logistic.
    diff = (home_ppg + home_adv_ppg) - away_ppg
    e = 1.0 / (1.0 + 10.0 ** (-diff / 1.5))
    p_draw = draw_max * (1.0 - abs(2.0 * e - 1.0))
    return {
        "home": e * (1.0 - p_draw),
        "draw": p_draw,
        "away": (1.0 - e) * (1.0 - p_draw),
    }
