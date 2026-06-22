"""Two-tier accumulator builder.

Both tiers draw from the same pool of value opportunities but optimize different goals:

* **Banker** — 2-3 legs, combined odds ~1.3-1.8, *maximize joint win probability* (legs that
  are at least non-negative EV). High strike, low odds.
* **Value** — 2-4 legs, combined odds 3.0-5.0, *maximize combined EV* (legs that clear the
  full value gate). Profit comes from EV, not strike rate.

No two legs may come from the same match. Joint probability uses independence across
distinct matches (cross-match correlation is low; same-match combos are forbidden, which is
where correlation actually bites). A copula refinement can replace the product later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

from accubet.config import AppConfig, TierCfg

_POOL_CAP = 12  # cap candidate legs to keep the combinatorial search cheap


@dataclass
class AccaLeg:
    match_id: int
    home: str
    away: str
    market: str
    selection: str
    line: float | None
    odds: float
    prob: float


@dataclass
class AccaTicket:
    tier: str
    mode: str
    legs: list[AccaLeg]
    combined_odds: float
    combined_prob: float
    ev: float
    risk_rating: str = field(default="")


def _risk_rating(combined_prob: float) -> str:
    if combined_prob >= 0.60:
        return "Low"
    if combined_prob >= 0.40:
        return "Medium"
    if combined_prob >= 0.25:
        return "High"
    return "Very High"


def _opp_to_leg(o: Any) -> AccaLeg:
    return AccaLeg(o.match_id, o.home, o.away, o.market, o.selection, o.line, o.price, o.fair_prob)


def build_tier(
    opps: list[Any],
    tier_cfg: TierCfg,
    tier_name: str,
    *,
    objective: str,                       # "prob" (banker) | "ev" (value)
    pool: Callable[[Any], bool],
    mode: str = "balanced",
    max_legs: int | None = None,
) -> AccaTicket | None:
    legs = [_opp_to_leg(o) for o in opps if pool(o)]
    # keep the strongest candidates by the tier's objective
    key = (lambda leg: leg.prob) if objective == "prob" else (lambda leg: leg.prob * leg.odds - 1)
    legs = sorted(legs, key=key, reverse=True)[:_POOL_CAP]

    hi = min(tier_cfg.max_legs, max_legs) if max_legs else tier_cfg.max_legs
    best: tuple[AccaLeg, ...] | None = None
    best_score = float("-inf")

    for n in range(tier_cfg.min_legs, hi + 1):
        for combo in combinations(legs, n):
            if len({leg.match_id for leg in combo}) < n:
                continue  # never two legs from the same match
            combined_odds = math.prod(leg.odds for leg in combo)
            if not (tier_cfg.min_combined_odds <= combined_odds <= tier_cfg.max_combined_odds):
                continue
            combined_prob = math.prod(leg.prob for leg in combo)
            ev = combined_prob * combined_odds - 1.0
            score = combined_prob if objective == "prob" else ev
            if score > best_score:
                best, best_score = combo, score

    if best is None:
        return None
    combined_odds = math.prod(leg.odds for leg in best)
    combined_prob = math.prod(leg.prob for leg in best)
    return AccaTicket(
        tier=tier_name,
        mode=mode,
        legs=list(best),
        combined_odds=combined_odds,
        combined_prob=combined_prob,
        ev=combined_prob * combined_odds - 1.0,
        risk_rating=_risk_rating(combined_prob),
    )


def build_accumulators(opps: list[Any], cfg: AppConfig, mode: str = "balanced") -> dict[str, AccaTicket | None]:
    """Build the banker and value tickets from a pool of value opportunities."""
    mode_cfg = cfg.accumulator.modes.get(mode, {}) if cfg.accumulator.modes else {}
    max_legs = mode_cfg.get("max_legs")

    banker = build_tier(
        opps, cfg.accumulator.banker, "banker",
        objective="prob",
        pool=lambda o: o.price_source == "betway" and o.ev >= 0,
        mode=mode, max_legs=max_legs,
    )
    value = build_tier(
        opps, cfg.accumulator.value, "value",
        objective="ev",
        pool=lambda o: getattr(o, "_passes", False),
        mode=mode, max_legs=max_legs,
    )
    return {"banker": banker, "value": value}
