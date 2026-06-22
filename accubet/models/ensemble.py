"""Ensemble: blend model probabilities into a single 'true probability' per selection.

Market odds are the primary signal (~55%); internal models tilt it. Combination is done
per outcome group (e.g. {home,draw,away}) so each blended distribution still sums to 1.

**Graceful degradation** is the key property: a model that can't price a match (no team
history → Poisson/Glicko absent) is simply dropped and its weight is redistributed across
the models that *are* present. Market is always present, so we always get a number — for
data-sparse matches (e.g. internationals) that number collapses to the market view.

A by-product is **model agreement**, surfaced as a confidence score (the signal the spec
wanted for the confidence gate): tight agreement among several models = high confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

# group key = (market, line). line is None except for over_under.
Group = tuple


@dataclass
class GroupPrediction:
    dist: dict[str, float]      # {selection: probability}
    confidence: float           # 0-1, model agreement * coverage
    n_models: int
    models: list[str]


# --- converters: each model -> {group: {selection: prob}} ------------------

def market_to_groups(consensus_rows) -> dict[Group, dict[str, float]]:
    """consensus_rows: iterable of objects with .market/.selection/.line/.fair_prob."""
    out: dict[Group, dict[str, float]] = {}
    for c in consensus_rows:
        out.setdefault((c.market, c.line), {})[c.selection] = c.fair_prob
    return out


def poisson_to_groups(poisson_out: dict | None) -> dict[Group, dict[str, float]]:
    out: dict[Group, dict[str, float]] = {}
    if not poisson_out:
        return out
    for market, dist in poisson_out.items():
        if market == "over_under":
            for (sel, line), p in dist.items():
                out.setdefault(("over_under", line), {})[sel] = p
        else:
            out[(market, None)] = dict(dist)
    return out


def onex2_to_groups(dist: dict[str, float] | None) -> dict[Group, dict[str, float]]:
    """For glicko/form which only produce a {home,draw,away} 1X2 distribution."""
    return {("match_winner", None): dict(dist)} if dist else {}


# --- combination -----------------------------------------------------------

def _normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return dist
    return {k: v / total for k, v in dist.items()}


def _total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def combine_group(model_dists: dict[str, dict[str, float]], weights: dict[str, float]) -> GroupPrediction:
    """Blend several models' distributions for one outcome group."""
    present = [m for m, d in model_dists.items() if d]
    total_w = sum(weights.get(m, 0.0) for m in present) or 1.0

    selections = set()
    for m in present:
        selections |= set(model_dists[m])

    ens = {
        sel: sum(weights.get(m, 0.0) / total_w * model_dists[m].get(sel, 0.0) for m in present)
        for sel in selections
    }
    ens = _normalize(ens)

    # agreement = 1 - mean total-variation distance of each model from the blend
    if len(present) > 1:
        dispersion = sum(_total_variation(model_dists[m], ens) for m in present) / len(present)
        agreement = 1.0 - dispersion
    else:
        agreement = 0.6  # single model (usually market only): moderate confidence
    coverage = min(1.0, len(present) / 4.0)
    confidence = max(0.0, min(1.0, 0.5 * coverage + 0.5 * agreement))

    return GroupPrediction(dist=ens, confidence=round(confidence, 3),
                           n_models=len(present), models=present)


def ensemble(models_by_name: dict[str, dict[Group, dict[str, float]]],
             weights: dict[str, float]) -> dict[Group, GroupPrediction]:
    """models_by_name: {model_name: {group: {selection: prob}}} -> blended per group."""
    all_groups: set[Group] = set()
    for groups in models_by_name.values():
        all_groups |= set(groups)

    result: dict[Group, GroupPrediction] = {}
    for group in all_groups:
        model_dists = {
            name: groups.get(group, {}) for name, groups in models_by_name.items()
        }
        if not any(model_dists.values()):
            continue
        result[group] = combine_group(model_dists, weights)
    return result
