"""Model calibration metrics.

Calibration checks whether predicted probabilities match real-world outcome rates.
A perfectly calibrated model that says 0.70 should win exactly 70% of the time.

Brier score (lower is better, 0 = perfect) and a reliability curve (binned) are the
two standard outputs.  Both work off (prob, outcome) pairs from settled paper bets.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class CalibrationBin:
    bin_mid: float      # centre of the probability bucket (e.g. 0.65 for the 60-70% bin)
    mean_pred: float    # average predicted probability inside this bin
    actual_freq: float  # fraction of bets in this bin that won
    n: int              # number of settled bets in this bin


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error between predicted probs and binary win/loss outcomes."""
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def reliability_curve(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Divide predictions into n_bins equal-width buckets; compare mean pred vs win rate."""
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        buckets[idx].append((p, o))

    result = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        bin_mid = (i + 0.5) / n_bins
        mean_pred = statistics.mean(p for p, _ in bucket)
        actual_freq = sum(o for _, o in bucket) / len(bucket)
        result.append(CalibrationBin(
            bin_mid=bin_mid,
            mean_pred=mean_pred,
            actual_freq=actual_freq,
            n=len(bucket),
        ))
    return result


def calibration_summary(probs: list[float], outcomes: list[int]) -> dict:
    """Return a dict of headline calibration metrics suitable for CLI display."""
    bs = brier_score(probs, outcomes)
    curve = reliability_curve(probs, outcomes)
    # Mean absolute calibration error: average |predicted - actual| across bins
    if curve:
        mace = sum(abs(b.mean_pred - b.actual_freq) for b in curve) / len(curve)
    else:
        mace = float("nan")
    return {
        "n": len(probs),
        "brier_score": bs,
        "mace": mace,
        "curve": curve,
    }
