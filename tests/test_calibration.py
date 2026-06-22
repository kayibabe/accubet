"""Tests for the calibration metrics module."""

import math

import pytest

from accubet.backtest.calibration import (
    CalibrationBin,
    brier_score,
    calibration_summary,
    reliability_curve,
)


def test_brier_score_perfect():
    # Perfect predictions: predict 1.0 for all wins, 0.0 for all losses.
    assert brier_score([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_brier_score_worst():
    # Worst case: predict 1.0 for losses and 0.0 for wins -> BS = 1.0
    assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_brier_score_random_50_50():
    # Predicting 0.5 for everything on balanced data gives BS = 0.25
    probs = [0.5] * 100
    outcomes = [1] * 50 + [0] * 50
    assert brier_score(probs, outcomes) == pytest.approx(0.25)


def test_brier_score_empty_returns_nan():
    assert math.isnan(brier_score([], []))


def test_reliability_curve_single_bin():
    # All predictions in the 60-70% bucket, 2 of 4 win.
    probs = [0.65, 0.62, 0.68, 0.66]
    outcomes = [1, 0, 1, 0]
    curve = reliability_curve(probs, outcomes, n_bins=10)
    assert len(curve) == 1
    b = curve[0]
    assert b.n == 4
    assert b.actual_freq == pytest.approx(0.5)
    assert b.bin_mid == pytest.approx(0.65)


def test_reliability_curve_empty_bins_skipped():
    probs = [0.1, 0.9]
    outcomes = [0, 1]
    curve = reliability_curve(probs, outcomes, n_bins=10)
    # Only 2 non-empty bins (the 0-10% bin and the 80-100% bin).
    assert len(curve) == 2
    assert all(b.n >= 1 for b in curve)


def test_reliability_curve_frequencies():
    # Low bucket: predict 0.2 for 5 bets, none win.
    # High bucket: predict 0.8 for 5 bets, all win.
    probs = [0.2] * 5 + [0.8] * 5
    outcomes = [0] * 5 + [1] * 5
    curve = reliability_curve(probs, outcomes, n_bins=10)
    low = next(b for b in curve if b.bin_mid < 0.5)
    high = next(b for b in curve if b.bin_mid > 0.5)
    assert low.actual_freq == pytest.approx(0.0)
    assert high.actual_freq == pytest.approx(1.0)


def test_calibration_summary_keys():
    probs = [0.6, 0.7, 0.4, 0.3]
    outcomes = [1, 1, 0, 0]
    s = calibration_summary(probs, outcomes)
    assert "n" in s and "brier_score" in s and "mace" in s and "curve" in s
    assert s["n"] == 4
    assert 0.0 <= s["brier_score"] <= 1.0
    assert 0.0 <= s["mace"] <= 1.0
