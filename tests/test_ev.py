"""Unit tests for the core EV math."""

import math

import pytest

from accubet.value.ev import expected_value, implied_probability, is_value, value_pct


def test_implied_probability():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)


def test_implied_probability_rejects_bad_odds():
    with pytest.raises(ValueError):
        implied_probability(1.0)


def test_expected_value_hand_worked():
    # true 30% chance at 4.0 odds: 0.30*4 - 1 = +0.20
    assert expected_value(0.30, 4.0) == pytest.approx(0.20)
    # fair bet: 50% at 2.0 → EV 0
    assert expected_value(0.50, 2.0) == pytest.approx(0.0)


def test_value_pct():
    # 30% true vs 25% implied (odds 4.0) → +5% value
    assert value_pct(0.30, 4.0) == pytest.approx(0.05)


def test_is_value_gate():
    assert is_value(0.30, 4.0, min_ev=0.05) is True   # EV +20%
    assert is_value(0.26, 4.0, min_ev=0.05) is False  # EV +4% < 5%
