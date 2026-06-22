"""Tests for Telegram alert formatting — no real HTTP calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from accubet.alerts.telegram import (
    build_daily_message,
    build_scan_message,
    notify_daily,
    send,
)
from accubet.value.accumulator import AccaLeg, AccaTicket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opp(market="match_winner", selection="home", price=1.85, prob=0.62,
         home="Arsenal", away="Chelsea", line=None, src="betway"):
    ev = prob * price - 1
    return SimpleNamespace(
        home=home, away=away, market=market, selection=selection, line=line,
        price=price, fair_prob=prob, price_source=src, ev=ev, _passes=True,
    )


def _ticket(tier="value"):
    legs = [
        AccaLeg(match_id=1, home="Arsenal", away="Chelsea",
                market="match_winner", selection="home", line=None, odds=1.85, prob=0.62),
        AccaLeg(match_id=2, home="Barca", away="Madrid",
                market="btts", selection="yes", line=None, odds=1.72, prob=0.68),
    ]
    combined_odds = 1.85 * 1.72
    combined_prob = 0.62 * 0.68
    return AccaTicket(
        tier=tier, mode="balanced", legs=legs,
        combined_odds=combined_odds, combined_prob=combined_prob,
        ev=combined_prob * combined_odds - 1, risk_rating="Medium",
    )


# ---------------------------------------------------------------------------
# build_daily_message
# ---------------------------------------------------------------------------

def test_daily_message_contains_date():
    from datetime import date
    msg = build_daily_message([], {}, 40, 100)
    assert date.today().strftime("%Y") in msg   # year is always present


def test_daily_message_no_bets():
    msg = build_daily_message([], {"banker": None, "value": None}, 40, 100)
    assert "No gate-passing" in msg
    assert "40/100" in msg


def test_daily_message_with_bets():
    opps = [_opp(), _opp("btts", "yes", 1.72, 0.68, "Barca", "Madrid")]
    msg = build_daily_message(opps, {}, 55, 100)
    assert "Arsenal v Chelsea" in msg
    assert "Barca v Madrid" in msg
    assert "2 value bet" in msg
    assert "55/100" in msg


def test_daily_message_includes_acca():
    msg = build_daily_message([_opp()], {"value": _ticket("value")}, 20, 100)
    assert "VALUE ACCA" in msg
    assert "Arsenal" in msg


def test_daily_message_truncates_long_content():
    # 100 opportunities with long names should be truncated to <= 4093 chars
    opps = [_opp(home="A" * 30, away="B" * 30)] * 100
    msg = build_daily_message(opps, {}, 50, 100)
    assert len(msg) <= 4093


# ---------------------------------------------------------------------------
# build_scan_message
# ---------------------------------------------------------------------------

def test_scan_message_no_bets():
    msg = build_scan_message([])
    assert "No gate-passing" in msg


def test_scan_message_with_bets():
    msg = build_scan_message([_opp()])
    assert "Arsenal v Chelsea" in msg
    assert "EV" in msg


# ---------------------------------------------------------------------------
# send — mocked requests
# ---------------------------------------------------------------------------

def test_send_success():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    with patch("accubet.alerts.telegram.requests.post", return_value=mock_resp) as mock_post:
        result = send("token123", "chat456", "hello")
    assert result is True
    mock_post.assert_called_once()
    call_data = mock_post.call_args[1]["data"]
    assert call_data["chat_id"] == "chat456"
    assert call_data["text"] == "hello"


def test_send_network_error_returns_false():
    with patch("accubet.alerts.telegram.requests.post", side_effect=ConnectionError("timeout")):
        result = send("token", "chat", "msg")
    assert result is False


def test_send_http_error_returns_false():
    import requests as req
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req.HTTPError("403")
    with patch("accubet.alerts.telegram.requests.post", return_value=mock_resp):
        result = send("bad_token", "chat", "msg")
    assert result is False


# ---------------------------------------------------------------------------
# notify_daily — no token means no send
# ---------------------------------------------------------------------------

def test_notify_daily_skipped_when_not_configured():
    from accubet.config import AppConfig, Secrets
    cfg = AppConfig(secrets=Secrets(apifootball_key="x"))  # no telegram keys
    with patch("accubet.alerts.telegram.send") as mock_send:
        result = notify_daily(cfg, [], {}, 0)
    assert result is False
    mock_send.assert_not_called()
