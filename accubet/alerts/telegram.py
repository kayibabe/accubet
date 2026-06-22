"""Telegram Bot API alert delivery.

Uses the raw Bot API over HTTPS so no extra dependency is needed beyond the
existing `requests` package.  All functions are fire-and-forget — errors are
logged but never propagate so a broken token never crashes the main pipeline.

Setup
-----
1. Create a bot via @BotFather, copy the token.
2. Start a chat with the bot (or add it to a group) and obtain the chat ID.
3. Set in .env:
       TELEGRAM_BOT_TOKEN=123456:ABC-...
       TELEGRAM_CHAT_ID=987654321
"""

from __future__ import annotations

import logging
from datetime import date

import requests

from accubet.config import AppConfig

log = logging.getLogger(__name__)

_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def send(token: str, chat_id: str, text: str) -> bool:
    """POST a message via the Telegram Bot API.  Returns True on HTTP success."""
    try:
        r = requests.post(
            _SEND_URL.format(token=token),
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        log.warning("Telegram alert failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_opp(opp, n: int) -> str:
    src = "BW" if opp.price_source == "betway" else "best"
    line = f" {opp.line}" if opp.line is not None else ""
    return (
        f"{n}. <b>{opp.home} v {opp.away}</b>\n"
        f"   {opp.market}: <b>{opp.selection}{line}</b>  "
        f"true={opp.fair_prob:.0%}  @{opp.price:.2f} ({src})  "
        f"EV=<b>{opp.ev * 100:+.1f}%</b>"
    )


def _fmt_acca(tier: str, ticket) -> str:
    legs = "\n".join(
        f"  - {lg.home} v {lg.away}: {lg.selection} @{lg.odds:.2f}"
        for lg in ticket.legs
    )
    return (
        f"\n<b>{tier.upper()} ACCA</b>  "
        f"odds={ticket.combined_odds:.2f}  EV={ticket.ev * 100:+.1f}%\n{legs}"
    )


def build_daily_message(
    passers: list,
    tickets: dict,
    quota_used: int,
    quota_limit: int,
) -> str:
    today = date.today().strftime("%a %d %b %Y")
    parts = [f"<b>AccuBet Daily - {today}</b>"]

    if passers:
        parts.append(f"\n<b>{len(passers)} value bet(s) found</b>")
        for i, o in enumerate(passers, 1):
            parts.append(_fmt_opp(o, i))
    else:
        parts.append("\nNo gate-passing bets today.")

    for tier, ticket in tickets.items():
        if ticket is not None:
            parts.append(_fmt_acca(tier, ticket))

    parts.append(f"\n<i>Quota: {quota_used}/{quota_limit}</i>")

    text = "\n".join(parts)
    return text[:4090] + "..." if len(text) > 4090 else text


def build_scan_message(passers: list) -> str:
    today = date.today().strftime("%a %d %b %Y")
    if not passers:
        return f"<b>AccuBet Scan - {today}</b>\n\nNo gate-passing bets right now."
    parts = [f"<b>AccuBet Scan - {today}</b>\n\n<b>{len(passers)} opportunity(ies)</b>"]
    for i, o in enumerate(passers, 1):
        parts.append(_fmt_opp(o, i))
    text = "\n".join(parts)
    return text[:4090] + "..." if len(text) > 4090 else text


# ---------------------------------------------------------------------------
# Convenience wrappers called from CLI
# ---------------------------------------------------------------------------

def notify_daily(cfg: AppConfig, passers: list, tickets: dict,
                 quota_used: int) -> bool:
    """Send daily summary alert if Telegram is configured.  Returns True if sent."""
    token = cfg.secrets.telegram_bot_token
    chat_id = cfg.secrets.telegram_chat_id
    if not token or not chat_id:
        return False
    text = build_daily_message(passers, tickets, quota_used, cfg.apifootball.daily_request_limit)
    ok = send(token, chat_id, text)
    if ok:
        log.info("Telegram daily alert sent (%d bets, %d accas).",
                 len(passers), sum(1 for t in tickets.values() if t))
    return ok


def notify_scan(cfg: AppConfig, passers: list) -> bool:
    """Send scan alert if Telegram is configured.  Returns True if sent."""
    token = cfg.secrets.telegram_bot_token
    chat_id = cfg.secrets.telegram_chat_id
    if not token or not chat_id:
        return False
    return send(token, chat_id, build_scan_message(passers))
