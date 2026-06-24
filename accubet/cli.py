"""AccuBet command-line interface.

Commands
--------
init-db   Create the database schema.
ingest    Pull fixtures + odds (cache-first / quota-guarded) and build consensus.
scan      Rank value opportunities (fair consensus vs Betway / best price).
quota     Show today's API request usage.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from accubet.config import get_config
from accubet.ingestion.apifootball import ApiFootballClient
from accubet.ingestion.pipeline import (
    ingest_fixtures, ingest_history, ingest_local_odds, ingest_odds,
)
from accubet.ingestion.quota import QuotaExceeded, remaining, requests_used_today
from accubet.ingestion.scrapers.betpawa import BetPawaMalawiConnector
from accubet.ingestion.scrapers.betway import BetwayMalawiConnector
from accubet.logging_setup import setup_logging
from accubet.market.comparison import persist_value_bets, scan as scan_value
from accubet.market.consensus import build_all_consensus
from accubet.models.predictor import run_predictions
from accubet.storage.db import init_db as _init_db, session_scope
from accubet.storage.models import Consensus, Match
from accubet.tracking.performance import report as perf_report
from accubet.tracking.tracked_bets import log_accumulators, log_singles, settle as settle_bets
from accubet.value.accumulator import build_accumulators

app = typer.Typer(add_completion=False, help="AccuBet — football value-betting engine.")
console = Console()


def _dates(start: str | None, days: int) -> list[str]:
    base = datetime.strptime(start, "%Y-%m-%d").date() if start else date.today()
    return [(base + timedelta(days=i)).isoformat() for i in range(max(1, days))]


def _upcoming_match_ids(session) -> list[int]:
    """Matches with consensus that are upcoming OR played within the last 3 days.

    Including recent past matches lets the pipeline retroactively log paper bets
    for games that kicked off between pipeline runs.
    """
    cutoff = datetime.now() - timedelta(days=3)
    return list(session.execute(
        select(Match.id)
        .join(Consensus, Consensus.match_id == Match.id)
        .where(
            (Match.status.in_(("NS", "TBD", "PST"))) |
            (Match.kickoff >= cutoff)
        )
        .distinct()
    ).scalars().all())


def _pct(x: float | None) -> str:
    return f"{x * 100:+.1f}%" if x is not None else "-"


def _rate(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "-"


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the database schema."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    _init_db()
    console.print("[green]Database initialised.[/green]")


@app.command()
def ingest(
    date_: str = typer.Option(None, "--date", help="Start date YYYY-MM-DD (default: today)."),
    days: int = typer.Option(1, "--days", help="Number of days from start to ingest."),
    no_odds: bool = typer.Option(False, "--no-odds", help="Fixtures only, skip odds."),
    force: bool = typer.Option(False, "--force", help="Bypass cache (spends quota)."),
) -> None:
    """Ingest fixtures + odds and build consensus (cache-first; re-runs spend ~0 quota)."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    _init_db()

    with session_scope() as session:
        client = ApiFootballClient(cfg, session)
        connectors = []
        if cfg.local_book.enabled:
            connectors.append(BetwayMalawiConnector(cfg))
            connectors.append(BetPawaMalawiConnector(cfg))

        for d in _dates(date_, days):
            matches, report = ingest_fixtures(session, cfg, client, d, force=force)
            if not no_odds and matches:
                ingest_odds(session, cfg, client, matches, report)
            local_n = 0
            for connector in connectors:
                if matches:
                    local_n += ingest_local_odds(session, connector, matches)
            built = build_all_consensus(session, [m.id for m in matches], min_books=1)

            console.print(
                f"[bold]{d}[/bold]  fixtures_seen={report.fixtures_seen} "
                f"tracked={report.matches_tracked} new={report.new_matches} "
                f"odds_pulled={report.odds_pulled} odds_cached={report.odds_from_cache} "
                f"betway_rows={local_n} consensus_rows={built}"
            )

        used = requests_used_today(session)
        console.print(
            f"[dim]API requests used today: {used}/"
            f"{cfg.apifootball.daily_request_limit} "
            f"(remaining {remaining(session, cfg)})[/dim]"
        )


@app.command()
def scan(
    top: int = typer.Option(20, "--top", help="How many opportunities to show."),
    gates_only: bool = typer.Option(False, "--gates-only", help="Only show gate-passers."),
    all_markets: bool = typer.Option(
        False, "--all-markets", help="Include all markets/lines (bypass liquidity filter)."
    ),
    notify: bool = typer.Option(False, "--notify", help="Send Telegram alert for gate-passers."),
) -> None:
    """Rank value opportunities: fair consensus probability vs the price you can take."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)

    with session_scope() as session:
        # scan upcoming matches that have a consensus.
        match_ids = _upcoming_match_ids(session)

        if not match_ids:
            console.print(
                "[yellow]No upcoming matches with consensus yet. Run `accubet ingest` first.[/yellow]"
            )
            raise typer.Exit()

        opps = scan_value(session, cfg, list(match_ids), restrict=not all_markets)
        stored = persist_value_bets(session, opps)

        shown = [o for o in opps if o._passes] if gates_only else opps
        shown = shown[:top]

        table = Table(title=f"AccuBet value scan - EV gate {cfg.value.min_ev:.0%}, "
                            f"min {cfg.value.min_books_for_consensus} books")
        table.add_column("#", justify="right")
        table.add_column("Match", overflow="fold", max_width=34)
        table.add_column("Market")
        table.add_column("Pick")
        table.add_column("True%", justify="right")
        table.add_column("Mdl")
        table.add_column("Price", justify="right")
        table.add_column("Src")
        table.add_column("EV%", justify="right")
        table.add_column("Val%", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Bk", justify="right")
        table.add_column("STM", justify="center")

        for i, o in enumerate(shown, 1):
            mark = "[green]*[/green]" if o._passes else ""
            line = f" {o.line}" if o.line is not None else ""
            stm = "[cyan]S[/cyan]" if getattr(o, "steam_move", False) else ""
            table.add_row(
                f"{i}{mark}",
                f"{o.home} v {o.away}",
                o.market,
                f"{o.selection}{line}",
                f"{o.fair_prob:.0%}",
                "ens" if o.prob_source == "ensemble" else "mkt",
                f"{o.price:.2f}",
                {"betway": "BW", "betpawa": "BP"}.get(o.price_source, o.price_source),
                f"{o.ev*100:+.1f}",
                f"{o.value_pct*100:+.1f}",
                f"{o.confidence:.2f}",
                str(o.n_books),
                stm,
            )
        console.print(table)
        console.print(
            f"[dim]{len(opps)} opportunities | {sum(o._passes for o in opps)} pass gates | "
            f"{stored} saved to value_bets.[/dim]"
        )
        if notify:
            from accubet.alerts.telegram import notify_scan
            passers_list = [o for o in opps if o._passes]
            if notify_scan(cfg, passers_list):
                console.print("[dim]Telegram alert sent.[/dim]")
            elif not cfg.secrets.telegram_bot_token:
                console.print(
                    "[yellow]--notify set but TELEGRAM_BOT_TOKEN not configured in .env[/yellow]"
                )


@app.command()
def quota() -> None:
    """Show today's API-Football request usage."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    _init_db()
    with session_scope() as session:
        used = requests_used_today(session)
        console.print(
            f"API-Football today: [bold]{used}[/bold]/{cfg.apifootball.daily_request_limit} used | "
            f"[bold]{remaining(session, cfg)}[/bold] remaining "
            f"(soft stop at {cfg.apifootball.quota_soft_stop})."
        )


@app.command()
def history(
    season: int = typer.Option(2024, "--season", help="Season year (free tier: 2022-2024)."),
    league: int = typer.Option(None, "--league", help="Single league id (default: all configured)."),
) -> None:
    """Pull league-season results (training data for the Poisson/Glicko/form models)."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    _init_db()
    with session_scope() as session:
        client = ApiFootballClient(cfg, session)
        league_ids = [league] if league else cfg.league_ids
        for lid in league_ids:
            try:
                rep = ingest_history(session, cfg, client, lid, season)
            except QuotaExceeded as exc:
                console.print(f"[yellow]{exc}[/yellow]")
                break
            console.print(
                f"league {lid} season {season}: fixtures={rep['fixtures']} "
                f"stored={rep['stored']} results={rep['results']}"
            )
        console.print(f"[dim]API requests used today: {requests_used_today(session)}/"
                      f"{cfg.apifootball.daily_request_limit}[/dim]")


@app.command()
def predict() -> None:
    """Run the model ensemble for upcoming matches and store true-probability predictions."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    with session_scope() as session:
        match_ids = _upcoming_match_ids(session)
        if not match_ids:
            console.print("[yellow]No upcoming matches with consensus. Run `accubet ingest` first.[/yellow]")
            raise typer.Exit()
        n = run_predictions(session, cfg, match_ids)
        console.print(
            f"[green]Stored {n} ensemble prediction(s) across {len(match_ids)} match(es).[/green] "
            f"[dim](matches without league history degrade to the market view.)[/dim]"
        )


@app.command()
def track(mode: str = typer.Option("balanced", "--mode", help="conservative|balanced|aggressive")) -> None:
    """Auto-log the top-N EV singles + both accumulator tiers as paper bets."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    with session_scope() as session:
        match_ids = _upcoming_match_ids(session)
        if not match_ids:
            console.print("[yellow]No upcoming matches with consensus. Run `accubet ingest` first.[/yellow]")
            raise typer.Exit()

        opps = scan_value(session, cfg, match_ids)
        n_singles = log_singles(session, cfg, opps)
        tickets = build_accumulators(opps, cfg, mode)
        n_acc = log_accumulators(session, cfg, tickets)

        console.print(f"[green]Tracked {n_singles} new single(s) and {n_acc} new accumulator(s).[/green]")
        for tier, t in tickets.items():
            if t is None:
                console.print(f"  [dim]{tier} tier: no qualifying ticket today.[/dim]")
                continue
            legs = " + ".join(f"{leg.home} v {leg.away}: {leg.selection}@{leg.odds:.2f}" for leg in t.legs)
            console.print(
                f"  [bold]{tier}[/bold] odds={t.combined_odds:.2f} prob={t.combined_prob:.0%} "
                f"EV={t.ev*100:+.1f}% risk={t.risk_rating}\n    {legs}"
            )


@app.command()
def settle() -> None:
    """Grade tracked paper bets whose matches have finished."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    with session_scope() as session:
        result = settle_bets(session, cfg)
        console.print(f"[green]Settled {result['settled']} tracked bet(s).[/green]")


@app.command()
def report() -> None:
    """Per-market and overall performance of the tracked paper bets."""
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    with session_scope() as session:
        rep = perf_report(session)
        o = rep["overall"]

        def _table(title: str, rows) -> Table:
            t = Table(title=title)
            for col, just in (("", "left"), ("Bets", "right"), ("Settled", "right"),
                              ("W-L-V", "right"), ("Win%", "right"), ("Staked", "right"),
                              ("P&L", "right"), ("ROI", "right"), ("Pending", "right")):
                t.add_column(col, justify=just)
            for r in rows:
                t.add_row(
                    r.label, str(r.total), str(r.settled),
                    f"{r.wins}-{r.losses}-{r.voids}", _rate(r.win_rate),
                    f"{r.staked:.0f}", f"{r.pnl:+.2f}", _pct(r.roi), str(r.pending),
                )
            return t

        console.print(_table("Overall + by kind", [o] + rep["by_kind"]))
        console.print(_table("By market", rep["by_market"]))
        if o.settled == 0:
            console.print("[dim]No settled bets yet - run `accubet ingest` (captures results) "
                          "then `accubet settle` after kickoff.[/dim]")


@app.command()
def daily(
    days: int = typer.Option(2, "--days", help="Days ahead to ingest (default: 2 = today + tomorrow)."),
    back: int = typer.Option(1, "--back", help="Days back to fetch results for settlement (default: 1)."),
    mode: str = typer.Option("balanced", "--mode", help="Accumulator mode: conservative|balanced|aggressive."),
    no_settle: bool = typer.Option(False, "--no-settle", help="Skip the auto-settlement step."),
    top: int = typer.Option(5, "--top", help="Top opportunities to show in summary."),
) -> None:
    """Full daily pipeline: ingest -> predict -> track -> settle -> summarise.

    Runs every step in sequence so one command keeps the system current.
    Schedule it with Task Scheduler (Windows) or cron (Linux/macOS).
    """
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    _init_db()

    with session_scope() as session:
        client = ApiFootballClient(cfg, session)
        daily_connectors = []
        if cfg.local_book.enabled:
            daily_connectors.append(BetwayMalawiConnector(cfg))
            daily_connectors.append(BetPawaMalawiConnector(cfg))

        # --- 1. Ingest ---------------------------------------------------
        console.rule("[bold cyan]1 / 4  Ingest[/bold cyan]")
        back_dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(back, 0, -1)]
        fwd_dates = _dates(None, days)
        total_new = 0
        for d in back_dates + fwd_dates:
            past = d < date.today().isoformat()
            matches, rep = ingest_fixtures(session, cfg, client, d, force=past)
            if not past and matches:
                ingest_odds(session, cfg, client, matches, rep)
            if not past and matches:
                for connector in daily_connectors:
                    ingest_local_odds(session, connector, matches)
            built = build_all_consensus(session, [m.id for m in matches], min_books=1)
            total_new += rep.new_matches
            tag = "past" if past else "upcoming"
            console.print(
                f"  {d} [{tag}]  fixtures={rep.fixtures_seen} new={rep.new_matches}"
                + (f" consensus={built}" if not past else "")
            )
        used = requests_used_today(session)
        console.print(
            f"  [dim]quota {used}/{cfg.apifootball.daily_request_limit} used "
            f"({remaining(session, cfg)} left)[/dim]"
        )

        # --- 2. Predict --------------------------------------------------
        console.rule("[bold cyan]2 / 4  Predict[/bold cyan]")
        match_ids = _upcoming_match_ids(session)
        if match_ids:
            n_preds = run_predictions(session, cfg, match_ids)
            console.print(
                f"  {n_preds} prediction(s) stored across {len(match_ids)} upcoming match(es)."
            )
        else:
            console.print("  [dim]No upcoming matches with consensus — skipping.[/dim]")

        # --- 3. Track ----------------------------------------------------
        console.rule("[bold cyan]3 / 4  Track[/bold cyan]")
        n_singles = n_acc = 0
        passers: list = []
        if match_ids:
            opps = scan_value(session, cfg, match_ids)
            persist_value_bets(session, opps)
            n_singles = log_singles(session, cfg, opps)
            tickets = build_accumulators(opps, cfg, mode)
            n_acc = log_accumulators(session, cfg, tickets)
            passers = [o for o in opps if o._passes]
            console.print(
                f"  {len(opps)} opportunities | {len(passers)} pass gates | "
                f"{n_singles} new single(s) | {n_acc} new accumulator(s)"
            )
            for o in passers[:top]:
                line = f" {o.line}" if o.line is not None else ""
                src_label = {"betway": "BW", "betpawa": "BP"}.get(o.price_source, o.price_source)
                src = f"[green]{src_label}[/green]" if o.price_source != "best" else "best"
                console.print(
                    f"    [green]*[/green] {o.home} v {o.away}  "
                    f"{o.market} {o.selection}{line}  "
                    f"true={o.fair_prob:.0%}  @{o.price:.2f} ({src})  "
                    f"EV=[bold green]{o.ev * 100:+.1f}%[/bold green]"
                )
            for tier, t in tickets.items():
                if t:
                    legs = " + ".join(
                        f"{lg.selection}@{lg.odds:.2f}" for lg in t.legs
                    )
                    console.print(
                        f"    [dim]{tier}[/dim] odds={t.combined_odds:.2f} "
                        f"EV={t.ev * 100:+.1f}%  [{legs}]"
                    )
        else:
            console.print("  [dim]No upcoming matches — nothing to track.[/dim]")

        # --- 4. Settle ---------------------------------------------------
        if not no_settle:
            console.rule("[bold cyan]4 / 4  Settle[/bold cyan]")
            result = settle_bets(session, cfg)
            if result["settled"]:
                console.print(f"  Settled {result['settled']} tracked bet(s).")
                rep_out = perf_report(session)
                ov = rep_out["overall"]
                if ov.settled:
                    console.print(
                        f"  Lifetime: {ov.settled} settled | roi={_pct(ov.roi)} | "
                        f"win%={_rate(ov.win_rate)} | pnl={ov.pnl:+.2f}"
                    )
            else:
                console.print("  [dim]Nothing to settle yet.[/dim]")
        else:
            console.rule("[dim]4 / 4  Settle (skipped)[/dim]")

    # --- Telegram alert -----------------------------------------------
    from accubet.alerts.telegram import notify_daily
    sent = notify_daily(cfg, passers, tickets if match_ids else {}, used)
    if sent:
        console.print("[dim]Telegram alert sent.[/dim]")
    elif cfg.secrets.telegram_bot_token:
        console.print("[dim]Telegram configured but alert failed (check logs).[/dim]")

    console.print("\n[bold green]Done.[/bold green]  "
                  f"{len(passers)} gate-passing bet(s) tracked today.")


@app.command()
def backtest(
    months: int = typer.Option(6, "--months", help="Months back from today (default: 6)."),
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (default: today)."),
    window: int = typer.Option(30, "--window", help="Rolling window in days (default: 30)."),
) -> None:
    """Walk-forward performance analysis over the settled paper-bet history."""
    from accubet.backtest.walkforward import months_ago, overall_metrics, run_walkforward

    cfg = get_config()
    setup_logging(cfg.secrets.log_level)

    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    start_d = datetime.strptime(start, "%Y-%m-%d").date() if start else months_ago(end_d, months)

    with session_scope() as session:
        windows = run_walkforward(session, start_d, end_d, window_days=window)
        overall = overall_metrics(session, start_d, end_d)

        def _roi_cell(v: float | None) -> str:
            if v is None:
                return "-"
            color = "green" if v > 0 else ("red" if v < 0 else "white")
            return f"[{color}]{v * 100:+.1f}[/{color}]"

        table = Table(
            title=f"Walk-forward  {start_d} to {end_d}  (window={window}d)",
        )
        for col, just in [
            ("Period start", "left"), ("Bets", "right"), ("Staked", "right"),
            ("P&L", "right"), ("ROI%", "right"), ("Win%", "right"),
            ("Sharpe", "right"), ("MaxDD", "right"),
            ("CLV avg", "right"), ("CLV+%", "right"),
        ]:
            table.add_column(col, justify=just)

        for wr in windows:
            table.add_row(
                str(wr.period_start),
                str(wr.n_bets),
                f"{wr.staked:.0f}",
                f"{wr.pnl:+.2f}",
                _roi_cell(wr.roi),
                _rate(wr.win_rate),
                f"{wr.sharpe:.2f}" if wr.sharpe is not None else "-",
                f"{wr.max_dd:.2f}",
                f"{wr.clv_mean * 100:+.2f}%" if wr.clv_mean is not None else "-",
                _rate(wr.clv_positive_pct),
            )
        console.print(table)

        o = overall
        clv_str = (
            f"clv={o.clv_mean * 100:+.2f}% ({o.n_with_clv} bets)"
            if o.clv_mean is not None
            else f"clv=- ({o.n_with_clv} bets with CLV data)"
        )
        console.print(
            f"\n[bold]Overall[/bold]  {o.n_bets} bets  staked={o.staked:.0f}  "
            f"pnl={o.pnl:+.2f}  roi={_pct(o.roi)}  win%={_rate(o.win_rate)}  "
            f"sharpe={'%.2f' % o.sharpe if o.sharpe is not None else '-'}  "
            f"max-drawdown={o.max_dd:.2f}  {clv_str}"
        )

        if o.n_bets == 0:
            console.print(
                "[dim]No settled bets in this range. Run `accubet ingest` then "
                "`accubet settle` after matches finish.[/dim]"
            )


@app.command()
def calibrate(
    months: int = typer.Option(6, "--months", help="Months of settled bets to analyse (default: 6)."),
    bins: int = typer.Option(10, "--bins", help="Number of probability bins for the reliability curve."),
    market: str = typer.Option(None, "--market", help="Filter to a single market, e.g. match_winner."),
) -> None:
    """Model calibration: Brier score + reliability curve for settled paper bets.

    Requires bets to have been logged with `accubet track` (which stores predicted_prob).
    Calibration tells you whether the model is over- or under-confident at each probability level.
    """
    from accubet.backtest.calibration import calibration_summary
    from accubet.backtest.walkforward import months_ago
    from accubet.storage.models import TrackedBet

    cfg = get_config()
    setup_logging(cfg.secrets.log_level)

    end_d = date.today()
    start_d = months_ago(end_d, months)

    with session_scope() as session:
        q = select(TrackedBet).where(
            TrackedBet.settled == True,  # noqa: E712
            TrackedBet.kind == "single",
            TrackedBet.predicted_prob.is_not(None),
            TrackedBet.placed_at >= datetime.combine(start_d, datetime.min.time()),
        )
        if market:
            q = q.where(TrackedBet.market == market)
        bets = list(session.execute(q).scalars().all())

    if not bets:
        console.print(
            "[yellow]No settled bets with calibration data found.[/yellow]\n"
            "[dim]Calibration data is stored for bets tracked after this upgrade. "
            "Once you have settled bets, run this command again.[/dim]"
        )
        raise typer.Exit()

    probs = [b.predicted_prob for b in bets]
    outcomes = [1 if b.result == "win" else 0 for b in bets]

    summary = calibration_summary(probs, outcomes)
    bs = summary["brier_score"]
    mace = summary["mace"]
    curve = summary["curve"]

    market_label = f" [{market}]" if market else ""
    console.print(
        f"\n[bold]Calibration{market_label}[/bold]  "
        f"{summary['n']} settled bets  "
        f"period {start_d} to {end_d}"
    )
    console.print(
        f"  Brier score : [bold]{bs:.4f}[/bold]  "
        f"[dim](0 = perfect, 0.25 = random 50/50 guess)[/dim]"
    )
    console.print(
        f"  Mean abs cal error: [bold]{mace:.4f}[/bold]  "
        f"[dim](how far off mean predicted prob is from actual win rate per bin)[/dim]"
    )

    if not curve:
        raise typer.Exit()

    table = Table(title="Reliability curve (predicted prob vs actual win rate)")
    table.add_column("Bin", justify="center")
    table.add_column("Mean pred%", justify="right")
    table.add_column("Actual%", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Bar", justify="left")

    for b in curve:
        diff = b.actual_freq - b.mean_pred
        color = "green" if abs(diff) < 0.05 else ("yellow" if abs(diff) < 0.10 else "red")
        # ASCII bar for actual freq (20 chars wide)
        bar_len = round(b.actual_freq * 20)
        bar = "[green]" + "#" * bar_len + "[/green]" + "." * (20 - bar_len)
        table.add_row(
            f"{b.bin_mid:.0%}",
            f"{b.mean_pred:.1%}",
            f"{b.actual_freq:.1%}",
            f"[{color}]{diff:+.1%}[/{color}]",
            str(b.n),
            bar,
        )
    console.print(table)
    console.print(
        "[dim]Green diff = well-calibrated (<5pp), yellow = slight over/under-confidence, "
        "red = poorly calibrated (>10pp).[/dim]"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (use 0.0.0.0 for LAN)."),
    port: int = typer.Option(8080, "--port", help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)."),
) -> None:
    """Start the AccuBet web dashboard and REST API.

    Visit http://localhost:8080 to open the dashboard.
    API docs are at http://localhost:8080/api/docs.
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is not installed. Run: pip install uvicorn[standard][/red]")
        raise typer.Exit(1)
    cfg = get_config()
    setup_logging(cfg.secrets.log_level)
    _init_db()
    console.print(f"[cyan]AccuBet dashboard[/cyan]  http://{host}:{port}")
    console.print(f"[dim]API docs: http://{host}:{port}/api/docs[/dim]")
    uvicorn.run(
        "accubet.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=cfg.secrets.log_level.lower(),
    )


if __name__ == "__main__":
    app()
