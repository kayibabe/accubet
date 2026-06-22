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
    """Upcoming matches that have a consensus (the scan/track universe)."""
    return list(session.execute(
        select(Match.id)
        .join(Consensus, Consensus.match_id == Match.id)
        .where(Match.status.in_(("NS", "TBD", "PST")))
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
        connector = BetwayMalawiConnector(cfg) if cfg.local_book.enabled else None

        for d in _dates(date_, days):
            matches, report = ingest_fixtures(session, cfg, client, d, force=force)
            if not no_odds and matches:
                ingest_odds(session, cfg, client, matches, report)
            local_n = 0
            if connector and matches:
                local_n = ingest_local_odds(session, connector, matches)
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

        for i, o in enumerate(shown, 1):
            mark = "[green]*[/green]" if o._passes else ""
            line = f" {o.line}" if o.line is not None else ""
            table.add_row(
                f"{i}{mark}",
                f"{o.home} v {o.away}",
                o.market,
                f"{o.selection}{line}",
                f"{o.fair_prob:.0%}",
                "ens" if o.prob_source == "ensemble" else "mkt",
                f"{o.price:.2f}",
                o.price_source,
                f"{o.ev*100:+.1f}",
                f"{o.value_pct*100:+.1f}",
                f"{o.confidence:.2f}",
                str(o.n_books),
            )
        console.print(table)
        console.print(
            f"[dim]{len(opps)} opportunities | {sum(o._passes for o in opps)} pass gates | "
            f"{stored} saved to value_bets.[/dim]"
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
            title=f"Walk-forward  {start_d} → {end_d}  (window={window}d)",
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


if __name__ == "__main__":
    app()
