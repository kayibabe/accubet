"""Ingest orchestration: upsert fixtures/teams/competitions, then pull & store odds.

Cache-first + diff-before-pull means a re-run on an unchanged slate writes nothing new and
spends zero quota: fixtures come from cache, per-fixture odds come from cache, and we only
persist fresh odds snapshots when a real network pull happened (so movement is captured
over time without duplicating rows on every run).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from accubet.config import AppConfig
from accubet.ingestion.apifootball import ApiFootballClient
from accubet.ingestion.normalize import iter_normalized_odds, parse_fixture
from accubet.ingestion.quota import QuotaExceeded
from accubet.logging_setup import get_logger
from accubet.storage.models import Competition, Match, OddsSnapshot, Result, Team

_FINISHED_STATUSES = {"FT", "AET", "PEN"}

log = get_logger(__name__)


@dataclass
class IngestReport:
    date: str
    fixtures_seen: int = 0
    matches_tracked: int = 0
    new_matches: int = 0
    odds_pulled: int = 0
    odds_from_cache: int = 0


# --- upserts ---------------------------------------------------------------

def _upsert_competition(session: Session, fd: dict, scope: str) -> Competition | None:
    if fd["league_id"] is None:
        return None
    comp = session.execute(
        select(Competition).where(Competition.api_league_id == fd["league_id"])
    ).scalar_one_or_none()
    if comp is None:
        comp = Competition(api_league_id=fd["league_id"])
        session.add(comp)
    comp.name = fd["league_name"] or comp.name
    comp.country = fd["country"] or comp.country
    comp.scope = scope
    comp.season = fd["season"] or comp.season
    session.flush()
    return comp


def _upsert_team(session: Session, api_id: int | None, name: str, country: str) -> Team | None:
    if api_id is None:
        return None
    team = session.execute(
        select(Team).where(Team.api_team_id == api_id)
    ).scalar_one_or_none()
    if team is None:
        team = Team(api_team_id=api_id)
        session.add(team)
    team.name = name or team.name
    team.country = country or team.country
    session.flush()
    return team


def _upsert_match(session: Session, fd: dict, comp, home, away) -> tuple[Match | None, bool]:
    if fd["fixture_id"] is None:
        return None, False
    match = session.execute(
        select(Match).where(Match.api_fixture_id == fd["fixture_id"])
    ).scalar_one_or_none()
    created = match is None
    if match is None:
        match = Match(api_fixture_id=fd["fixture_id"])
        session.add(match)
    match.competition_id = comp.id if comp else None
    match.home_team_id = home.id if home else None
    match.away_team_id = away.id if away else None
    match.kickoff = fd["kickoff"]
    match.venue = fd["venue"]
    match.status = fd["status"]
    session.flush()
    return match, created


def _maybe_store_result(session: Session, match: Match, fd: dict) -> None:
    """Record the final score for finished fixtures (from the same cached payload)."""
    if fd["status"] not in _FINISHED_STATUSES:
        return
    if fd["home_goals"] is None or fd["away_goals"] is None:
        return
    res = session.execute(
        select(Result).where(Result.match_id == match.id)
    ).scalar_one_or_none()
    if res is None:
        res = Result(match_id=match.id)
        session.add(res)
    res.home_goals = fd["home_goals"]
    res.away_goals = fd["away_goals"]
    res.status = fd["status"]
    session.flush()


# --- ingest steps ----------------------------------------------------------

def ingest_fixtures(
    session: Session, cfg: AppConfig, client: ApiFootballClient, date_str: str, *, force: bool = False
) -> tuple[list[Match], IngestReport]:
    league_scope = {lg.id: lg.scope for lg in cfg.leagues}
    league_ids = set(league_scope)
    report = IngestReport(date=date_str)

    items = client.fixtures_by_date(date_str, force=force)
    report.fixtures_seen = len(items)

    matches: list[Match] = []
    for item in items:
        fd = parse_fixture(item)
        if fd["league_id"] not in league_ids:
            continue
        comp = _upsert_competition(session, fd, league_scope.get(fd["league_id"], "global"))
        home = _upsert_team(session, fd["home_id"], fd["home_name"], fd["country"])
        away = _upsert_team(session, fd["away_id"], fd["away_name"], fd["country"])
        match, created = _upsert_match(session, fd, comp, home, away)
        if match is None:
            continue
        report.new_matches += int(created)
        _maybe_store_result(session, match, fd)
        matches.append(match)

    report.matches_tracked = len(matches)
    return matches, report


def ingest_odds(
    session: Session, cfg: AppConfig, client: ApiFootballClient, matches: list[Match], report: IngestReport
) -> IngestReport:
    for match in matches:
        try:
            resp = client.odds_for_fixture(match.api_fixture_id)
        except QuotaExceeded as exc:
            log.warning("%s — stopping odds pulls for today.", exc)
            break

        if client.last_from_cache:
            report.odds_from_cache += 1
            continue

        wrote = 0
        for item in resp:
            for row in iter_normalized_odds(item):
                session.add(OddsSnapshot(match_id=match.id, source="apifootball", **row))
                wrote += 1
        if wrote:
            report.odds_pulled += 1
        session.flush()
    return report


def ingest_local_odds(session: Session, connector, matches: list[Match]) -> int:
    """Store local-book (e.g. Betway) odds as snapshots. Deduped so repeated runs of a
    static manual file don't pile up identical rows."""
    stored = 0
    for match in matches:
        for row in connector.fetch_match_odds(match):
            if not row.get("market") or not row.get("selection") or not row.get("price"):
                continue
            exists = session.execute(
                select(OddsSnapshot.id).where(
                    OddsSnapshot.match_id == match.id,
                    OddsSnapshot.source == "betway",
                    OddsSnapshot.market == row["market"],
                    OddsSnapshot.selection == row["selection"],
                    OddsSnapshot.line == row.get("line"),
                    OddsSnapshot.price == row["price"],
                )
            ).first()
            if exists:
                continue
            session.add(
                OddsSnapshot(
                    match_id=match.id,
                    source="betway",
                    bookmaker=row.get("bookmaker", connector.name),
                    market=row["market"],
                    selection=row["selection"],
                    line=row.get("line"),
                    price=row["price"],
                )
            )
            stored += 1
        session.flush()
    return stored


def ingest_history(
    session: Session, cfg: AppConfig, client: ApiFootballClient, league_id: int, season: int
) -> dict:
    """Pull a whole league-season's fixtures (one API call) and store finished results —
    the training corpus for the Poisson / Glicko / form models."""
    scope = next((lg.scope for lg in cfg.leagues if lg.id == league_id), "global")
    items = client.response(client.get("fixtures", {"league": league_id, "season": season}))
    stored = results = 0
    for item in items:
        fd = parse_fixture(item)
        comp = _upsert_competition(session, fd, scope)
        home = _upsert_team(session, fd["home_id"], fd["home_name"], fd["country"])
        away = _upsert_team(session, fd["away_id"], fd["away_name"], fd["country"])
        match, _ = _upsert_match(session, fd, comp, home, away)
        if match is None:
            continue
        _maybe_store_result(session, match, fd)
        stored += 1
        if fd["status"] in _FINISHED_STATUSES and fd["home_goals"] is not None:
            results += 1
    return {"fixtures": len(items), "stored": stored, "results": results}


def run_ingest(
    session: Session,
    cfg: AppConfig,
    client: ApiFootballClient,
    date_str: str,
    *,
    with_odds: bool = True,
    force: bool = False,
    local_connector=None,
) -> IngestReport:
    matches, report = ingest_fixtures(session, cfg, client, date_str, force=force)
    if with_odds and matches:
        ingest_odds(session, cfg, client, matches, report)
    if local_connector is not None and matches:
        ingest_local_odds(session, local_connector, matches)
    return report
