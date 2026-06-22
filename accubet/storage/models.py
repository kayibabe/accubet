"""SQLAlchemy ORM models — the AccuBet relational schema.

Design notes
------------
* ``odds_snapshots`` carries a ``source`` ("apifootball" | "betway") so the same table
  feeds movement tracking, CLV, and the Betway-vs-fair comparison.
* ``consensus`` stores vig-removed *fair* probabilities derived from world books.
* ``predictions`` holds one row per (match, market, selection, model); the ensemble's
  final number is stored with ``model = 'ensemble'``.
* ``tracked_bets`` are *paper* bets auto-logged for performance analysis; ``bets`` are
  real wagers (kept separate so paper results never touch the real bankroll).
* ``api_cache`` + ``request_log`` back the cache-first / quota-guard ingestion.
"""

from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_league_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160), default="")
    country: Mapped[str] = mapped_column(String(80), default="")
    scope: Mapped[str] = mapped_column(String(16), default="global")  # global|domestic
    season: Mapped[int | None] = mapped_column(Integer, nullable=True)

    matches: Mapped[list["Match"]] = relationship(back_populates="competition")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_team_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    country: Mapped[str] = mapped_column(String(80), default="")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_fixture_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"), index=True)
    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    kickoff: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    venue: Mapped[str] = mapped_column(String(160), default="")
    status: Mapped[str] = mapped_column(String(24), default="NS")  # NS, LIVE, FT...
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    competition: Mapped["Competition"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    odds: Mapped[list["OddsSnapshot"]] = relationship(back_populates="match")


class OddsSnapshot(Base):
    """A single observed price. Many rows per match (sources, books, time)."""

    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    source: Mapped[str] = mapped_column(String(24), index=True)  # apifootball|betway
    bookmaker: Mapped[str] = mapped_column(String(60), default="")
    market: Mapped[str] = mapped_column(String(40), index=True)  # match_winner, btts...
    selection: Mapped[str] = mapped_column(String(40))           # home, draw, away, yes...
    line: Mapped[float | None] = mapped_column(Float, nullable=True)  # e.g. O/U 2.5
    price: Mapped[float] = mapped_column(Float)                  # decimal odds
    captured_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    match: Mapped["Match"] = relationship(back_populates="odds")


class Consensus(Base):
    """Vig-removed fair probability across world bookmakers, per selection."""

    __tablename__ = "consensus"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    market: Mapped[str] = mapped_column(String(40), index=True)
    selection: Mapped[str] = mapped_column(String(40))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    fair_prob: Mapped[float] = mapped_column(Float)
    consensus_odds: Mapped[float] = mapped_column(Float)   # fair (de-vigged) odds
    best_odds: Mapped[float | None] = mapped_column(Float, nullable=True)  # best available
    n_books: Mapped[int] = mapped_column(Integer, default=0)
    overround: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Prediction(Base):
    """One model's probability for a selection. model='ensemble' is the final."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    market: Mapped[str] = mapped_column(String(40), index=True)
    selection: Mapped[str] = mapped_column(String(40))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    model: Mapped[str] = mapped_column(String(24), index=True)  # market|glicko|ensemble..
    prob: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ValueBet(Base):
    __tablename__ = "value_bets"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    market: Mapped[str] = mapped_column(String(40), index=True)
    selection: Mapped[str] = mapped_column(String(40))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    book: Mapped[str] = mapped_column(String(60), default="")  # where best/exec price is
    odds: Mapped[float] = mapped_column(Float)                 # price we'd take
    true_prob: Mapped[float] = mapped_column(Float)            # ensemble / fair prob
    implied_prob: Mapped[float] = mapped_column(Float)         # 1/odds
    ev: Mapped[float] = mapped_column(Float, index=True)
    value_pct: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed_gates: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Accumulator(Base):
    __tablename__ = "accumulators"

    id: Mapped[int] = mapped_column(primary_key=True)
    tier: Mapped[str] = mapped_column(String(16), index=True)  # banker|value
    mode: Mapped[str] = mapped_column(String(16), default="balanced")
    combined_odds: Mapped[float] = mapped_column(Float)
    combined_prob: Mapped[float] = mapped_column(Float)
    expected_return: Mapped[float] = mapped_column(Float)  # combined EV
    risk_rating: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    legs: Mapped[list["AccumulatorLeg"]] = relationship(
        back_populates="accumulator", cascade="all, delete-orphan"
    )


class AccumulatorLeg(Base):
    __tablename__ = "accumulator_legs"

    id: Mapped[int] = mapped_column(primary_key=True)
    accumulator_id: Mapped[int] = mapped_column(ForeignKey("accumulators.id"), index=True)
    value_bet_id: Mapped[int | None] = mapped_column(ForeignKey("value_bets.id"), nullable=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    market: Mapped[str] = mapped_column(String(40))
    selection: Mapped[str] = mapped_column(String(40))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds: Mapped[float] = mapped_column(Float)
    prob: Mapped[float] = mapped_column(Float)

    accumulator: Mapped["Accumulator"] = relationship(back_populates="legs")


class TrackedBet(Base):
    """Auto-logged PAPER bet (no real money) for performance measurement."""

    __tablename__ = "tracked_bets"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # single|accumulator
    ref_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # value_bet/acc id
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"), nullable=True)
    market: Mapped[str] = mapped_column(String(40), index=True, default="")
    selection: Mapped[str] = mapped_column(String(40), default="")
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    book: Mapped[str] = mapped_column(String(60), default="")
    odds: Mapped[float] = mapped_column(Float)
    ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    stake: Mapped[float] = mapped_column(Float, default=0.0)
    placed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    settled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    result: Mapped[str] = mapped_column(String(12), default="pending")  # win|loss|void
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_prob: Mapped[float | None] = mapped_column(Float, nullable=True)


class Bet(Base):
    """Real wager (manually confirmed). Kept separate from paper TrackedBet."""

    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"), nullable=True)
    market: Mapped[str] = mapped_column(String(40), default="")
    selection: Mapped[str] = mapped_column(String(40), default="")
    book: Mapped[str] = mapped_column(String(60), default="")
    stake: Mapped[float] = mapped_column(Float)
    odds: Mapped[float] = mapped_column(Float)
    placed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    settled: Mapped[bool] = mapped_column(Boolean, default=False)
    result: Mapped[str] = mapped_column(String(12), default="pending")
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True, index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="FT")
    settled_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class BankrollLog(Base):
    __tablename__ = "bankroll_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    balance: Mapped[float] = mapped_column(Float)
    change: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(String(120), default="")


class ApiCache(Base):
    """Cache-first store. One row per (endpoint, params)."""

    __tablename__ = "api_cache"
    __table_args__ = (UniqueConstraint("endpoint", "params_hash", name="uq_cache_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(120), index=True)
    params_hash: Mapped[str] = mapped_column(String(64), index=True)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    payload: Mapped[dict] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class RequestLog(Base):
    """One row per real API call — backs the daily quota counter."""

    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    day: Mapped[date] = mapped_column(Date, index=True)
    endpoint: Mapped[str] = mapped_column(String(120), default="")
    success: Mapped[bool] = mapped_column(Boolean, default=True)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    params_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
