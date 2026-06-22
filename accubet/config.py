"""Configuration loader.

Two sources, merged into a single immutable ``AppConfig``:

* **Secrets** (API key, DB URL, log level) come from the environment / ``.env`` file
  via :class:`Secrets`. These never live in version control.
* **Tunables** (thresholds, weights, leagues, staking, accumulator rules) come from
  ``config/settings.yaml`` and are safe to commit.

Use :func:`get_config` everywhere; it is cached so the files are read once.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `accubet` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Load .env once, as early as possible, so os.environ is populated for Secrets.
load_dotenv(PROJECT_ROOT / ".env")


class Secrets(BaseSettings):
    """Secret / environment-driven configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    apifootball_key: str = Field(default="", alias="APIFOOTBALL_KEY")
    apifootball_base_url: str = Field(
        default="https://v3.football.api-sports.io", alias="APIFOOTBALL_BASE_URL"
    )
    database_url: str = Field(
        default="sqlite:///data/accubet.db", alias="ACCUBET_DATABASE_URL"
    )
    log_level: str = Field(default="INFO", alias="ACCUBET_LOG_LEVEL")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")


# --- YAML-backed tunables (validated, with sane defaults so the app runs bare) ---


class League(BaseModel):
    id: int
    name: str = ""
    scope: str = "global"  # "global" | "domestic"


class ApiFootballCfg(BaseModel):
    daily_request_limit: int = 100
    quota_soft_stop: int = 90
    rate_limit_per_minute: int = 10
    cache_ttl_minutes: int = 720
    odds_refresh_per_match: int = 3


class ValueCfg(BaseModel):
    min_ev: float = 0.05
    min_confidence: float = 0.70
    min_books_for_consensus: int = 3


class ScanCfg(BaseModel):
    """Which markets/lines the scan considers by default (liquidity filter).

    Restricting to the most liquid, efficiently-priced markets removes the exotic-line
    noise (e.g. Over 6.5) that otherwise dominates a raw EV ranking.
    """

    markets: list[str] = ["match_winner", "over_under", "btts"]
    ou_lines: list[float] = [2.5]


class ModelWeights(BaseModel):
    market: float = 0.55
    glicko: float = 0.15
    goals: float = 0.10
    form: float = 0.10
    ml: float = 0.10


class StakingCfg(BaseModel):
    kelly_fraction: float = 0.25
    max_stake_per_event: float = 0.03
    max_daily_exposure: float = 0.15
    starting_bankroll: float = 1000.0


class TierCfg(BaseModel):
    min_legs: int = 2
    max_legs: int = 4
    min_combined_odds: float = 3.0
    max_combined_odds: float = 5.0
    target_strike: float | None = None


class AccumulatorCfg(BaseModel):
    banker: TierCfg = TierCfg(
        min_legs=2, max_legs=3, min_combined_odds=1.3, max_combined_odds=1.8, target_strike=0.70
    )
    value: TierCfg = TierCfg(
        min_legs=2, max_legs=4, min_combined_odds=3.0, max_combined_odds=5.0
    )
    modes: dict[str, dict] = Field(default_factory=dict)


class TrackingCfg(BaseModel):
    top_singles_per_day: int = 10
    paper_stake: float = 10.0


class LocalBookCfg(BaseModel):
    name: str = "Betway Malawi"
    base_url: str = "https://www.betway.co.mw"
    enabled: bool = True


class AppConfig(BaseModel):
    """The fully-resolved configuration object passed around the app."""

    secrets: Secrets
    apifootball: ApiFootballCfg = ApiFootballCfg()
    leagues: list[League] = Field(default_factory=list)
    markets: list[str] = Field(default_factory=list)
    value: ValueCfg = ValueCfg()
    scan: ScanCfg = ScanCfg()
    model_weights: ModelWeights = ModelWeights()
    staking: StakingCfg = StakingCfg()
    accumulator: AccumulatorCfg = AccumulatorCfg()
    tracking: TrackingCfg = TrackingCfg()
    local_book: LocalBookCfg = LocalBookCfg()

    # convenience -----------------------------------------------------------
    @property
    def league_ids(self) -> list[int]:
        return [lg.id for lg in self.leagues]

    def data_dir(self) -> Path:
        d = PROJECT_ROOT / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache(maxsize=1)
def get_config(settings_path: str | None = None) -> AppConfig:
    """Load and cache the merged configuration."""
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return AppConfig(
        secrets=Secrets(),
        apifootball=ApiFootballCfg(**raw.get("apifootball", {})),
        leagues=[League(**lg) for lg in raw.get("leagues", [])],
        markets=raw.get("markets", []),
        value=ValueCfg(**raw.get("value", {})),
        scan=ScanCfg(**raw.get("scan", {})),
        model_weights=ModelWeights(**raw.get("model_weights", {})),
        staking=StakingCfg(**raw.get("staking", {})),
        accumulator=AccumulatorCfg(**raw.get("accumulator", {})),
        tracking=TrackingCfg(**raw.get("tracking", {})),
        local_book=LocalBookCfg(**raw.get("local_book", {})),
    )
