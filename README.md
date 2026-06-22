# AccuBet — Football Value-Betting Intelligence Engine

[![CI](https://github.com/kayibabe/accubet/actions/workflows/ci.yml/badge.svg)](https://github.com/kayibabe/accubet/actions/workflows/ci.yml)

A **modeling-first** engine that finds *value bets* — spots where bookmaker odds misprice
the true probability of an outcome — and only flags a bet when its **expected value** clears
a threshold. It is **not** a winner predictor: the edge comes from probability
miscalibration, with market odds as the primary signal (~55% weight) and internal models
hunting small, exploitable disagreements. **Closing Line Value (CLV)** is the real scoreboard.

> ⚠️ Betting involves risk. No model guarantees profit. AccuBet auto-tracks **paper bets**
> for analysis — it never places real wagers. Use responsibly and within the law.

## Status

Phase 0–5 complete (ingestion → market intelligence → models → value/tracking → backtest).
No UI yet — everything runs from the CLI against a local SQLite database.

> Branch protection (GitHub Settings → Branches → main): require CI to pass before merge,
> require at least one review, disallow force-pushes.

## Data sources

| Role | Source |
|---|---|
| Fixtures + world-bookmaker odds + stats | **API-Football** (free tier, 100 req/day) |
| Local execution price (what you'd actually bet) | **Betway Malawi** (scraped connector) |

A *value bet* = Betway's price beats the API-derived fair probability.

## Quick start

```powershell
# 1. Create a virtual environment and install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 2. Configure secrets
copy .env.example .env
#   then edit .env and set APIFOOTBALL_KEY=...

# 3. Initialise the database
.\.venv\Scripts\accubet.exe init-db

# 4. Ingest fixtures + odds (cache-first; re-runs spend ~0 quota)
.\.venv\Scripts\accubet.exe ingest

# 4b. (one-time per league) Pull season history to train the models (free tier: 2022-2024)
.\.venv\Scripts\accubet.exe history --league 39 --season 2024

# 4c. Run the model ensemble -> true-probability predictions for upcoming matches
.\.venv\Scripts\accubet.exe predict

# 5. Scan for value (ensemble true prob vs Betway; market fallback where no history)
.\.venv\Scripts\accubet.exe scan

# 6. Auto-track top-10 EV singles + both accumulator tiers as PAPER bets
.\.venv\Scripts\accubet.exe track

# 7. After matches finish: settle paper bets and review performance
.\.venv\Scripts\accubet.exe ingest --no-odds --force   # refresh results
.\.venv\Scripts\accubet.exe settle
.\.venv\Scripts\accubet.exe report                     # per-market + overall ROI/win-rate

# 8. Full daily pipeline (ingest + predict + track + settle in one shot)
.\.venv\Scripts\accubet.exe daily
.\.venv\Scripts\accubet.exe daily --back 2 --days 3   # wider window
.\.venv\Scripts\accubet.exe daily --no-settle          # skip grading step

# Schedule it (Windows Task Scheduler — run once per day at 07:00)
# schtasks /create /tn "AccuBet Daily" /tr "D:\WebApps\accubet\.venv\Scripts\accubet.exe daily" /sc daily /st 07:00

# 9. Walk-forward backtest — rolling 30-day windows over the last 6 months
.\.venv\Scripts\accubet.exe backtest
.\.venv\Scripts\accubet.exe backtest --months 3 --window 14    # fortnightly, last quarter
.\.venv\Scripts\accubet.exe backtest --start 2024-08-01 --end 2025-01-31

# Check remaining API quota for today
.\.venv\Scripts\accubet.exe quota
```

## Configuration

* **Secrets** → `.env` (git-ignored). Never commit your API key.
* **Tunables** → `config/settings.yaml` (thresholds, weights, leagues, staking,
  accumulator rules) — editable without touching code.

## Layout

```
accubet/
  config.py · logging_setup.py · cli.py
  ingestion/   apifootball · cache · quota · normalize · scrapers/betway
  storage/     db · models
  market/      consensus · comparison · movement · clv · efficiency
  models/      goals_poisson · ratings_glicko · form · ensemble · predictor
  value/       ev · accumulator · staking
  tracking/    tracked_bets · performance
  backtest/    walkforward · metrics   (Phase 5 — CLV, Sharpe, drawdown, rolling windows)
```
