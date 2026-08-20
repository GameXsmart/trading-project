# Crypto Market Intelligence Engine (MIE)

An analytical engine that observes cryptocurrency markets, evaluates evidence, and
produces **probabilistic, uncertainty-bearing** assessments of market state.

> **This is not a trading bot.** It has no order-execution path, holds no trading
> keys, and never will. It produces scenarios and probabilities for a human to
> interpret — never guarantees, and never investment advice.

**Status: Phase 1 (market data ingestion + database) is complete and tested.**
The full architecture is designed in [`ARCHITECTURE.md`](ARCHITECTURE.md); the
remaining phases are specified in [`docs/PHASES.md`](docs/PHASES.md) and not yet
built. Nothing in this repo pretends to be further along than it is.

---

## What Phase 1 delivers

A production-shaped ingestion layer that keeps a multi-asset, multi-timeframe market
history correct and current, and — critically — **knows when it cannot be trusted**.

| Capability | Detail |
|---|---|
| **Multi-source ingestion** | Binance, Coinbase, Kraken (candles) and CoinGecko (market-wide aggregates), all public read-only endpoints, no API keys. |
| **Automatic failover** | Priority-ordered, with per-provider rate limiting and circuit breakers. A dead API costs one skipped check, not a stalled pipeline. |
| **10 assets × 9 timeframes** | BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT — extensible via `config/assets.yaml`; 1m → 1w. |
| **Derivatives context** | Funding rates and open interest, the leverage-positioning inputs the regime model needs. |
| **Ten quality checks** | Shape, grid alignment, duplicates, ordering, gaps, outliers, impossible moves, staleness, flatlines, cross-source discrepancy. |
| **Trust scoring** | Every (source, asset, timeframe) carries a rolling 0–1 score that later phases multiply into published confidence. |
| **Look-ahead defence** | The forming bar is stored as `is_final = false` and excluded from every query by default. |
| **Full provenance** | Every row records its source and ingestion time; every job records what it fetched and how it went. |

Verified against live exchange APIs: **150,000+ candles ingested across 30 series at
100% grid completeness, with zero quality events.**

---

## Quick start

Requires Python 3.12+. No database server, no Docker, nothing else.

```bash
python -m venv .venv && ./.venv/Scripts/activate
```

```bash
pip install -e ".[dev]"
```

```bash
mie db init
```

```bash
mie backfill BTC 1h --days 90
```

```bash
mie status
```

That is a working install: SQLite in `./data/`, ten assets registered, and 90 days of
hourly BTC history validated and stored.

### Running it for real

```bash
mie backfill-all --timeframes 1d,4h,1h
```

```bash
mie run
```

`run` starts every loop — live polling on bar boundaries, derivatives, global
metrics, and quality scoring — until you stop it with Ctrl-C.

---

## Commands

| Command | Purpose |
|---|---|
| `mie db init` | Create the schema and register assets and sources. Idempotent. |
| `mie db info` | Show the resolved database target and connectivity. |
| `mie db reset --yes` | Drop and recreate everything. Destroys stored data. |
| `mie providers` | Probe provider health, latency, and capabilities. |
| `mie assets` | List the configured observation universe. |
| `mie backfill BTC 1h --days 90` | Backfill one series. |
| `mie backfill-all --timeframes 1d,4h` | Backfill the whole asset × timeframe matrix. |
| `mie poll --once` | Run a single live-poll tick. |
| `mie run` | Run the full ingestion service. |
| `mie status` | Coverage, completeness, and freshness per series. |
| `mie quality --hours 24` | Recent quality events and trust scores. |
| `mie audit BTC 1h` | Compare providers against each other for the same window. |

---

## How it is put together

```
providers/  →  quality/  →  storage/  →  core/events  →  (Phase 2+)
 failover      validate     Timescale     candle.closed    features
 throttle      score        SQLite                         models
 breakers                                                  predictions
```

Four ideas carry most of the weight:

**1. Data quality gates confidence, rather than being logged and ignored.**
Structurally impossible bars (`high < low`, off-grid timestamps) are *rejected*;
suspicious-but-possible ones (a 20% hourly move, a gap, a flat run) are *flagged and
kept*, because discarding them would fabricate a calmer market than the real one.
Flags accumulate into a per-series trust score. When Phase 7 publishes a prediction,
that score multiplies its confidence — so degraded inputs produce a quieter system,
not a confidently wrong one.

**2. The forming bar is never treated as history.**
The candle covering the current moment is incomplete. Storing it as final is how
look-ahead bias silently enters a live pipeline, so it is written with
`is_final = false` and every query excludes it unless a caller explicitly opts in.

**3. Failover is never silent.**
When the primary provider fails, the next one serves the request *and* a
`PROVIDER_FAILOVER` quality event is recorded. "The data arrived, but from the
third-choice venue" is information the confidence layer needs.

**4. Thresholds are measured, not guessed.**
The outlier threshold is set at 25 robust sigma because measurement on real BTC data
showed that z > 10 fires on 0.2–0.4% of perfectly normal bars (crypto returns are
fat-tailed) while nothing at all exceeded z > 30. A detector that cries wolf on
ordinary volatility is worse than no detector.

---

## Configuration

Three layers, lowest priority first:

1. defaults in the typed models (`src/mie/config/settings.py`)
2. `config/default.yaml` and `config/assets.yaml` — **checked in, never secrets**
3. environment variables / `.env` — `MIE_` prefix, `__` nests

```bash
MIE_DATABASE__URL=postgresql+asyncpg://mie:pw@localhost:5432/mie
```

Copy `.env.example` to `.env` to start. No API keys are required — every Phase 1
provider uses public read-only endpoints.

### Production storage

SQLite is the zero-infrastructure default. For real use, point at TimescaleDB:

```bash
pip install -e ".[postgres]"
```

Set `MIE_DATABASE__URL` to a `postgresql+asyncpg://` URL and run `mie db init`. The
hypertables, compression policies, and continuous aggregates in
[`sql/timescale.sql`](sql/timescale.sql) are applied automatically, and skipped
gracefully on a plain PostgreSQL without the extension.

---

## Tests

```bash
pytest
```

164 tests, no network and no infrastructure required — ingestion, validation and
failover run against a deterministic synthetic provider with injectable faults
(gaps, duplicates, malformed bars, price spikes, outages).

A separate suite verifies the live provider contracts and is excluded by default,
because a test that needs an exchange to be up is a test that will eventually fail
for reasons unrelated to the code:

```bash
pytest -m network
```

---

## What is deliberately not here

Phases 2–12 — features, regime detection, pattern and sequence discovery, news
intelligence, the model ensemble, backtesting, the learning loop, the dashboard, and
alerts — are **designed** in [`ARCHITECTURE.md`](ARCHITECTURE.md) and **not
implemented**. Their directories do not exist rather than containing stubs: an empty
module pretending to be a model is worse than no module.

The prediction contract, the multi-timeframe agreement model, and the learning loop
are specified now because they constrain Phase 1's schema and event design — not
because any of them run yet.

---

## Safety

- No order execution, no trading keys, no withdrawal permissions. Providers use
  public read-only endpoints.
- Outputs are probabilistic scenarios with explicit confidence and invalidation
  conditions, never guarantees.
- **Nothing here is investment advice.** Cryptocurrency markets are volatile and
  adversarial; a well-calibrated model is still wrong a great deal of the time, and
  the system is built to say "insufficient evidence" rather than guess.

---

## License

MIT — see [`pyproject.toml`](pyproject.toml).
