# Architecture — Crypto Market Intelligence & Prediction Engine (MIE)

> **This is an analytical system, not a trading bot.** It produces probabilistic,
> uncertainty-bearing assessments of market state. It has no order-execution path
> and never will — see [Safety boundary](#12-safety-boundary).

---

## 1. Design principles

| # | Principle | Consequence in the code |
|---|-----------|-------------------------|
| 1 | **Evidence over cleverness** | Every prediction carries the evidence *and* counter-evidence that produced it. A model that cannot explain itself does not ship. |
| 2 | **Simplest thing that measurably wins** | Model complexity must be justified by walk-forward performance against a naive baseline (persistence / random walk). ML is not a goal. |
| 3 | **Uncertainty is a first-class output** | No component returns a point estimate without a distribution or confidence. `Insufficient evidence` is a valid, expected answer. |
| 4 | **Data quality gates confidence** | Degraded inputs *reduce* published confidence rather than being silently ignored. The quality score flows all the way into the ensemble. |
| 5 | **No look-ahead, ever** | Feature computation is a pure function of data whose `open_time + timeframe <= as_of`. Enforced structurally, not by convention. |
| 6 | **Everything is evaluated** | Predictions are stored *before* the outcome is knowable, scored automatically when the horizon expires, and those scores feed model weighting. |
| 7 | **Pluggable everything** | Providers, storage, event bus, and models sit behind interfaces. One dead API must not take down the system. |
| 8 | **Incremental compute** | A new candle triggers incremental feature updates, never a full historical recompute. |

---

## 2. System topology

```
                    ┌──────────────────────────────────────────┐
                    │            DATA COLLECTORS               │
                    │  Binance · Coinbase · Kraken · CoinGecko │
                    │  derivatives (funding/OI) · on-chain     │
                    └────────────────────┬─────────────────────┘
                                         │  raw envelopes
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │      PROVIDER MANAGER (failover)         │
                    │  rate limit · circuit breaker · priority │
                    └────────────────────┬─────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │   VALIDATION & DATA-QUALITY ENGINE       │
                    │  shape · grid · dupes · gaps · outliers  │
                    │  staleness · cross-source discrepancy    │
                    └──────────┬──────────────────┬────────────┘
                               │ clean            │ quality events
                               ▼                  ▼
    ┌──────────────────────────────────┐   ┌────────────────────┐
    │   TIME-SERIES STORE (Timescale)  │   │ QUALITY SCORE STORE│
    │   ohlcv · funding · OI · global  │   │ per src/asset/tf   │
    └──────────┬───────────────────────┘   └─────────┬──────────┘
               │  CandleClosed events                │
               ▼                                     │
    ┌──────────────────────────────────┐             │
    │        EVENT BUS (async)         │◀────────────┘
    │  in-proc → NATS/Redpanda at scale│
    └──────────┬───────────────────────┘
               ▼
┌───────────────────────────────────────────────────────────────┐
│                       ANALYTICAL LAYERS                       │
│  Phase 2  Feature engine        (indicators, incremental)     │
│  Phase 3  Multi-timeframe state (hierarchical, agreement)     │
│  Phase 4  Pattern + sequence    (motifs, HMM, similarity)     │
│  Phase 5  News intelligence     (dedupe, relevance, impact)   │
│  Phase 6  Models A-H            (independent predictors)      │
│  Phase 7  Ensemble + calibration                              │
└───────────────────────────────┬───────────────────────────────┘
                                ▼
    ┌──────────────────────────────────────────────┐
    │  PREDICTION STORE  (immutable, append-only)  │
    └──────────┬───────────────────────┬───────────┘
               │                       │ horizon expiry
               ▼                       ▼
    ┌────────────────────┐   ┌──────────────────────────────┐
    │  API (FastAPI)     │   │ EVALUATION / LEARNING LOOP   │
    │  + WebSocket push  │   │ score → calibrate → reweight │
    └──────────┬─────────┘   └──────────────┬───────────────┘
               ▼                            │ weights, calibration maps
    ┌────────────────────┐                  │
    │ DASHBOARD (Next.js)│◀─────────────────┘
    │ ALERTS (multi-ch)  │
    └────────────────────┘
```

---

## 3. Language & technology choices

Chosen per workload, not per fashion.

| Concern | Choice | Why |
|---|---|---|
| Ingestion, features, ML, evaluation | **Python 3.12+ (async)** | The entire quantitative and ML ecosystem lives here. Ingestion is I/O-bound, so `asyncio` — not threads — is the right concurrency model. |
| Time-series storage | **PostgreSQL + TimescaleDB** | Hypertables, continuous aggregates, compression and native `time_bucket` give multi-timeframe rollups in the database. Ordinary SQL keeps the system debuggable. |
| Dev/test storage | **SQLite (aiosqlite)** | The repo must be runnable and testable with zero infrastructure. Same SQLAlchemy models; dialect-specific extras are skipped. |
| Cache / hot state | **Redis** (Phase 12) | Latest candle, feature vectors, shared rate-limit buckets across processes. |
| Event bus | **In-process asyncio → NATS** | Start in-proc (one deployable, zero ops). The `EventBus` interface makes the swap mechanical when processes split. |
| Models | **statsmodels / scikit-learn / LightGBM**; PyTorch only if it wins | Principle 2. Tabular financial data with low signal-to-noise rarely rewards deep nets. |
| API | **FastAPI** | Async-native, typed, generates the schema the dashboard consumes. |
| Dashboard | **Next.js + TypeScript + Tailwind + Recharts** | Phase 10. Server components for heavy panels, WebSocket for live state. |
| Rust / Go | **Deliberately not used yet** | Nothing in the current workload is CPU-bound enough to justify a second toolchain. Revisit if order-book microstructure (Phase 12) becomes the bottleneck. |

---

## 4. Repository layout

```
.
├── src/mie/
│   ├── core/          # types, timeframes, logging, errors, event bus
│   ├── config/        # typed settings, YAML + env layering
│   ├── storage/       # SQLAlchemy models, engine, repositories
│   ├── providers/     # provider interface + implementations + manager
│   ├── ingestion/     # backfill, live poller, orchestration service
│   ├── quality/       # validators, anomaly detection, scoring
│   ├── features/      # incremental indicator engine + market structure
│   ├── state/         # multi-timeframe market state + regime
│   ├── patterns/      # (Phase 4) pattern + sequence discovery
│   ├── news/          # (Phase 5) news ingestion + impact model
│   ├── models/        # (Phase 6) predictors A-H
│   ├── ensemble/      # (Phase 7) meta-model + calibration
│   ├── backtest/      # (Phase 8) walk-forward harness
│   ├── evaluation/    # (Phase 9) scoring + learning loop
│   ├── api/           # (Phase 10) FastAPI service
│   └── alerts/        # (Phase 11) alert rules + channels
├── apps/dashboard/    # (Phase 10) Next.js
├── config/            # default.yaml, assets.yaml — no secrets
├── sql/               # TimescaleDB-specific DDL
├── tests/             # unit + integration
├── docs/              # phases, data model, ADRs
└── scripts/           # operational entry points
```

Entries annotated with a phase are the **target** layout and do not exist yet:
directories for unbuilt phases are intentionally absent rather than filled with stubs,
because an empty module pretending to be a model is worse than no module. What is on
disk today is `core`, `config`, `storage`, `providers`, `ingestion`, `quality`,
`features`, `state`, plus `config/`, `sql/`, `tests/` and `docs/`.

---

## 5. Data model (Phase 1 core)

Canonical entities:

- **`assets`** — canonical asset identity (`BTC`), decoupled from any exchange symbol.
- **`data_sources`** — a provider (`binance`), with priority and enabled flag.
- **`instruments`** — the (asset, source, market_type) triple mapping `BTC` →
  `BTCUSDT` on Binance spot and `BTC-USD` on Coinbase. This indirection is what
  makes multi-source failover and cross-source comparison possible.
- **`ohlcv`** — the central hypertable, keyed `(instrument_id, timeframe, open_time)`.
- **`funding_rates`**, **`open_interest`** — derivatives context.
- **`global_metrics`** — BTC dominance, total market cap, stablecoin share.
- **`data_quality_events`** — every anomaly the validation layer found.
- **`source_quality_scores`** — rolling 0–1 trust score per (source, asset, timeframe).
- **`ingest_runs`** — provenance: what was fetched, from where, when, and how it went.

Every stored row carries source, timestamp, asset, timeframe and `ingested_at`,
satisfying the provenance requirement. Full DDL and the planned Phase 5–9 tables
are in [`docs/data-model.md`](docs/data-model.md).

### Why `is_final` matters

The in-progress candle is stored with `is_final = false`. **No feature, pattern or
model may read a non-final candle for anything but display.** This is the primary
structural defense against look-ahead bias in live operation.

---

## 6. Provider abstraction

```python
class MarketDataProvider(ABC):
    name: str
    capabilities: ProviderCapabilities        # timeframes, market types, history depth
    async def fetch_ohlcv(...) -> list[Candle]
    async def health() -> ProviderHealth
```

The **`ProviderManager`** wraps N providers with:

- **priority ordering** — cheapest / highest-quality source first;
- **token-bucket rate limiting** — per provider, enforced before the request;
- **circuit breaker** — after `k` consecutive failures a provider is opened for a
  cooldown and skipped entirely, so a dead API costs one check rather than N timeouts;
- **failover** — the next capable provider is tried transparently, and the fact that
  failover occurred is recorded as a quality event, not silently swallowed;
- **cross-source audit** — the same window fetched from two sources and compared,
  producing `SOURCE_DISCREPANCY` events.

---

## 7. Data quality engine

Checks run on every ingested batch, before persistence:

| Check | Detects | Severity |
|---|---|---|
| `shape` | `high < low`, `high < max(o,c)`, negative volume, NaN | error (row rejected) |
| `grid_alignment` | timestamps not on the timeframe boundary | error |
| `duplicates` | repeated `open_time` in one batch | warning (deduped, last wins) |
| `ordering` | non-monotonic timestamps | warning (sorted) |
| `gaps` | missing candles vs. the expected grid | warning / error by ratio |
| `outliers` | robust z-score (MAD) of returns beyond threshold | warning |
| `impossible_move` | absolute return beyond a per-timeframe hard cap | error |
| `staleness` | newest candle older than `k ×` timeframe | warning / error |
| `flatline` | zero volume or zero range across a run of candles | warning |
| `source_discrepancy` | two providers disagree on the same candle beyond tolerance | warning / error |

Events are persisted and folded into a rolling **quality score** per
(source, asset, timeframe). **That score is a multiplier on published prediction
confidence** — the mechanism behind requirement §20: bad data lowers confidence
instead of being papered over.

---

## 8. Multi-timeframe strategy (Phase 3 design, committed now)

Timeframes are not peers. The state engine builds a **hierarchy**:

```
1W / 1D  → macro regime      (sets the prior)
4H       → structural trend  (constrains medium-term scenarios)
1H       → active trend
15M      → momentum
5M / 1M  → execution-grade micro behaviour
```

Each level emits `{direction, strength, confidence}`. The state engine computes an
**agreement tensor** rather than a vote: conflict between a bullish daily and a
bearish 15m is *information* ("pullback within uptrend"), not noise to be averaged
away. Higher timeframes set the prior; lower ones update it. Storing the per-level
states — not just the aggregate — is what makes the "Why?" panel and
regime-conditional evaluation possible.

Rollups (1m → 5m → … → 1w) are derived by TimescaleDB continuous aggregates from a
single stored base resolution where the provider supports it, guaranteeing that
timeframes are mutually consistent by construction.

---

## 9. Prediction contract (Phases 6–7 design, committed now)

Every model emits the same envelope so that the ensemble, the store and the UI never
special-case a model:

```python
Prediction(
    asset, timeframe, horizon,           # what and when
    distribution={UP: .68, FLAT: .22, DOWN: .10},
    expected_move_pct, expected_volatility,
    confidence,                          # distinct from probability
    evidence=[...], counter_evidence=[...],
    invalidation=[...],                  # explicit falsifiers
    alternative_scenarios=[...],
    model_id, model_version, feature_snapshot_id,
    regime, data_quality_score, as_of,
)
```

**Probability ≠ confidence.** Probability is the model's estimate of the outcome;
confidence is how much the system trusts that estimate right now, given regime,
recent calibration, model agreement and data quality. A "70% up" from a model that
has been badly calibrated in the current regime is published with low confidence.

**Super predictions** require independent agreement across ≥6 of 8 model families
*with* a calibration record in the current regime. Disagreement suppresses the super
signal entirely — it never averages into a confident-looking number.

---

## 10. Learning loop (Phase 9 design, committed now)

This is what separates the system from a dashboard of indicators:

1. Prediction written at `t0`, **before** the outcome exists — append-only, hash-stamped.
2. A scheduled evaluator wakes at `t0 + horizon`, resolves the realized outcome from
   final candles only, and writes a `prediction_outcome` row.
3. Metrics are computed **sliced**: by asset, timeframe, horizon, regime and
   volatility bucket — because "this model is good" is meaningless, while "this model
   is good on BTC 4H in low-vol regimes" is actionable.
4. Calibration: reliability curves per model per regime; isotonic recalibration
   applied to future outputs.
5. Weighting: the ensemble weights models by *recent, regime-matched* skill
   (log-loss / Brier against a persistence baseline), with shrinkage toward equal
   weights to avoid chasing noise.
6. Retraining is **walk-forward only** — rolling-window fit, next-window predict,
   never a random split. Every fit records the exact data window used so leakage is
   auditable after the fact.

---

## 11. Failure posture

The system degrades along a defined ladder rather than failing open:

```
full confidence → reduced confidence → "insufficient evidence" → no publication
```

A provider outage, a quality-score collapse, a regime the models have no calibration
record for, or strong inter-model disagreement each move the system down that ladder.
Silence is an acceptable output. A confident-looking number produced from broken
inputs is not.

---

## 12. Safety boundary

- No order-execution code, no exchange trading keys, no withdrawal permissions.
  Providers use **public, read-only endpoints**; where a key is ever needed it must be
  read-only and supplied via environment variable.
- Outputs are probabilistic scenarios, never guarantees, and the UI is required to
  render probability, confidence and invalidation conditions together.
- This is not investment advice, and nothing in the system may phrase it as such.

---

## 13. Build order

Phases are gated: a phase ships only when the previous one is correct and tested.
See [`docs/PHASES.md`](docs/PHASES.md) for the gate criteria of each.
**Phases 1 (ingestion + database), 2 (feature engine) and 3 (multi-timeframe
state) are implemented.** Phases 4+ are designed above and not yet built.
