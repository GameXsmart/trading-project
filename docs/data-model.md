# Data model

Reference for the Phase 1 schema, plus the planned tables for later phases. Source of
truth for the implemented tables is [`src/mie/storage/models.py`](../src/mie/storage/models.py).

---

## Conventions

Applied without exception across every table:

| Rule | Why |
|---|---|
| All timestamps are **timezone-aware UTC** | Mixing naive and aware datetimes across a storage boundary is a classic source of off-by-hours corruption. The `UTCDateTime` decorator rejects naive input rather than assuming a zone, and re-attaches UTC on read (SQLite has no timezone type and would otherwise silently return naive values). |
| Every time-series row carries `source` and `ingested_at` | Provenance. "Where did this number come from and when did we learn it" must always be answerable. |
| Prices are `Float` (IEEE double), not `Numeric` | This is an analytical system, not a ledger. Doubles carry ~15 significant digits — far beyond any exchange's tick precision — and every consumer is floating-point anyway. A settlement system would choose the opposite. |
| Windows are half-open `[start, end)` | The only convention under which adjacent windows tile without overlapping or gapping. Providers that disagree (Coinbase's inclusive `end`, Kraken's exclusive `since`) are normalised at the provider boundary. |
| Assets are decoupled from exchange symbols | `assets` holds canonical identity; `instruments` maps it per venue. Without this, multi-source failover and cross-source comparison become string-munging at query time. |

---

## Phase 1 tables (implemented)

### `assets`
Canonical asset identity, independent of any venue.

| column | type | notes |
|---|---|---|
| `id` | int PK | |
| `symbol` | str, unique | `BTC` — normalised upper case |
| `name` | str | `Bitcoin` |
| `tier` | int | 1 = full coverage; 2 = slower timeframes only |
| `is_active` | bool | |
| `meta` | JSON | |

### `data_sources`
A provider. `priority` drives failover order (lower wins).

| column | type | notes |
|---|---|---|
| `id` | int PK | |
| `name` | str, unique | `binance` |
| `kind` | str | `exchange` \| `aggregator` \| `synthetic` |
| `enabled`, `priority` | bool, int | |

### `instruments`
The `(asset, source, market_type)` triple → the venue's own symbol.

| column | type | notes |
|---|---|---|
| `id` | int PK | |
| `asset_id`, `source_id` | FK | |
| `provider_symbol` | str | `BTCUSDT`, `BTC-USD`, `XBTUSD` |
| `market_type` | str | `spot` \| `perp` \| `futures` |
| `quote` | str | `USDT`, `USD` |

Unique on `(asset_id, source_id, market_type)`.

### `ohlcv`
The central time-series table. **Hypertable** on `open_time`.

| column | type | notes |
|---|---|---|
| `instrument_id` | FK, **PK** | |
| `timeframe` | str, **PK** | `1m` … `1w` |
| `open_time` | timestamptz, **PK** | grid-aligned, half-open bar start |
| `open`/`high`/`low`/`close` | float | |
| `volume`, `quote_volume`, `trades` | float, float, int | |
| `is_final` | bool | **false while the bar is still forming** |
| `revision` | int | incremented on rewrite, so revisions are visible |
| `ingested_at` | timestamptz | |

The composite PK gives idempotent upserts for free and includes the partitioning
column, which TimescaleDB requires of a hypertable's unique constraints.

**`is_final` is load-bearing.** No feature, pattern, or model may read a non-final
candle for anything but display; `OHLCVRepository.fetch` excludes them by default and
callers must opt in explicitly. This is the primary structural defence against
look-ahead bias in live operation.

**Upsert semantics.** A stored bar is overwritten only when the incoming one is at
least as authoritative: a final candle replaces a provisional one, but a provisional
candle never overwrites a final one — that would reintroduce an unfinished bar the
analytics have already consumed.

### `funding_rates` / `open_interest`
Perp funding and outstanding leverage, keyed `(instrument_id, ts)`. Inputs to regime
detection and the order-flow model.

### `global_metrics`
Market-wide context keyed `(source_id, ts)`: BTC/ETH dominance, total market cap,
24h volume, stablecoin share.

### `data_quality_events`
Every defect the validation layer found.

| column | type | notes |
|---|---|---|
| `event_type` | str | see the table below |
| `severity` | str | `info` \| `warning` \| `error` |
| `source`, `asset`, `timeframe` | str | free text, **not FKs** |
| `window_start`, `window_end` | timestamptz | |
| `message`, `details` | text, JSON | |
| `detected_at`, `resolved_at` | timestamptz | |

Scope is stored as free text rather than foreign keys deliberately: a quality event
may concern a source or asset that cannot be resolved to a row, and losing the event
to a broken reference would defeat its purpose.

**Event types**

| type | severity | data is |
|---|---|---|
| `shape_invalid` | error | rejected |
| `grid_misaligned` | error | rejected |
| `duplicate` | warning | deduped (last wins) |
| `out_of_order` | warning | sorted |
| `gap` | info→error by ratio | flagged |
| `outlier` | warning | flagged, kept |
| `impossible_move` | error | flagged, kept |
| `stale_feed` | warning/error | flagged |
| `flatline` | warning | flagged |
| `source_discrepancy` | warning/error | flagged |
| `provider_failover` | warning | recorded |
| `provider_error` | warning | recorded |
| `empty_response` | info | recorded |

The rejected/flagged split matters: structurally impossible bars are discarded because
they would corrupt downstream maths, while suspicious-but-possible bars are kept
because discarding them would fabricate a calmer market than the real one.

### `source_quality_scores`
Rolling trust score in `[0, 1]` per `(source, asset, timeframe)`.

```
mass    = Σ(severity_weight × recency_decay)
rate    = 1000 × mass / candles_assessed
penalty = 1 - exp(-rate / event_rate_tolerance)
score   = clamp(1 - penalty - staleness_penalty, min_score, 1)
```

The **rate**, not the count, is what the score measures. Forty warnings across a year
of hourly history is a healthy feed; forty in an hour is a broken one, and a
count-based score saturates on the first and therefore says nothing useful about the
second. `candles_assessed` counts bars written in the window (by `ingested_at`), so a
bulk backfill reads as one large recent assessment rather than a catastrophe.

Staleness is scored separately from events, because a feed that silently stops
producing generates no events at all and an event-only score would rate it perfect.

**This score is the mechanism behind requirement §20.** Phase 7 multiplies it into
published confidence: degraded inputs produce quieter output rather than confidently
wrong output.

### `ingest_runs`
Append-only provenance for every job: what was requested, what was covered, rows
fetched/written/rejected, event count, status, error, duration.

---

## TimescaleDB specifics

Applied automatically on PostgreSQL from [`sql/timescale.sql`](../sql/timescale.sql);
skipped with a warning if the extension is absent, since plain PostgreSQL is a
usable (if slower) target and refusing to start would be worse.

- **Hypertables**: `ohlcv` (7-day chunks), `funding_rates`, `open_interest`,
  `global_metrics`, `data_quality_events` (30-day chunks).
- **Compression** on `ohlcv` after 14 days, segmented by `(instrument_id, timeframe)`,
  ordered by `open_time DESC` — historical candles are read in wide per-instrument
  ranges and never updated once final, exactly the pattern columnar compression rewards.
  Fourteen days is comfortably past the point at which a bar can still be revised.
- **Retention**: quality events are diagnostic, not analytical — one year, not forever.
- **Continuous aggregates**: `ohlcv_5m_from_1m` is the reference shape. Deriving higher
  timeframes from a stored base resolution guarantees that timeframes agree by
  construction rather than by hope; Phase 2 extends this to the full ladder.

### Migrations

`create_all` is adequate while the schema is additive. **The moment a destructive or
transforming migration is needed, this becomes Alembic** — that is the threshold, and
it is stated here rather than pre-solved, because an unused migration framework is
just more surface area to maintain.

---

## Planned tables (Phases 5–9)

Not implemented. Specified here because they constrain Phase 1's design and because
guessing at them later would be worse than sketching them now.

### `features` (Phase 2)
`(instrument_id, timeframe, open_time, feature_set_version)` → JSON/columnar values.
Versioned so a feature-definition change is visible rather than retroactively
rewriting history.

### `market_states` (Phase 3)
Per-timeframe `{direction, strength, confidence}` plus the cross-timeframe agreement
tensor. Per-level states are stored, not just the aggregate — the explanation panel
and regime-conditional evaluation both require them.

### `news_events` (Phase 5)
`source, published_at, ingested_at, title_hash, cluster_id, category, assets[],
sentiment, importance, confidence, is_recycled`. `cluster_id` collapses the same story
republished across outlets into one event with wider coverage.

### `predictions` (Phase 6)
Append-only, hash-stamped, written **before** the outcome exists:
`id, created_at, asset, timeframe, horizon, distribution JSON, expected_move_pct,
expected_volatility, confidence, evidence JSON, counter_evidence JSON, invalidation
JSON, model_id, model_version, feature_snapshot_id, regime, data_quality_score`.

### `prediction_outcomes` (Phase 9)
`prediction_id, resolved_at, realised_direction, realised_move_pct, error, brier,
log_loss, invalidated, invalidation_reason`. Resolved from final candles only.

### `model_performance` (Phase 9)
Sliced metrics keyed `(model_id, asset, timeframe, horizon, regime, volatility_bucket,
window_start, window_end)`. The slicing is the point: "this model is good" is
meaningless, "good on BTC 4H in low-vol regimes" is actionable.

### `model_weights` / `calibration_maps` (Phase 7/9, not yet persisted)

Phase 7's `SkillWeights` and `CalibrationLibrary` are currently derived in-process from
a walk-forward run rather than stored. That is deliberate for now: a stored weight or
curve is only valid for the data window it was fitted on, and persisting one without
also persisting and enforcing that window is how a stale calibration silently outlives
the regime it was measured in. `CalibrationRecord.fitted_through` is the field these
tables will key on. Until then the derivation is repeated on demand, which is slower
and cannot go stale.

### `model_weights` / `calibration_maps` (Phase 7/9)
Current ensemble weights and isotonic calibration curves per model per regime, with
the evaluation window that produced them — so any published number can be traced back
to the evidence that justified its weight.
