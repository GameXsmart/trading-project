# Build phases and gate criteria

Phases are gated. A phase ships only when the previous one is **correct**, not merely
present — because every later layer inherits the errors of the one below it, and a
prediction built on a silently broken feature is worse than no prediction.

Each phase below states its **gate**: the condition that must hold before the next
phase begins.

---

## Phase 1 — Market data ingestion + database ✅ COMPLETE

**Delivered**

- Typed configuration (YAML + env layering, secrets only from the environment).
- Schema for assets, sources, instruments, OHLCV, funding, open interest, global
  metrics, quality events, trust scores, and ingest runs — TimescaleDB-targeted,
  SQLite-portable.
- Four providers (Binance, Coinbase, Kraken, CoinGecko) behind one interface, plus a
  deterministic synthetic provider with injectable faults.
- Provider manager: priority failover, token-bucket rate limiting, circuit breakers,
  cross-source auditing.
- Backfill with two-directional segment planning, paging, and post-hoc coverage
  verification against stored data.
- Live poller scheduled on bar boundaries, publishing `candle.closed`.
- Ten-check validation layer and a rolling trust score.
- CLI covering every operation; 164 offline tests plus an 11-test live contract suite.

**Gate: met.**
- 150,000+ candles ingested across 30 live series at 100% grid completeness.
- Zero quality events on clean live data; injected faults are detected in tests.
- Re-running any ingest command is idempotent.
- No non-final candle is ever returned by a default query.

---

## Phase 2 — Technical feature engine ✅ COMPLETE

**Delivered**

- Sixteen incremental indicators (SMA/EMA/Wilder, RSI, MACD, ATR, Bollinger, ADX,
  Stochastic, OBV, anchored VWAP, realised volatility, ROC), each a state machine fed
  one closed bar at a time.
- Market structure: confirmed swing detection, level clustering, volume profile with
  POC and value area, and direction-aware Fibonacci retracements.
- `FeatureEngine` subscribing to `candle.closed`, with warm-up from stored history and
  per-bar persistence to a versioned `features` table.
- CLI: `mie features compute`, `compute-all`, `show`.

**Gate: met.**
- Every indicator matches an **independent, naively-written** reference implementation
  to 1e-9 across a 300-bar fixture (45 tests).
- Resuming from primed state is **bit-identical** to running continuously, verified
  over a 2,000-bar series — which is what makes warm-up on restart safe.
- Look-ahead is disproved directly: two series sharing 250 bars and diverging
  violently afterwards produce **identical feature vectors** for every shared bar.
- Verified on live data: 43,104 vectors computed across BTC/ETH/SOL, and stored
  SMA/Bollinger/MACD values re-derived from raw candles match exactly.

**Design decisions worth carrying forward**

- Windowed indicators recompute from a bounded deque rather than keeping a running
  sum; running sums drift from fresh summation and would break exact reproducibility.
- Features are keyed by *instrument*, not asset: feeding two venues' bars into one EMA
  during a failover would silently corrupt it.
- Provisional and out-of-order bars are **refused**, not tolerated — recursive
  indicator state cannot be rewound.

---

## Phase 3 — Multi-timeframe market state

**Build**: the hierarchical state model — per-timeframe `{direction, strength,
confidence}` plus the agreement tensor across levels.

**Design constraints**

- Higher timeframes set the prior; lower ones update it. Conflict is *information*
  ("pullback inside an uptrend"), not noise to average away.
- Per-level states are persisted, not just the aggregate — the "Why?" panel and
  regime-conditional evaluation both need them.

**Gate**
- Hand-labelled historical scenarios (clear uptrend, chop, capitulation, recovery)
  are classified correctly.
- A bullish daily with a bearish 15m yields "pullback within uptrend", never a flat
  neutral average.

---

## Phase 4 — Pattern and sequence discovery

**Build**: pattern detection (breakouts, fakeouts, accumulation, distribution,
liquidity sweeps, compression/expansion, exhaustion, divergences, structure breaks),
historical similarity search, and statistical sequence mining.

**Design constraints**

- A pattern is only admitted if it has a **measured historical base rate** with a
  confidence interval. "This looks like a breakout" is worthless; "this configuration
  resolved upward 58% of the time (n=214, CI 51–65%)" is not.
- Similarity search must be honest about regime: the nearest historical neighbour
  from a different volatility regime is not a comparable situation.

**Gate**
- Every detector reports its historical base rate, sample size, and interval.
- Detectors with base rates statistically indistinguishable from chance are removed,
  not shipped with a caveat.

---

## Phase 5 — News and sentiment intelligence

**Build**: news ingestion, deduplication (including recycled/reposted stories), asset
relevance, event categorisation, importance estimation, and the event-impact model.

**Design constraints**

- Social posts are *signals about sentiment*, never facts about the world.
- Deduplication is mandatory: the same story republished by twelve outlets is one
  event with wider coverage, not twelve events.
- Every event carries source, timestamp, relevance, category, sentiment, estimated
  importance, and confidence.

**Gate**
- Recycled-news detection is measured on a labelled set (target: >90% recall).
- Impact estimates are validated against realised post-event volatility, not asserted.

---

## Phase 6 — Prediction models A–H

**Build**: eight independent predictors — technical structure, time-series
forecasting, pattern similarity, regime, sentiment, cross-asset, order-flow /
derivatives, and sequence analysis.

**Design constraints**

- **Independence is the point.** Eight models sharing one feature set is one model
  with extra steps and produces false agreement in the ensemble.
- Every model emits the common `Prediction` envelope (see `ARCHITECTURE.md` §9).
- Every model must beat a **persistence baseline** on walk-forward evaluation, or it
  does not ship. Complexity is justified by measured skill or not at all.

**Gate**
- Each model beats persistence out-of-sample on at least one (asset, timeframe,
  regime) slice, with the losing slices documented rather than hidden.

---

## Phase 7 — Ensemble, calibration, and confidence

**Build**: the meta-model, isotonic calibration per model per regime, the confidence
computation, and the super-prediction gate.

**Design constraints**

- **Probability ≠ confidence.** Probability is the outcome estimate; confidence is
  how much the system trusts that estimate given regime, recent calibration, model
  agreement, and data quality.
- The Phase 1 trust score multiplies into published confidence here. This is where
  requirement §20 becomes visible to a user.
- Super predictions require ≥6 of 8 independent model families agreeing **and** a
  calibration record in the current regime. Disagreement suppresses the signal
  entirely — it never averages into a confident-looking number.

**Gate**
- Reliability diagrams are within tolerance: of everything published at 70%, 70% ±5%
  occurs.
- Deliberately induced model disagreement produces *no* super prediction.
- Degrading the input feed measurably lowers published confidence.

---

## Phase 8 — Backtesting and walk-forward evaluation

**Build**: the walk-forward harness — rolling-window fit, next-window predict, across
bull, bear, sideways, and high-volatility regimes.

**Design constraints**

- **Never a random split.** Time-series data shuffled into train/test leaks the
  future into the past, and the resulting metrics are fiction.
- Every fit records the exact data window used, so leakage is auditable afterwards.
- Survivorship bias is addressed explicitly: delisted assets stay in the historical
  universe.

**Gate**
- A deliberately leaky model is *caught by the harness* — the test suite proves the
  leakage detector works, rather than assuming it.
- Results are reported per regime; a single blended accuracy number is not accepted.

---

## Phase 9 — Self-evaluation and learning

**Build**: the outcome scorer, sliced performance metrics, calibration updates, and
dynamic model reweighting.

**Design constraints**

- Predictions are written **before** the outcome exists, append-only and hash-stamped.
- Outcomes resolve from final candles only.
- Metrics are sliced by asset, timeframe, horizon, regime, and volatility bucket —
  "this model is good" is meaningless; "good on BTC 4H in low-vol" is actionable.
- Reweighting uses recent, regime-matched skill with shrinkage toward equal weights,
  so the ensemble does not chase noise.
- **No fake learning.** Storing predictions is not learning. The loop must
  demonstrably change future behaviour.

**Gate**
- An injected model whose skill degrades in a specific regime is demonstrably
  down-weighted in that regime and only that regime.
- Calibration measurably improves after recalibration on held-out data.

---

## Phase 10 — API and dashboard

**Build**: FastAPI + WebSocket, and the Next.js dark-mode dashboard — market state,
per-asset panels, correlation matrix, prediction timeline, news feed, regime
indicator, model performance, calibration, super predictions, and the "Why?" panel.

**Design constraints**

- The UI must render probability, confidence, and invalidation conditions **together**.
  A number without its uncertainty is a lie of omission.
- Predictions and guarantees must be visually unmistakable from each other.
- "Insufficient evidence" is a first-class display state, not an error.

**Gate**
- No screen can display a directional call without its confidence and invalidation
  conditions visible in the same view.

---

## Phase 11 — Alerts

**Build**: configurable rules (strong/super prediction, regime change, breakout,
reversal, volume anomaly, liquidation spike, major news, correlation breakdown, model
disagreement, prediction invalidation) over browser, desktop, Discord, Telegram, email.

**Gate**
- Alert volume under a simulated volatile week stays within a rate budget. An alerting
  system nobody reads is worse than none.

---

## Phase 12 — Optimisation and scaling

**Build**: WebSocket feeds, Redis hot state, an out-of-process event bus (NATS),
parallel feature computation, and query tuning.

**Design constraints**

- WebSockets are added **alongside** polling, never instead of it: they are better for
  latency and worse for reliability, and the fallback must already exist.
- Rust or Go enters only if profiling shows a genuine CPU bottleneck (most likely
  order-book microstructure). A second toolchain must be earned.

**Gate**
- 50+ assets across all timeframes sustained within latency budget, with no increase
  in data-quality events versus the polling baseline.
