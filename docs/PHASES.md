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

## Phase 3 — Multi-timeframe market state ✅ COMPLETE

**Delivered**

- `Direction` / `Regime` / `Alignment` vocabulary, keeping direction, strength and
  confidence as three separate quantities rather than one blended score.
- `TimeframeClassifier`: feature vector → per-timeframe state, from four weighted
  signal groups (trend, momentum, structure, volume), with evidence and
  counter-evidence enumerated.
- `HierarchyAnalyzer`: per-timeframe states → one `MarketState` with named alignment,
  regime, agreement, conflicts, and a plain-English interpretation.
- `StateEngine` with `as_of` reconstruction, persistence of every per-timeframe level,
  and the Phase 1 trust score multiplying into confidence.
- CLI: `mie state BTC`.

**Gate: met.**
- Hand-labelled scenarios classify correctly: uptrend, downtrend, chop (no confident
  direction), capitulation, recovery.
- A bullish daily with a bearish 15m yields `pullback_in_uptrend` with a bullish
  bias — never a flat neutral average — and the conflict is listed explicitly.
- Verified against real history: BTC state reconstructed at 360 points over 60 days
  produced 40% aligned bearish, 31% aligned bullish, 13% conflicted, 8% possible
  reversal, 2% counter-trend rally.

**Design decisions worth carrying forward**

- The structural/tactical split is **relative to the set of timeframes analysed**, not
  anchored to a fixed one. An absolute hinge left the tactical group empty whenever
  every requested timeframe fell on one side of it, silently disabling pullback
  detection — a failure invisible to unit tests built from hand-made states, and
  caught only by reconstructing real history.
- Within the structural group the slowest timeframe leads; within the tactical group
  the **fastest** leads, because a correction appears there first.
- Confidence is capped below 1.0. A hand-weighted rule set that agrees with itself is
  still only a rule set, and clean trending data reaches the ceiling routinely.
- Group summaries state their **net** lean rather than implying unanimity.

---

## Phase 4 — Pattern and sequence discovery ✅ COMPLETE

**Delivered**

- Eleven detectors covering nineteen pattern kinds (breakouts, fakeouts, liquidity
  sweeps, compression/expansion, accumulation/distribution, divergences, momentum
  exhaustion, trend continuation, structure breaks, volume anomalies).
- Dependency-free statistics: Wilson score intervals, pooled two-proportion tests,
  Benjamini-Hochberg false-discovery control.
- `PatternEvaluator`: scans history, measures forward outcomes over three horizons,
  compares each pattern against the **unconditional** rate over the same sample.
- `PatternRegistry`: the gate. A pattern influences predictions only with stored
  evidence for that exact asset, timeframe and horizon.
- `pattern_stats` table and CLI: `mie patterns measure`, `mie patterns show`.

- `SimilarityEngine`: historical analogue search on scale-free features, with a
  self-calibrating distance ceiling, past-only normalisation and a forward embargo.
- `SequenceMiner`: enumerates pattern chains of length 2–3 and tests every one; plus a
  state-transition matrix with Wilson intervals.
- CLI: `mie similar BTC`.

**Gate: met for the detectors built.**

Every detector reports a base rate, sample size and confidence interval, and those
that fail are withheld rather than shipped with a caveat. Measured across BTC/ETH/SOL
on 1h and 4h, three horizons — 342 combinations:

**9 of 342 (2.6%) are informative. Every one of them is direction-neutral.**

| Pattern | Assets | Edge over baseline |
|---|---|---|
| `volume_anomaly` | BTC, ETH, SOL (h=3 and h=12) | +8.0% to +20.8% |
| `compression` | BTC, SOL 4h (h=3) | +22.4%, +24.8% |
| `expansion` | ETH 1h (h=3) | +12.1% |

**No directional pattern survived on any asset.** Breakouts, structure breaks,
divergences, trend continuation, liquidity sweeps and momentum exhaustion are all
statistically indistinguishable from the market's own drift after correction. On BTC
1h, `breakout_up` at h=3 was *negative* (43.5% vs a 49.8% baseline).

What did survive is volatility clustering — a well-established effect. Volume spikes
and range compression genuinely predict *movement*; they say nothing about direction,
and the system does not pretend otherwise.

**Methodological decisions worth carrying forward**

- The baseline is the **unconditional outcome rate over the same sample**, never 50%.
  In a market that rose 54% of hours, a "56% accurate" pattern has a two-point edge,
  not six.
- Benjamini-Hochberg is applied across the **whole sweep at once**. Uncorrected, 342
  tests at p<0.05 would produce ~17 "discoveries" from noise alone.
- Overlapping detections are thinned to one horizon apart: consecutive hits share
  nearly all of their forward window, so counting each as independent inflates n and
  shrinks the interval past what the evidence supports.
- Detector thresholds are conventional and fixed **before** measurement. Tuning them
  against the same history would guarantee a good-looking result and nothing else.
- Pattern direction is declared in advance. Deciding it after seeing the outcome is
  relabelling, not analysis.

**Sequence mining found nothing.** 84 chains occurred often enough on BTC 1h to be
tested; **none survived correction**. The strongest candidates
(`structure_break_down -> compression`, p=0.0082; `volume_anomaly -> fakeout_down`,
p=0.0083) do not clear Benjamini-Hochberg across 84 tests. Reported as a negative
result rather than quietly dropped.

**Similarity search gives three different honest answers.** On the same timestamp:
BTC returns *insufficient evidence* (only 16 comparable situations in 8,548 — the
current state genuinely has few precedents); ETH finds 200 analogues that rose 36% of
the time against a 52% baseline, an interval clear of it; SOL finds 200 analogues that
match the baseline exactly. "I don't know", "history leans down", and "history says
nothing" are all valid outputs, and the engine produces each where it belongs.

**Two bugs this phase surfaced.** The first `fakeout` implementation required only
`high > prior_high and close < prior_high`, which fired on **37% of all bars** — a
"pattern" present in a third of history describes the market rather than signalling
anything in it. It initially produced the sweep's one apparently-significant
directional finding (ETH `fakeout_down`, p=0.0031). Requiring the breach and the
rejection to be *material* dropped its firing rate to ~1%, and the finding vanished.
A frequency check now guards against the same class of error.

The similarity ceiling was also miscalibrated. Set at a fixed 1.0, it returned a
single analogue from 8,548 candidates — because for k standardised dimensions the
expected distance between two *unrelated* samples is √2 ≈ 1.414 (measured: 1.426),
so a ceiling below that rejects almost everything by construction. It is now derived
from the data's own dispersion, which also makes it self-calibrating across assets
instead of a number tuned to BTC.

---

## Phase 5 — News and sentiment intelligence 🟡 PARTIAL

**Delivered**

- Seven public RSS feeds, all keyless and verified reachable — no scraping, no
  free-tier credentials to expire.
- Stdlib RSS/Atom parsing; a feed that fails to parse is skipped with a warning rather
  than taking the fetch down.
- IDF-weighted deduplication merging one story across outlets, plus recycled-story
  detection keyed on content-derived stable cluster ids.
- Classification: asset relevance (title-weighted, ambiguous tickers excluded), event
  category, negation-aware lexicon sentiment, coverage-driven importance, and a
  separate confidence score.
- CLI: `mie news`.

**Still to build:** the event-impact model of requirement §9 — magnitude, direction,
affected assets and duration, **validated against realised post-event volatility
rather than asserted**. That validation is the harder half of this gate and is
deliberately not claimed yet.

**Gate: partially met.** Deduplication works on live data — four correct cross-outlet
merges from 129 articles with no false merges. Impact validation is outstanding.

**Design decisions worth carrying forward**

- **Rules, not a fitted model.** There are no labels for "category" or "market
  sentiment". Having an LLM label headlines and training on those labels produces a
  model that imitates the labeller, errors included, while making them unauditable.
  A transparent rule set an operator can read and correct is worth more here, and
  Phases 6–9 will measure whether the signal has any value at all.
- **Coverage is the importance signal.** How many independent outlets ran a story is
  real evidence about its significance, gathered before any price data is consulted.
- **Sentiment is not impact.** Sentiment describes the text; impact is a claim about
  prices and has to be measured against them.

**Three bugs this phase surfaced, all found by measuring rather than reasoning.**

1. **Bigram shingles were the wrong representation.** Outlets rewrite headlines rather
   than republishing them, so bigram overlap collapses even for unmistakably identical
   stories: two articles on the same Ray Dalio statement scored 0.167 against a 0.55
   threshold, and 129 live articles produced *zero* merges. IDF-weighted token overlap
   scores that pair at 0.384; the threshold was then calibrated against hand-marked
   cross-outlet pairs, settling at 0.30 where every merge is correct.
2. **Document-frequency cutoffs do not survive corpus size, at either end.** A hard
   "ignore tokens above 30% DF" rule zeroed the shared rare token identifying a story
   whenever the batch was small — in six headlines, a token in two is already 33%.
   Pure IDF has the mirror cliff: a token in *every* document weighs exactly zero, so
   two identical headlines in a two-article batch scored zero similarity. A weight
   floor fixes both, degrading gracefully into plain Jaccard when IDF has nothing to
   say.
3. **Filtering by age before clustering splits stories.** Coverage straddling the
   cutoff was broken apart — one outlet at 84 hours, its pair at 71 — turning three
   merged stories into six single-outlet ones and destroying the coverage count
   deduplication exists to produce. Clustering now runs first, and the age test is
   applied to the story using its most recent coverage.

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
