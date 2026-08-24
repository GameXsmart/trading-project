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

## Phase 5 — News and sentiment intelligence ✅ COMPLETE (impact awaits data)

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

- `EventImpactModel` (estimates from priors) and `ImpactValidator` (measures what
  actually followed), kept rigorously separate — an estimate resting on an untested
  prior reports `grounded_in_measurement=False` and low confidence, so a hypothesis
  can never be mistaken for a finding.
- `news_events` table, so history accumulates across fetches.
- CLI: `mie news`, `mie news-impact`.

**Gate: the machinery meets it; the data does not exist yet.**

Deduplication works on live data — four correct cross-outlet merges from 129 articles,
no false merges. The impact validator is built, tested, and correct, but **cannot yet
be exercised on real news**: RSS carries about a week of history, and after thinning
overlapping events that leaves 6 security-incident and 5 ETF stories for BTC against a
25-event minimum. It reports *insufficient evidence*, which is the right answer.

This is a data limitation, not a code one, and it is why `news_events` is persisted:
the sample grows with every fetch, and the same command becomes meaningful after
months of accumulation. Lowering the threshold to manufacture a result was the one
option not on the table.

The validator is therefore verified against synthetic price series where ground truth
is known by construction — a genuine volatility jump after events is detected, and a
series with no such effect is *not* certified.

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

**Four bugs this phase surfaced, all found by measuring rather than reasoning.**

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
4. **A one-sided claim was tested two-sided.** The validator certified "precedes
   elevated volatility" for a category whose events were followed by *0% elevated vs a
   17% baseline* — significantly **calmer** than usual. Significance testing says
   "different"; the claim says "higher". Calmer-than-usual is now a named finding of
   its own rather than being mislabelled or folded into "no impact".

---

## Phase 6 — Prediction models ✅ COMPLETE (result: no model has skill)

**Delivered**

- The `Prediction` envelope from ARCHITECTURE §9: a distribution over up/flat/down,
  confidence kept separate from probability, evidence and counter-evidence,
  invalidation conditions, and the threshold the outcome will be scored against.
- Eight independent models, each mapping a *different* substrate — indicators (2),
  return dynamics (1), analogues (4), regime (3), news (5), peers (1), derivatives (1),
  event chains (4). `inputs_used()` declares the substrate so overlap is auditable.
- Three baselines as first-class forecasters: climatology, persistence, uniform.
- `WalkForwardEvaluator`: no random splits, non-overlapping evaluation points, Brier
  skill against a baseline, paired significance testing, Benjamini-Hochberg across
  every slice, results sliced by regime.
- CLI: `mie evaluate`.

**Gate: enforced, and NOT met by any model.**

Measured on BTC/ETH/SOL 1h, 12-bar horizon, 699 non-overlapping points each,
**160 slices in total**:

| Baseline | Models passing |
|---|---|
| **climatology** (the honest bar) | **0 of 8** |
| persistence (the folk bar) | 8 of 8 — *including models that abstain entirely* |

That second row is the more instructive one. `sentiment`, `orderflow` and `sequence`
abstain on this data — they emit a uniform distribution and no opinion — and they
"beat" persistence with identical skill of +0.0864. **Doing nothing beats persistence.**
Any result quoted against it is worthless, which is exactly why climatology is the
standard here.

Against climatology, every model that actually expresses a view scores *worse* than the
unconditional base rates. Best skill on any slice was +0.0123, which did not survive
significance testing across the family.

**Confirmed across the horizon grid, not just at 12 hours.** The obvious objection to
a negative result is that the horizon was wrong — too short to escape microstructure
noise, too long for anything to persist. So the whole gate was re-run across three
timeframes and five to six horizons each: **48 configurations, forecast reaches from
3 hours to 60 days, 2,032 slices**, each capped at the same number of evaluation points
so the significance test is doing comparable work in every row.

**0 of 2,032 slices passed.** Exactly one configuration produced any p < 0.05 at all
(SOL 1h, 3-hour reach, `similarity`, skill +0.0253, p = 0.015), and it does not survive
correction across the family.

Two things in that sweep are worth stating plainly.

*The apparent skill lives entirely in the small samples.*

| Evaluation points | Configurations | Mean best skill | Largest | Smallest p |
|---|---|---|---|---|
| n < 100 | 9 | **+0.0303** | +0.0529 | 0.101 |
| 100–199 | 12 | −0.0041 | +0.0266 | 0.124 |
| 200–449 | 9 | +0.0015 | +0.0096 | 0.168 |
| n ≥ 450 | 18 | +0.0041 | +0.0253 | **0.015** |

Every headline number in the top ten came from the sparsest rows — BTC 1d at a 20-day
reach posted +0.0529 skill on 68 points, and +0.0447 on 22. None reached significance.
As the sample grows the effect collapses toward zero, which is the signature of noise,
not of an edge that is merely hard to detect.

*The best model is usually the one that says nothing.* `sentiment`, `orderflow` and
`sequence` abstain on **100% of points** on this data — verified directly, not
inferred. Yet `sentiment` was the highest-scoring model in **34 of the 48
configurations**. Only `similarity` (8), `timeseries` (3), `crossasset` (2) and
`regime` (1) ever beat an abstention, and never significantly. When a uniform
distribution outscores every considered opinion across two thirds of a grid this size,
the considered opinions are worse than useless.

*A minor calibration note.* The flat class holds 35% of outcomes at a 3-hour reach and
25–29% at longer ones, so `move_threshold`'s √horizon scaling slightly under-scales as
the horizon grows — real moves accumulate a little faster than a random walk implies.
The classes stay balanced enough everywhere for Brier scores to remain interpretable,
so this is recorded rather than corrected.

**This is the answer the phase was built to obtain**, and it is consistent with
everything measured earlier: Phase 4 found no directional pattern beating drift, and
Phase 5 could not yet validate that news moves prices. A system that reported a winner
here would be reporting noise.

**A flaw in my own gate, caught by measurement.** The original criterion — "beats the
baseline on at least one slice" — is nearly guaranteed by chance across forty slices,
and `timeseries` initially "passed" on exactly one with skill +0.0123. Inspection showed
the four *abstaining* models scored an identical +0.0047 on that same slice, meaning
climatology itself was slightly off there and the model's real margin over doing nothing
was ~0.008. The gate now requires a one-sided paired test on per-prediction Brier
differences, corrected with Benjamini-Hochberg across every slice. After that, nothing
passes.

**Design decisions worth carrying forward**

- **Abstention is a first-class output.** A model with no substrate emits a uniform
  distribution and zero confidence, and is scored on having done so.
- **Data quality multiplies into confidence centrally**, in `Predictor.build`, so no
  model can forget it.
- **Look-ahead is structural, not conventional.** Models receive a `PredictionContext`
  built from `candles[:i+1]` and have no database handle; the realised return is read
  from `candles[i+horizon]`, and the two never meet.
- **The threshold is volatility-scaled.** A fixed band would make long horizons
  trivially directional and would mean different things in different regimes, breaking
  the regime-sliced comparison.

---

## Phase 7 — Ensemble, calibration, and confidence ✅ COMPLETE (result: publishes nothing)

**Delivered**

- **Isotonic calibration** per model per regime (`ensemble/calibration.py`), fitted by
  pool-adjacent-violators. Fitted on an earlier window, judged on a later one; a curve
  that does not improve held-out calibration by a margin is discarded and the model's
  own numbers kept. Records carry `fitted_through`, and applying one to a prediction
  at or before that instant raises.
- **Reliability diagrams** with Wilson intervals per bin, classwise ECE across all
  three outcomes, and both the literal ±5-point criterion and the interval-based one.
- **Independence-discounted agreement** (`ensemble/agreement.py`). Each model's vote is
  weighted `1 / (1 + Σ jaccard(inputs_i, inputs_j))`, so six models reading the same
  substrate count as roughly one opinion rather than six.
- **Confidence as a product of six named, measured factors** (`ensemble/confidence.py`):
  skill, calibration, agreement, data quality, sample size, regime familiarity. Every
  published number can be decomposed into which factor limited it. Capped at 0.85.
- **The meta-model** (`ensemble/meta.py`): a linear pool weighted by measured
  out-of-sample skill, per regime, significant after Benjamini-Hochberg across every
  slice tested. A model without demonstrated skill gets weight **zero**, not a floor.
- **The super-prediction gate** (`ensemble/gate.py`): nine conjunctive conditions, each
  reporting pass/fail with its numbers.
- CLI: `mie calibrate`, `mie ensemble`.

**Gate: all three criteria met — by machinery that then refuses to publish.**

| Criterion | Result |
|---|---|
| Reliability within tolerance for what is published | Met vacuously — nothing is published. Measured directly on synthetic forecasters instead: a 70%-stated / 70%-observed panel passes, a 70%/40% panel fails. |
| Induced disagreement produces no super prediction | Met. 4-vs-4 and 5-vs-3 splits both fail `families agreeing` and `no material dissent`; a unanimous panel with skill and calibration passes, proving the gate is not simply always false. |
| Degrading the feed measurably lowers confidence | Met. Confidence falls monotonically with data quality end-to-end through the ensemble, and a feed at 0.25 suppresses publication entirely. |

**Measured on live data: the ensemble publishes nothing, and the gate never fires.**

Swept across BTC, ETH and SOL on 1h with a 12-bar horizon — **2,097 non-overlapping
evaluation points**:

| | BTC | ETH | SOL |
|---|---|---|---|
| Evaluation points | 699 | 699 | 699 |
| Models with a non-zero weight | none | none | none |
| Usable calibration records (of 42) | 3 | 10 | 6 |
| **Ensemble published** | **0** | **0** | **0** |
| **Super predictions** | **0** | **0** | **0** |

Six of the gate's nine conditions fail at every single point on all three assets:
`families agreeing`, `independent agreement`, `calibrated in this regime`,
`demonstrated skill`, `confidence`, `ensemble published`. A seventh — `no material
dissent` — fails on 54% to 62% of points depending on the asset.

Two of those failures are worth separating, because they say different things. The
`demonstrated skill` failure is inherited from Phase 6: nothing has earned a weight.
But `families agreeing` fails independently of skill — **the eight models never reach
six votes in one direction at any of the 2,097 points**, mostly because half of them
abstain. Even if a model were later shown to have skill, the panel as currently
constituted would still not produce a super prediction.

**Calibration overfits at these sample sizes, and the code caught it.** Fitting isotonic
curves for every (model, regime) pair on BTC 1h gave:

| Outcome | Count |
|---|---|
| Curve kept (improved held-out ECE by ≥0.005) | **3 of 42** |
| Curve fitted but discarded — *worse* out-of-sample | 21 |
| Too little data to fit at all | 18 |

The three survivors improved held-out ECE by +0.0051, +0.0069 and +0.0157. Everything
else made calibration worse on data it had not seen — including climatology itself,
which got worse in all four of its regimes. This is isotonic regression doing exactly
what the module's docstring predicts: with enough freedom to fit noise and only ~100
holdout points, it fits noise. The defence is the held-out judgement, and it worked.

**What the raw panel's reliability actually shows.** Across 12,258 (probability,
outcome) pairs on BTC:

| Stated | n | Observed | 95% interval |
|---|---|---|---|
| 0.1–0.2 | 457 | 0.309 | [0.268, 0.352] |
| 0.2–0.3 | 2,313 | 0.324 | [0.305, 0.344] |
| 0.3–0.4 | 8,036 | 0.330 | [0.319, 0.340] |
| 0.4–0.5 | 1,421 | 0.374 | [0.350, 0.400] |

The models are systematically compressed toward the base rate: across a stated range of
0.30 the observed frequency moves only about +0.065, and no populated bin's stated
probability lies inside its own observed interval. There is a faint monotone signal —
higher stated probability does correspond to higher observed frequency — but far too
weak to be worth publishing, which is the same conclusion Phase 6 reached by a
different route.

**A leak the guard caught, in my own code.** The first sweep script fitted calibration
over all history and then replayed it from the beginning of that same history. The
`fitted_through` check raised immediately. The ensemble now treats an inapplicable
record as *uncalibrated at that instant* rather than using it — the honest degradation
— and a regression test covers it. Rolling refits belong to Phase 8.

**Why build the layer at all, given Phase 6.** Because the layer's job here is to
*suppress*, and a suppression mechanism that has never been shown to also permit is
indistinguishable from a bug. Every refusal test is paired with a test that the same
machinery fires when a skilled, agreeing, calibrated panel is supplied.

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
