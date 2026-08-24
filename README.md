# Crypto Market Intelligence Engine (MIE)

An analytical engine that observes cryptocurrency markets, evaluates evidence, and
produces **probabilistic, uncertainty-bearing** assessments of market state.

> **This is not a trading bot.** It has no order-execution path, holds no trading
> keys, and never will. It produces scenarios and probabilities for a human to
> interpret — never guarantees, and never investment advice.

**Status: Phases 1 (ingestion + database), 2 (feature engine), 3 (multi-timeframe
market state), 4 (pattern and sequence discovery), 5 (news intelligence), 6
(prediction models), 7 (ensemble, calibration, confidence), 8 (walk-forward
backtesting), 9 (self-evaluation and learning), 10 (API and dashboard) and 11 (alerts)
are complete and tested.**

**The headline result: none of the eight models beats a climatology baseline, so the
ensemble publishes nothing.** See
[What the measurements say](#what-the-measurements-say). The full architecture is designed in [`ARCHITECTURE.md`](ARCHITECTURE.md);
the remaining phases are specified in [`docs/PHASES.md`](docs/PHASES.md) and not yet
built. Nothing in this repo pretends to be further along than it is.

---

## What Phase 1 delivers — ingestion

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

## What Phase 2 delivers — features

Sixteen incremental indicators plus market structure, computed one bar at a time and
stored per bar.

| Capability | Detail |
|---|---|
| **Indicators** | SMA/EMA/Wilder, RSI, MACD, ATR, Bollinger, ADX, Stochastic, OBV, anchored VWAP, realised volatility, ROC. |
| **Market structure** | Confirmed swings, clustered support/resistance with touch counts, volume profile (POC + value area), direction-aware Fibonacci levels. |
| **Incremental** | A new bar costs O(window), independent of how much history exists — not a recompute over months of data per candle. |
| **Exactly reproducible** | Resuming from primed state is bit-identical to running continuously, so a restart is not a discontinuity in the feature history. |
| **Look-ahead proof** | Two series that share 250 bars and then diverge violently produce identical vectors for every shared bar. |

Verified on live data: **43,104 feature vectors** computed across BTC/ETH/SOL, with
stored values re-derived from raw candles and matching exactly.

## What Phase 3 delivers — market state

A hierarchical read across timeframes, where **conflict is information rather than
noise**.

| Capability | Detail |
|---|---|
| **Per-timeframe state** | Direction, strength and confidence kept as three separate quantities, with evidence and counter-evidence enumerated. |
| **Named alignment** | `aligned_bullish`, `pullback_in_uptrend`, `rally_in_downtrend`, `possible_reversal`, `rangebound`, `conflicted`. |
| **Regime** | Bull/bear bands plus high/low volatility, accumulation, distribution, capitulation, recovery — volatility outranks direction, because a violent market is a different environment either way. |
| **Explanation** | A plain-English interpretation plus an explicit list of conflicts. |
| **Historical reconstruction** | `as_of` rebuilds the state as it stood, using only bars that had closed by then. |

A bullish daily with a bearish 15m reads *"pullback within a larger uptrend"* — never
a flat neutral average, which is the one description that fits neither timeframe.

## What Phase 4 delivers — measured patterns, not folklore

Eleven detectors, each of which must **earn** the right to influence a prediction.

Every pattern is scored against the *unconditional* outcome rate over the same
sample — not against a coin flip — with Wilson confidence intervals and
Benjamini-Hochberg correction across the whole sweep. Patterns that fail are withheld
from the predictive path entirely; there is no reduced-weight or "directionally
suggestive" mode.

Measured across BTC/ETH/SOL on 1h and 4h over three horizons — **342 combinations,
9 informative (2.6%)**:

| Pattern | Where it held | Edge over baseline |
|---|---|---|
| `volume_anomaly` | BTC, ETH, SOL (h=3, h=12) | +8.0% to +20.8% |
| `compression` | BTC, SOL 4h (h=3) | +22.4%, +24.8% |
| `expansion` | ETH 1h (h=3) | +12.1% |

**Every surviving pattern is direction-neutral.** Not one directional pattern —
breakouts, structure breaks, divergences, trend continuation, liquidity sweeps,
momentum exhaustion — beat the market's own drift on any asset. What survives is
volatility clustering: these patterns predict *movement*, say nothing about
direction, and the system does not pretend otherwise.

That result is the point of the phase, not a disappointment in it.

**Sequence mining agrees.** 84 pattern chains occurred often enough on BTC 1h to be
tested; none survived correction.

**Historical similarity search** answers "when has it looked like this before?" — and
is willing to answer "it hasn't". On one timestamp it gave three different honest
answers: BTC *insufficient evidence* (16 comparable situations in 8,548), ETH *200
analogues rose 36% against a 52% baseline*, SOL *matches baseline exactly*.

---

## What Phase 7 delivers — the layer that refuses to publish

The ensemble weights models by measured out-of-sample skill against climatology, per
regime, significant after correction across every slice tested. A model that has not
demonstrated skill receives weight **zero** — not a floor, not a shrunk weight, because
shrinkage toward equal weights hands influence to models that have earned none.

Swept across BTC, ETH and SOL on 1h — **2,097 non-overlapping evaluation points, 0
published predictions, 0 super predictions.** Six of the gate's nine conditions fail at
every single point. One of them is worth separating out: the eight models never reach
six votes in one direction *anywhere* in those 2,097 points, mostly because half of
them abstain. That failure is independent of skill — even if a model were later shown
to have some, this panel would still not produce a super prediction.

| Capability | Detail |
|---|---|
| **Isotonic calibration** | Per model per regime, fitted on an earlier window and judged on a later one. Curves that do not improve held-out calibration are discarded and the model's own numbers kept. |
| **Look-ahead defence** | Every record carries the instant it was fitted through; applying it to a prediction at or before that instant raises. It caught a real leak in a sweep script on its first run. |
| **Independence-discounted agreement** | Votes weighted by declared input overlap, so six models reading one substrate count as roughly one opinion. |
| **Confidence as six named factors** | Skill, calibration, agreement, data quality, sample size, regime familiarity — multiplied, capped at 0.85, and decomposable so the UI can say *which* factor limited a number. |
| **Nine-condition gate** | Conjunctive, each reporting pass/fail with its numbers. A gate that only returns a boolean cannot be told from a broken one. |

**Calibration overfits at these sample sizes, and the code caught it.** Of 42 (model,
regime) pairs on BTC 1h: 3 curves kept, 21 fitted but *worse* out-of-sample and
discarded, 18 with too little data to fit at all. Climatology itself got worse under
calibration in all four of its regimes. That is isotonic regression doing what it does
with ~100 holdout points, and the held-out judgement is what stops it reaching
production.

**What reliability actually shows.** Across 12,258 (probability, outcome) pairs, the
models are compressed toward the base rate: a stated range of 0.30 moves the observed
frequency only about +0.065, and no populated bin's stated probability lies inside its
own observed interval.

---

## What Phase 8 delivers — a harness that catches its own leaks

Every other defence against look-ahead in this repo is structural — an argument that
the code is correct. Phase 8 tests the claim instead.

The **leakage probe** takes a prediction point, builds the context normally, then
rebuilds it from history in which everything strictly *after* the prediction instant
has been replaced with implausible data. A model that cannot see the future must
produce a bit-identical prediction; any difference is proof of a leak rather than
suspicion of one.

The **control matters as much**: the probe also corrupts the past and requires the
output to change. A model that ignores its inputs passes the future test trivially, so
its verdict is `INCONCLUSIVE`, never `CLEAN`. A detector that reports "clean" for
something it cannot test launders ignorance into assurance.

| Capability | Detail |
|---|---|
| **Purged, embargoed folds** | A training label reaching *h* bars forward contaminates the last *h* training bars, so they are dropped; an embargo follows to break serial correlation. `Fold.leaks()` verifies its own construction and an unsound fold is never run. |
| **Rolling fit, next-window predict** | Phase 7's calibration and skill weights are fitted on the training window only — the thing Phases 6 and 7 structurally could not measure, since they fitted and applied on the same data. |
| **Two independent leak detectors** | Perturbation proves pipeline leaks; an implausible-skill screen catches the class perturbation structurally cannot see. The boundary between them is asserted in tests, not implied away. |
| **Auditable windows** | Every fold records exact bar ranges and timestamps, so what a fit was allowed to see is checkable afterwards rather than re-derived from the code. |
| **Survivorship** | As-of universe selection, with the bias quantified. No delistings are recorded yet, so it currently corrects nothing — stated rather than left implicit. |

**Measured: nothing survives folding.** Across BTC/ETH/SOL, five folds each, under both
expanding and rolling schemes — **0 models pass in every fold, 0 in any fold, 0 caught
leaking, 0 ensemble predictions published.**

The spread is the number worth reading. On SOL, `regime` swings from −0.041 to +0.045
across five folds for a mean of +0.006 — a fold-to-fold swing fifteen times the
average. A result that depends this heavily on which slice of history it landed in is
an era, not an edge.

**Four "independent" models post identical results.** `crossasset`, `orderflow`,
`sentiment` and `sequence` score the same to four decimal places in every fold on every
asset, because all four abstain and four uniform distributions score alike. Four
numbers in a table that are actually one number is how an ensemble talks itself into
false confidence, so the harness now names it.

---

## What Phase 9 delivers — a loop that says whether it learned

Storing predictions is not learning. Computing metrics is not learning. The loop earns
the word only if it changes what the system does next, so its report distinguishes
three states rather than two:

- **nothing to learn from** — too few resolved outcomes to say anything;
- **learned nothing** — enough evidence, and it did not support a change;
- **learned something** — a weight or a calibration curve moved, with the sample that
  moved it attached.

| Capability | Detail |
|---|---|
| **Append-only, hash-stamped** | Prediction ids are derived from the prediction point, so a re-run collides and is dropped. Verified live: running `mie predict` twice over the same points left the count at 8,100. |
| **Refused, not repaired** | The hash covers the claim — distribution, confidence, the threshold the outcome is scored against — and is checked on read. A record that fails is refused; whatever it now says is not what the model said. |
| **Final candles only** | And within one bar of the resolution instant. Resolution uses the threshold *stored with the prediction*, never one recomputed later. |
| **Sliced five ways** | Asset, timeframe, horizon, regime, volatility bucket — the last recorded at prediction time, not reconstructed. Thin slices report insufficient evidence instead of a number. |
| **Regime-confined reweighting** | A model that degrades in one regime loses weight there and nowhere else. Skill is not a scalar property of a model. |

**Measured on 8,100 resolved predictions across BTC/ETH/SOL:**

| | Result |
|---|---|
| Weight slices evaluated | 160 |
| **Granted a non-zero weight** | **0** |
| Calibration curves fitted | 42 |
| **Adopted** | **6** |

The loop's own verdict: *"learned: 0 weight changes, 6 calibration curves adopted."*
That is a real change — six model/regime pairs will be calibrated differently tomorrow
— and it is not the change anyone was hoping for. Nothing learned to trust a model
more. It learned that six forecasters are systematically miscalibrated in specific
regimes, including **climatology itself** in downtrends.

**The clearest result in the project.** Ranked by Brier over 900 outcomes each:

| Forecaster | Brier | |
|---|---|---|
| `sentiment`, `orderflow`, `sequence` | **0.6667** | *these abstain on every point* |
| `baseline_climatology` | 0.6668 | |
| `similarity` | 0.6692 | |
| `regime`, `technical`, `crossasset`, `timeseries` | 0.6758 – 0.6861 | |

0.6667 is exactly 2/3 — the Brier score of a uniform distribution. **Saying nothing
beats climatology, and climatology beats every opinion any model formed.**

And once more, the apparent skill lives in the small samples: 29 slices posted skill
above +0.05, on sample sizes of 2, 4, 7, 10, 19, 61 and 98. The largest edge in the
entire run, +0.2558, rests on two observations. The gate rejects all of them.

---

## What Phase 10 delivers — an interface that cannot overstate itself

The gate reads: *no screen can display a directional call without its confidence and
invalidation conditions visible in the same view.* Enforced in a template that is a
convention. So it lives in the API contract instead.

A prediction endpoint returns exactly two shapes:

- **`DirectionalCall`** — cannot be constructed without a non-zero confidence, a
  decomposition that equals it, and at least one non-blank invalidation condition.
- **`InsufficientEvidence`** — no direction, no probabilities at all, and at least one
  reason required.

There is no third shape carrying a direction "with caveats", because a caveated
direction still reaches a reader as a direction. **The UI cannot violate the gate
because the API cannot express a violation.** Most of the tests are attempts to
violate it — empty invalidation, whitespace invalidation, confidence below the
publication floor, a confidence that disagrees with its own breakdown — and each must
raise.

The **safety boundary is asserted, not assumed**: no route accepts POST/PUT/PATCH/
DELETE, and no route path contains `order`, `trade`, `execute`, `buy`, `sell`,
`withdraw` or `position`. §21 is a claim about absence, and absence is exactly what
stops being true quietly.

```bash
mie serve
```

Verified running against the live database: 10 assets, 26,529 bars, 8,100 resolved
predictions, **0 models with weight** — and the assessment panel reads *"Insufficient
evidence — the system has no directional call to publish. This is a measured result,
not a failure to load."* An empty panel would be ambiguous between loading, broken and
having nothing to say. Only the third is true.

**One deviation, stated plainly:** the dashboard is a single self-contained page served
by the API, not the Next.js app the plan specifies. An interface that cannot be loaded
and inspected is not delivered work, and a Next.js build adds a toolchain this repo
cannot verify end to end. The gate is framework-independent, so the substance is
unaffected.

---

## What Phase 11 delivers — alerts that respect a budget

The failure mode of alerting is not missing an event. It is producing so many that the
reader stops looking, at which point the system has negative value: it consumed
attention and then trained someone to ignore the one alert that mattered.

Four mechanisms, in order: **dedup** (have I already said exactly this?), **cooldown**
(have I said something of this kind about this asset recently?), **budget** (have I
already spent this hour's attention?), and a small **reserve** only `CRITICAL` may draw
on, so a noisy hour cannot crowd out the message that a data feed collapsed.

And the property that matters most: **suppression is never silent.** Held alerts are
counted by reason and surfaced as a periodic digest — which is itself exempt from the
budget, because a suppression notice that can be suppressed fails exactly when it is
needed.

**Measured over 29 days of real BTC/ETH/SOL history**, replayed hour by hour:

| | |
|---|---|
| Rules raised | 1,238 |
| **Delivered** | **422** |
| Suppressed | **816 (66%)** |
| Busiest hour | 5 (ceiling 6 + 2 reserve) |
| Busiest 24h | 30 (ceiling 30 + 8 reserve) |

Delivered: `regime_change` 118, `suppression_digest` 113, `volume_anomaly` 93,
`volatility_compression` 52, `volatility_expansion` 46 — and **zero directional
alerts**, because no model has earned a weight and the ensemble never publishes. Both
directional rules are implemented and tested against synthetic input, so the silence is
a measured result rather than an unwritten branch.

A directional alert carries the same burden as a published prediction: `Alert` refuses
to construct one without a confidence and an invalidation condition. Non-directional
kinds carry no such burden, because a volume anomaly claims only that movement will
likely be larger than usual and says nothing about the sign.

Destinations are read from the environment and never from code — an unconfigured
channel is *disabled*, not silently broken, so "no alerts arrived" can be told apart
from "no alerts were sent".

```bash
mie alerts --dry-run
```

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
| `mie features compute BTC 1h` | Compute and store features over stored history. |
| `mie features compute-all` | Compute features for the whole universe. |
| `mie features show BTC 1h` | Show the latest stored feature vector. |
| `mie state BTC` | Hierarchical multi-timeframe market state. |
| `mie patterns measure` | Measure every detector against history. |
| `mie patterns show` | Which patterns earned predictive use. |
| `mie similar BTC` | Historical analogues of the current state. |
| `mie news --asset BTC` | Deduplicated, classified news feed. |
| `mie news-impact BTC` | Measured impact of news on realised volatility. |
| `mie evaluate BTC` | Walk-forward model skill against a baseline. |
| `mie calibrate BTC` | Fit per-model calibration and report whether it helped. |
| `mie ensemble BTC` | The ensemble, its confidence decomposition, and the gate. |
| `mie backtest BTC` | Walk-forward folds with a leakage probe on every model. |
| `mie predict BTC` | Record predictions for later scoring, before outcomes exist. |
| `mie learn` | Resolve, measure, reweight — and say whether anything changed. |
| `mie serve` | The read-only API and dashboard on http://127.0.0.1:8000. |
| `mie alerts` | Evaluate the alert rules once, within the rate budget. |

---

## How it is put together

```
providers/ → quality/ → storage/ → core/events → features/ → state/ → patterns/
 failover    validate    Timescale   candle.closed  indicators  hierarchy  detectors
 throttle    score       SQLite                     structure   regime     statistics
 breakers                                                       alignment  similarity
                                                                           sequences
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

**4. A pattern must earn its influence.**
Detection is descriptive and cheap; prediction is a claim about the future and needs
evidence. The registry withholds any pattern lacking a significant measured edge for
that exact asset, timeframe and horizon — and on real data that withheld 333 of 342
combinations. Absence of evidence is treated as absence of permission.

**5. Thresholds are measured, not guessed.**
The outlier threshold is set at 25 robust sigma because measurement on real BTC data
showed that z > 10 fires on 0.2–0.4% of perfectly normal bars (crypto returns are
fat-tailed) while nothing at all exceeded z > 30. A detector that cries wolf on
ordinary volatility is worse than no detector.

---

## What the measurements say

The system is built to answer whether it can predict anything, and to report the
answer whatever it is. So far the answer is mostly **no**, and that is stated here
rather than buried.

| Question | Measured answer |
|---|---|
| Do classical directional patterns beat the market's own drift? | **No.** 9 of 342 pattern/asset/timeframe/horizon combinations were informative, and every one was direction-*neutral* (volume anomaly, compression, expansion). Breakouts, structure breaks, divergences, trend continuation, liquidity sweeps and momentum exhaustion all failed. |
| Do event chains predict anything? | **No.** 84 chains occurred often enough to test; none survived correction. |
| Does the current state have historical analogues? | **Sometimes.** BTC: insufficient evidence (16 comparable moments in 8,548). ETH: 200 analogues rose 36% against a 52% baseline. SOL: matches baseline. |
| Does news move prices? | **Unknown.** RSS carries a week of history; after thinning, 6 events in the largest category against a 25-event minimum. Reports insufficient evidence and accumulates. |
| **Do any of the eight models beat a baseline?** | **No.** 0 of 8 against climatology across 160 slices on three assets. All 8 "beat" persistence — but so do the models that abstain entirely, which is why persistence is not the standard. |
| Is that just the wrong horizon? | **No.** Re-run across 48 configurations — three timeframes, forecast reaches from 3 hours to 60 days, **2,032 slices — 0 passed.** Exactly one row reached p < 0.05 before correction. The largest apparent skills all came from the smallest samples and collapsed as n grew. |
| What scored best across that grid? | **Saying nothing.** `sentiment` abstains on 100% of points and was still the top-scoring model in 34 of 48 configurations. |
| Does calibrating the models help? | **Almost never.** 3 of 42 (model, regime) curves improved held-out calibration; 21 made it worse and were discarded. Climatology got worse in all four of its regimes. |
| **Does the ensemble publish anything?** | **No.** 0 published and 0 super predictions across 2,097 evaluation points on three assets. Six of the gate's nine conditions fail at every point. |
| Does anything survive walk-forward folding? | **No.** 0 models pass in every fold, 0 in any fold, across three assets × five folds × two fold schemes. Fold-to-fold skill swings roughly an order of magnitude more than it averages. |
| Does any model read the future? | **No** — and this is tested, not assumed. A deliberately leaky pipeline is caught; the same model on a clean pipeline is not. Four of eight models come back `INCONCLUSIVE` rather than `CLEAN`, because they abstain and so cannot be tested at all. |
| Does the learning loop learn anything? | **Partly, and not the part that matters.** Over 8,100 resolved predictions it granted **0** of 160 weight slices a non-zero weight, and adopted **6** of 42 calibration curves. It learned that six forecasters are miscalibrated in specific regimes — including climatology in downtrends — not that any model is worth trusting. |
| What is the single clearest measured result? | **Saying nothing beats climatology, and climatology beats every opinion.** The three best forecasters by Brier score exactly 0.6667 — the uniform distribution — because they abstain on every point. |
| What does the dashboard show? | **"Insufficient evidence", with the specific conditions that failed.** Three of the super-prediction gate's nine conditions pass; the rest are reported with their numbers. That is the honest rendering of everything above. |
| How much does it alert? | **422 of 1,238 raised, over 29 days across three assets** — 66% suppressed, and reported as a digest rather than swallowed. Zero of those were directional. |

What *does* survive measurement is volatility clustering: volume spikes and range
compression genuinely precede larger-than-usual movement. They say nothing about
direction, and the system does not pretend otherwise.

Every one of these negative results is produced by machinery that has been shown to
report a positive one when a positive one exists — an oracle model is certified by the
evaluator, a miscalibrated forecaster is corrected by the calibrator, a skilled and
agreeing panel clears the gate. Without that, "we found nothing" would be
indistinguishable from "our detector is broken".

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

431 tests, no network and no infrastructure required — ingestion, validation and
failover run against a deterministic synthetic provider with injectable faults
(gaps, duplicates, malformed bars, price spikes, outages), and every indicator is
checked against an independently written reference implementation.

A separate suite verifies the live provider contracts and is excluded by default,
because a test that needs an exchange to be up is a test that will eventually fail
for reasons unrelated to the code:

```bash
pytest -m network
```

---

## What is deliberately not here

Phases 6–12 — the model ensemble, backtesting, the learning loop, the dashboard and
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
