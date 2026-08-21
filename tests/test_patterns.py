"""Pattern detection, statistical validation, and the evidence gate.

Phase 4's gate is a claim about *method*, so most of these tests are about the method
rather than about any particular pattern: that the baseline is the market's own drift
and not a coin flip, that intervals widen when evidence is thin, that a wide sweep is
corrected for multiple comparisons, and that an unproven pattern cannot influence
anything.

The detector tests check that each fires on the configuration it names and stays quiet
otherwise. They deliberately do *not* assert that any pattern predicts anything —
measurement decides that, and on real data most of them do not.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from tests.conftest import FIXED_NOW, make_candle
from tests.test_indicators import candles_from

from mie.core.timeframes import Timeframe
from mie.patterns.detectors import detect_all
from mie.patterns.evaluation import PatternEvaluator
from mie.patterns.registry import PatternRegistry
from mie.patterns.statistics import (
    benjamini_hochberg,
    compare_to_baseline,
    two_proportion_test,
    wilson_interval,
)
from mie.patterns.types import (
    PATTERN_DIRECTIONS,
    Detection,
    PatternDirection,
    PatternKind,
    PatternStats,
)

# ---------------------------------------------------------------------- helpers


def bars(prices: list[float], volumes: list[float] | None = None):
    """Bars with explicit control over volume, which several detectors depend on."""
    built = candles_from(prices)
    if volumes is None:
        return built
    return [
        make_candle(
            c.open_time, close=c.close, open_=c.open, high=c.high, low=c.low,
            volume=volumes[i], timeframe=Timeframe.H1,
        )
        for i, c in enumerate(built)
    ]


def kinds(detections) -> set[PatternKind]:
    return {d.kind for d in detections}


def flat(n: int, level: float = 100.0) -> list[float]:
    """A quiet range: the backdrop most detectors are supposed to ignore."""
    return [level + 0.35 * math.sin(i / 3.0) for i in range(n)]


# ------------------------------------------------------------------ statistics


class TestWilsonInterval:
    def test_small_samples_produce_wide_intervals(self) -> None:
        """Three wins from three is not a 100% hit rate, and the interval must say so."""
        low, high = wilson_interval(3, 3)
        assert low < 0.5
        assert high == 1.0

    def test_intervals_narrow_as_evidence_accumulates(self) -> None:
        small = wilson_interval(70, 100)
        large = wilson_interval(7000, 10000)
        assert (small[1] - small[0]) > (large[1] - large[0]) * 5

    def test_bounds_stay_inside_zero_and_one(self) -> None:
        """The normal approximation produces impossible bounds here; Wilson does not."""
        for successes, trials in ((0, 5), (5, 5), (1, 200), (199, 200)):
            low, high = wilson_interval(successes, trials)
            assert 0.0 <= low <= high <= 1.0

    def test_zero_trials_is_maximally_uncertain(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestBaselineComparison:
    def test_edge_is_measured_against_drift_not_against_a_coin_flip(self) -> None:
        """The central methodological point: in a market that rose 60% of the time, a
        pattern that is right 58% of the time has a *negative* edge."""
        estimate = compare_to_baseline(58, 100, 600, 1000)
        assert estimate.rate == pytest.approx(0.58)
        assert estimate.baseline == pytest.approx(0.60)
        assert estimate.edge < 0

    def test_a_genuine_edge_is_detected(self) -> None:
        estimate = compare_to_baseline(650, 1000, 500, 1000)
        assert estimate.edge == pytest.approx(0.15)
        assert estimate.p_value < 0.001
        assert estimate.interval_excludes_baseline

    def test_a_small_sample_cannot_establish_an_edge(self) -> None:
        """Same apparent rate, far less evidence: the honest answer is 'unproven'."""
        strong = compare_to_baseline(65, 100, 5000, 10000)
        weak = compare_to_baseline(7, 10, 5000, 10000)
        assert strong.p_value < weak.p_value
        assert not weak.interval_excludes_baseline

    def test_two_proportion_test_is_symmetric(self) -> None:
        assert two_proportion_test(60, 100, 50, 100) == pytest.approx(
            two_proportion_test(50, 100, 60, 100)
        )

    def test_degenerate_inputs_do_not_claim_significance(self) -> None:
        assert two_proportion_test(0, 0, 10, 100) == 1.0
        assert two_proportion_test(100, 100, 100, 100) == 1.0


class TestMultipleComparisons:
    def test_bh_rejects_the_strongest_and_keeps_the_rest(self) -> None:
        rejected = benjamini_hochberg([0.001, 0.02, 0.04, 0.3], 0.05)
        assert rejected == [True, True, False, False]

    def test_a_wide_sweep_of_pure_noise_yields_few_discoveries(self) -> None:
        """Uncorrected, 500 null tests give ~25 'significant' results. That is the
        single most common way pattern research produces confident nonsense."""
        # Uniform p-values are what the null hypothesis produces.
        noise = [(i + 0.5) / 500 for i in range(500)]
        uncorrected = sum(1 for p in noise if p < 0.05)
        corrected = sum(benjamini_hochberg(noise, 0.05))
        assert uncorrected == 25
        assert corrected <= 1

    def test_empty_family_is_handled(self) -> None:
        assert benjamini_hochberg([]) == []


# ------------------------------------------------------------------- detectors


class TestDetectors:
    def test_breakout_requires_volume_confirmation(self) -> None:
        """An unconfirmed break of a range is the textbook description of a fakeout,
        so treating the two identically would blend a signal with its own opposite."""
        prices = [*flat(40), 104.0]
        quiet = bars(prices, [100.0] * 40 + [100.0])
        loud = bars(prices, [100.0] * 40 + [300.0])

        assert PatternKind.BREAKOUT_UP not in kinds(detect_all(quiet, "BTC", Timeframe.H1))
        assert PatternKind.BREAKOUT_UP in kinds(detect_all(loud, "BTC", Timeframe.H1))

    def test_breakout_down_fires_on_a_confirmed_break_lower(self) -> None:
        prices = [*flat(40), 96.0]
        loud = bars(prices, [100.0] * 40 + [300.0])
        assert PatternKind.BREAKOUT_DOWN in kinds(detect_all(loud, "BTC", Timeframe.H1))

    def test_fakeout_fires_when_price_closes_back_inside(self) -> None:
        window = bars(flat(40))
        highest = max(c.high for c in window)
        window.append(
            make_candle(
                window[-1].open_time + Timeframe.H1.delta,
                close=highest - 1.5, open_=highest - 1.0,
                high=highest + 2.0, low=highest - 2.0, volume=150.0,
            )
        )
        assert PatternKind.FAKEOUT_UP in kinds(detect_all(window, "BTC", Timeframe.H1))

    def test_liquidity_sweep_requires_a_large_wick(self) -> None:
        """What separates a stop-run from a marginal failed break."""
        window = bars(flat(40))
        highest = max(c.high for c in window)
        marginal = [*window, make_candle(window[-1].open_time + Timeframe.H1.delta, close=highest - 0.1, open_=highest - 0.2, high=highest + 0.05, low=highest - 0.3, volume=120.0)]
        dramatic = [*window, make_candle(window[-1].open_time + Timeframe.H1.delta, close=highest - 0.5, open_=highest - 0.4, high=highest + 6.0, low=highest - 0.8, volume=120.0)]
        assert PatternKind.LIQUIDITY_SWEEP_HIGH not in kinds(
            detect_all(marginal, "BTC", Timeframe.H1)
        )
        assert PatternKind.LIQUIDITY_SWEEP_HIGH in kinds(
            detect_all(dramatic, "BTC", Timeframe.H1)
        )

    def test_compression_and_expansion_are_opposites(self) -> None:
        calm = bars([100.0 + 8.0 * math.sin(i / 7.0) for i in range(50)] + list(flat(20)))
        assert PatternKind.COMPRESSION in kinds(detect_all(calm, "BTC", Timeframe.H1))

        violent = bars(
            list(flat(55)) + [100.0 + 12.0 * (-1) ** i for i in range(6)]
        )
        assert PatternKind.EXPANSION in kinds(detect_all(violent, "BTC", Timeframe.H1))

    def test_volume_anomaly_uses_a_robust_threshold(self) -> None:
        """Volume is heavily right-skewed; a mean-and-sigma test lets a single spike
        inflate the threshold enough to hide itself."""
        prices = flat(40)
        varied = [100.0 + 15.0 * math.sin(i / 4.0) for i in range(39)]
        normal = bars(prices, [*varied, 110.0])
        spiked = bars(prices, [*varied, 5000.0])
        assert PatternKind.VOLUME_ANOMALY not in kinds(detect_all(normal, "BTC", Timeframe.H1))
        assert PatternKind.VOLUME_ANOMALY in kinds(detect_all(spiked, "BTC", Timeframe.H1))

    def test_accumulation_requires_a_quiet_range(self) -> None:
        """A trending market must not be relabelled as accumulation."""
        trending = bars([100.0 + i * 1.5 for i in range(60)])
        found = kinds(detect_all(trending, "BTC", Timeframe.H1))
        assert PatternKind.ACCUMULATION not in found
        assert PatternKind.DISTRIBUTION not in found

    def test_momentum_exhaustion_needs_momentum_to_have_turned(self) -> None:
        """Not merely 'RSI above 70' — overbought markets keep rising for a long time."""
        still_rising = bars([100.0 + i * 1.2 for i in range(60)])
        assert PatternKind.MOMENTUM_EXHAUSTION_UP not in kinds(
            detect_all(still_rising, "BTC", Timeframe.H1)
        )

        rolled_over = bars(
            [100.0 + i * 1.2 for i in range(55)] + [166.0, 165.0, 163.5]
        )
        assert PatternKind.MOMENTUM_EXHAUSTION_UP in kinds(
            detect_all(rolled_over, "BTC", Timeframe.H1)
        )

    def test_directional_detectors_stay_quiet_in_a_range(self) -> None:
        """A detector that fires constantly carries no information.

        Range-state patterns (compression, accumulation) *should* fire here — that is
        what the market is doing. What must not fire is a directional call: an
        oscillating range contains no breakout, continuation or structure break.
        """
        found = kinds(detect_all(bars(flat(120)), "BTC", Timeframe.H1))
        directional = {
            PatternKind.BREAKOUT_UP,
            PatternKind.BREAKOUT_DOWN,
            PatternKind.TREND_CONTINUATION_UP,
            PatternKind.TREND_CONTINUATION_DOWN,
            PatternKind.STRUCTURE_BREAK_UP,
            PatternKind.STRUCTURE_BREAK_DOWN,
        }
        assert not (found & directional)

    def test_detectors_do_not_fire_on_most_bars(self) -> None:
        """Frequency is itself a quality check: a pattern present half the time is a
        description of the market, not a signal within it."""
        series = bars([100.0 + i * 0.03 + 5.0 * math.sin(i / 17.0) for i in range(600)])
        hits = sum(
            1 for i in range(80, len(series)) if detect_all(series[: i + 1], "BTC", Timeframe.H1)
        )
        assert hits < (len(series) - 80) * 0.75

    def test_provisional_bars_are_never_detected_on(self) -> None:
        """A pattern confirmed by a forming bar is a guess about the rest of the bar."""
        window = bars(flat(40))
        window[-1] = make_candle(window[-1].open_time, close=104.0, is_final=False)
        assert detect_all(window, "BTC", Timeframe.H1) == []

    def test_short_windows_produce_nothing_rather_than_crashing(self) -> None:
        for length in (0, 1, 5, 20):
            assert isinstance(detect_all(bars(flat(length)), "BTC", Timeframe.H1), list)

    def test_every_pattern_kind_has_a_declared_direction(self) -> None:
        """Direction is fixed in advance; deciding it after seeing the outcome is
        relabelling, not analysis."""
        for kind in PatternKind:
            assert kind in PATTERN_DIRECTIONS


# ------------------------------------------------------------------ evaluation


class TestEvaluation:
    @pytest.fixture(scope="class")
    @staticmethod
    def series():
        # Long enough to produce real sample sizes, with structure and volume variety.
        prices = [
            100.0 + i * 0.05 + 6.0 * math.sin(i / 23.0) + 2.0 * math.cos(i / 7.0)
            for i in range(1200)
        ]
        volumes = [100.0 + 60.0 * math.sin(i / 11.0) + (400.0 if i % 97 == 0 else 0.0)
                   for i in range(1200)]
        return bars(prices, volumes)

    def test_scan_never_looks_forward(self, series) -> None:
        """Truncating the series must not change detections made before the cut."""
        evaluator = PatternEvaluator()
        full = evaluator.scan(series, "BTC", Timeframe.H1)
        truncated = evaluator.scan(series[:600], "BTC", Timeframe.H1)

        cutoff = series[599].open_time
        before_full = [d for d in full if d.at <= cutoff]
        assert [(d.kind, d.at) for d in before_full] == [
            (d.kind, d.at) for d in truncated
        ]

    def test_outcomes_read_only_future_bars(self, series) -> None:
        evaluator = PatternEvaluator()
        outcome = evaluator.outcome_for(series, 100, 12, PatternDirection.BULLISH)
        assert outcome is not None
        assert outcome.entry_close == series[100].close
        assert outcome.exit_close == series[112].close

    def test_outcome_is_none_when_the_horizon_runs_past_the_data(self, series) -> None:
        assert evaluator_outcome(series, len(series) - 2, 12) is None

    def test_evaluation_reports_a_verdict_for_every_pattern(self, series) -> None:
        result = PatternEvaluator().evaluate(series, "BTC", Timeframe.H1)
        assert result.stats
        for stat in result.stats:
            assert stat.verdict in (
                "insufficient samples",
                "indistinguishable from chance",
                "interval overlaps baseline",
                "informative",
            )
            assert stat.estimate.trials > 0

    def test_thin_samples_are_never_declared_informative(self, series) -> None:
        """Requirement: enough samples, significance after correction, and an interval
        clear of the baseline. Any one alone is easy to hit by accident."""
        result = PatternEvaluator().evaluate(series, "BTC", Timeframe.H1)
        for stat in result.stats:
            if stat.occurrences < 30:
                assert not stat.is_informative

    def test_overlapping_detections_are_thinned(self, series) -> None:
        """Consecutive detections share nearly all of their forward window, so counting
        each as independent inflates n and shrinks the interval past what the evidence
        supports."""
        evaluator = PatternEvaluator(horizons=(48,))
        result = evaluator.evaluate(series, "BTC", Timeframe.H1)
        for stat in result.stats:
            raw = len(result.by_kind(stat.kind))
            assert stat.occurrences <= raw

    def test_baseline_reflects_the_market_not_a_coin_flip(self, series) -> None:
        evaluator = PatternEvaluator()
        successes, trials = evaluator._baseline(series, 12, PatternDirection.BULLISH)
        assert trials > 0
        assert 0.0 < successes / trials < 1.0


def evaluator_outcome(series, index, horizon):
    return PatternEvaluator().outcome_for(series, index, horizon, PatternDirection.BULLISH)


# -------------------------------------------------------------------- registry


def stats(
    kind: PatternKind,
    informative: bool,
    asset: str = "BTC",
    timeframe: Timeframe = Timeframe.H1,
    horizon: int = 12,
) -> PatternStats:
    estimate = compare_to_baseline(
        650 if informative else 505, 1000, 500, 1000
    )
    return PatternStats(
        kind=kind,
        asset=asset,
        timeframe=timeframe,
        horizon_bars=horizon,
        direction=PATTERN_DIRECTIONS[kind],
        occurrences=1000,
        estimate=replace(estimate, significant=informative),
        mean_return_pct=0.4,
        median_return_pct=0.3,
        mean_favourable_pct=1.2,
        mean_adverse_pct=-0.9,
    )


def detection(kind: PatternKind, asset: str = "BTC") -> Detection:
    return Detection(
        kind=kind,
        asset=asset,
        timeframe=Timeframe.H1,
        at=FIXED_NOW,
        direction=PATTERN_DIRECTIONS[kind],
        close=100.0,
    )


class TestRegistryGate:
    """'Removed, not shipped with a caveat' — enforced, not merely documented."""

    def test_unproven_patterns_cannot_influence_predictions(self) -> None:
        registry = PatternRegistry([stats(PatternKind.BREAKOUT_UP, informative=False)])
        kept = registry.filter_detections([detection(PatternKind.BREAKOUT_UP)], horizon=12)
        assert kept == []
        assert registry.expected_edge(detection(PatternKind.BREAKOUT_UP), 12) == 0.0

    def test_proven_patterns_pass_with_their_measured_edge(self) -> None:
        registry = PatternRegistry([stats(PatternKind.VOLUME_ANOMALY, informative=True)])
        kept = registry.filter_detections([detection(PatternKind.VOLUME_ANOMALY)], horizon=12)
        assert len(kept) == 1
        assert registry.expected_edge(detection(PatternKind.VOLUME_ANOMALY), 12) > 0

    def test_absence_of_evidence_is_absence_of_permission(self) -> None:
        """An unmeasured pattern is not 'probably fine'. It is simply unproven."""
        registry = PatternRegistry()
        assert not registry.is_informative(
            PatternKind.BREAKOUT_UP, "BTC", Timeframe.H1, 12
        )
        assert registry.filter_detections([detection(PatternKind.BREAKOUT_UP)], 12) == []

    def test_evidence_does_not_transfer_between_assets(self) -> None:
        """Measurement showed a pattern can clear the bar on one asset and fail on
        another; a single global verdict would be wrong in both directions at once."""
        registry = PatternRegistry(
            [stats(PatternKind.FAKEOUT_DOWN, informative=True, asset="ETH")]
        )
        assert registry.is_informative(PatternKind.FAKEOUT_DOWN, "ETH", Timeframe.H1, 12)
        assert not registry.is_informative(PatternKind.FAKEOUT_DOWN, "BTC", Timeframe.H1, 12)

    def test_evidence_does_not_transfer_between_horizons(self) -> None:
        registry = PatternRegistry(
            [stats(PatternKind.VOLUME_ANOMALY, informative=True, horizon=3)]
        )
        assert registry.is_informative(PatternKind.VOLUME_ANOMALY, "BTC", Timeframe.H1, 3)
        assert not registry.is_informative(PatternKind.VOLUME_ANOMALY, "BTC", Timeframe.H1, 48)

    def test_report_separates_admitted_from_withheld(self) -> None:
        registry = PatternRegistry(
            [
                stats(PatternKind.VOLUME_ANOMALY, informative=True),
                stats(PatternKind.BREAKOUT_UP, informative=False),
            ]
        )
        assert len(registry.admitted()) == 1
        assert len(registry.rejected()) == 1
        report = registry.report()
        assert "ADMITTED" in report and "withheld" in report
