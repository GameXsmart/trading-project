"""Event impact: the estimate, and the measurement that must justify it.

Phase 5's gate is that impact is *validated against realised post-event volatility,
not asserted*. Real feeds cannot exercise that yet — RSS carries about a week of
history, and a week does not contain enough events of any one category to support a
claim. So the validator is tested against synthetic price series where the ground
truth is known by construction:

* inject a genuine volatility jump after events and confirm it is detected;
* inject nothing and confirm it is *not* detected.

The second matters more than the first. A validator that finds an effect in pure noise
would certify anything, and the whole point of this layer is to withhold certification.
"""

from __future__ import annotations

from datetime import timedelta

from tests.conftest import FIXED_NOW

from mie.core.timeframes import Timeframe
from mie.core.types import Candle
from mie.news.impact import (
    CATEGORY_IMPACT_PRIORS,
    EventImpactModel,
    ImpactValidator,
    _realised_volatility,
)
from mie.news.types import AssetRelevance, EventCategory, NewsEvent, Sentiment

# ---------------------------------------------------------------------- helpers


def event(
    category: EventCategory,
    hours_ago: float,
    asset: str = "BTC",
    importance: float = 0.6,
    cluster: str | None = None,
) -> NewsEvent:
    return NewsEvent(
        cluster_id=cluster or f"{category}-{hours_ago}",
        title=f"{category} story",
        url="https://example.test/x",
        published_at=FIXED_NOW - timedelta(hours=hours_ago),
        sources=["coindesk"],
        category=category,
        sentiment=Sentiment.NEUTRAL,
        relevance=[AssetRelevance(asset=asset, score=0.9, in_title=True)],
        importance=importance,
        confidence=0.7,
    )


def series(
    bars: int,
    volatile_at: set[int] | None = None,
    volatile_bars: int = 6,
    quiet_pct: float = 0.15,
    loud_pct: float = 2.5,
) -> list[Candle]:
    """A price series that is quiet except in explicitly noisy stretches.

    ``volatile_at`` gives the bar indices where a burst of volatility begins, so a
    test can place bursts exactly where its events are and know the answer in advance.
    """
    volatile_at = volatile_at or set()
    loud: set[int] = set()
    for start in volatile_at:
        loud.update(range(start, start + volatile_bars))

    price = 100.0
    candles: list[Candle] = []
    start_time = FIXED_NOW - timedelta(hours=bars)
    for i in range(bars):
        step = (loud_pct if i in loud else quiet_pct) / 100.0
        # Deterministic alternating moves: volatility without net drift, so the
        # directional test is not accidentally handed a trend.
        price *= 1.0 + step * (1 if i % 2 else -1)
        span = price * step
        candles.append(
            Candle(
                asset="BTC",
                source="test",
                timeframe=Timeframe.H1,
                open_time=start_time + timedelta(hours=i),
                open=price,
                high=price + span,
                low=price - span,
                close=price,
                volume=100.0,
                is_final=True,
            )
        )
    return candles


def events_at(bar_indices: list[int], total_bars: int, category: EventCategory) -> list[NewsEvent]:
    """Events placed so each lands just before the given bar index."""
    return [
        event(category, hours_ago=total_bars - index, cluster=f"c{index}")
        for index in bar_indices
    ]


# ------------------------------------------------------------------- mechanics


class TestRealisedVolatility:
    def test_a_quiet_series_has_lower_volatility_than_a_noisy_one(self) -> None:
        quiet = series(60)
        loud = series(60, volatile_at={0}, volatile_bars=60)
        assert _realised_volatility(loud) > _realised_volatility(quiet) * 5

    def test_too_few_bars_yields_zero_rather_than_a_spurious_number(self) -> None:
        assert _realised_volatility(series(2)) == 0.0


class TestValidatorDetection:
    """Ground truth is known by construction, so the verdict can be checked."""

    def test_a_real_volatility_effect_is_detected(self) -> None:
        """Events genuinely followed by turbulence must be certified.

        Uses a category whose directional prior is zero, so the directional test is
        skipped entirely and the verdict can only come from the volatility finding.
        The synthetic bursts carry a little incidental drift, and with a directional
        category the validator correctly reports that instead — accurate, but not what
        this test is about.
        """
        assert CATEGORY_IMPACT_PRIORS[EventCategory.REGULATION][1] == 0.0
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        events = events_at(spots, total, EventCategory.REGULATION)

        results = ImpactValidator(horizons_hours=(6,)).validate(events, candles, "BTC")
        assert results
        measurement = results[0]
        assert measurement.events >= 25
        assert measurement.median_volatility_ratio > 2.0
        assert measurement.moves_volatility, measurement.summary()
        assert "elevated volatility" in measurement.verdict

    def test_no_effect_is_not_certified(self) -> None:
        """The test that matters most. A validator that finds an effect in a series
        with none would certify anything."""
        total = 1400
        spots = list(range(200, 1300, 30))
        # Volatility bursts exist, but nowhere near the events.
        candles = series(total, volatile_at={i + 15 for i in spots}, volatile_bars=3)
        events = events_at(spots, total, EventCategory.SECURITY_INCIDENT)

        results = ImpactValidator(horizons_hours=(6,)).validate(events, candles, "BTC")
        assert results
        assert not results[0].moves_volatility, results[0].summary()
        # Calmer-than-usual is itself a finding, and is named rather than folded into
        # "no impact" — but it is emphatically not "precedes elevated volatility".
        assert "elevated volatility" not in results[0].verdict

    def test_a_flat_series_produces_no_claim(self) -> None:
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total)  # uniformly quiet: nothing to find anywhere
        events = events_at(spots, total, EventCategory.ETF)
        results = ImpactValidator(horizons_hours=(6,)).validate(events, candles, "BTC")
        assert not any(m.moves_volatility for m in results)


class TestValidatorGuards:
    def test_thin_samples_are_refused(self) -> None:
        """Six events cannot support a claim, however suggestive they look."""
        total = 600
        spots = [200, 260, 320, 380, 440, 500]
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        events = events_at(spots, total, EventCategory.ETF)

        results = ImpactValidator(horizons_hours=(6,)).validate(events, candles, "BTC")
        for measurement in results:
            assert not measurement.has_evidence
            assert "insufficient evidence" in measurement.verdict

    def test_overlapping_events_are_thinned(self) -> None:
        """Two stories an hour apart share almost their whole forward window and are
        not independent observations."""
        total = 1400
        clustered = [i for base in range(200, 1300, 60) for i in (base, base + 1, base + 2)]
        candles = series(total, volatile_at=set(clustered), volatile_bars=8)
        events = events_at(clustered, total, EventCategory.ETF)

        results = ImpactValidator(horizons_hours=(24,)).validate(events, candles, "BTC")
        assert results
        assert results[0].events < len(events) / 2

    def test_events_outside_the_price_window_are_skipped(self) -> None:
        candles = series(300, volatile_at={100}, volatile_bars=8)
        far_future = [event(EventCategory.ETF, hours_ago=-500)]
        assert ImpactValidator().validate(far_future, candles, "BTC") == []

    def test_irrelevant_assets_are_excluded(self) -> None:
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        events = [
            event(EventCategory.ETF, total - i, asset="SOL", cluster=f"c{i}") for i in spots
        ]
        assert ImpactValidator().validate(events, candles, "BTC") == []

    def test_recycled_stories_are_excluded(self) -> None:
        """A re-run is not new information reaching the market."""
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        events = [
            e.model_copy(update={"is_recycled": True})
            for e in events_at(spots, total, EventCategory.ETF)
        ]
        assert ImpactValidator().validate(events, candles, "BTC") == []

    def test_measurement_starts_after_the_event_not_inside_its_bar(self) -> None:
        """A story published mid-bar cannot have caused the part of that bar which
        already happened; measuring from inside it would credit the event with price
        action that preceded it."""
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        # Shift every event 30 minutes into the preceding bar.
        events = [
            e.model_copy(update={"published_at": e.published_at - timedelta(minutes=30)})
            for e in events_at(spots, total, EventCategory.SECURITY_INCIDENT)
        ]
        results = ImpactValidator(horizons_hours=(6,)).validate(events, candles, "BTC")
        assert results
        assert results[0].events >= 25


# ----------------------------------------------------------------- estimates


class TestImpactEstimates:
    def test_an_unvalidated_category_is_flagged_as_prior_only(self) -> None:
        """A hypothesis must never be presentable as a finding."""
        estimate = EventImpactModel().estimate(event(EventCategory.SECURITY_INCIDENT, 1))
        assert not estimate.grounded_in_measurement
        assert estimate.confidence <= 0.35
        assert "prior only" in estimate.summary()

    def test_a_validated_category_uses_the_measurement(self) -> None:
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        measurements = ImpactValidator(horizons_hours=(6,)).validate(
            events_at(spots, total, EventCategory.SECURITY_INCIDENT), candles, "BTC"
        )
        model = EventImpactModel(measurements)
        estimate = model.estimate(event(EventCategory.SECURITY_INCIDENT, 1))

        assert estimate.grounded_in_measurement
        assert estimate.confidence > 0.35
        assert estimate.volatility_multiple > 1.2

    def test_no_directional_claim_without_measured_directional_evidence(self) -> None:
        """Categories carry a directional prior, but it is only ever published when
        the measurement supports it."""
        estimate = EventImpactModel().estimate(event(EventCategory.SECURITY_INCIDENT, 1))
        prior_direction = CATEGORY_IMPACT_PRIORS[EventCategory.SECURITY_INCIDENT][1]
        assert prior_direction < 0
        assert abs(estimate.direction_bias) < abs(prior_direction)

    def test_importance_scales_the_expected_magnitude(self) -> None:
        """A story on one outlet is not the same event as one on seven."""
        model = EventImpactModel()
        minor = model.estimate(event(EventCategory.ETF, 1, importance=0.2))
        major = model.estimate(event(EventCategory.ETF, 1, importance=0.9))
        assert major.volatility_multiple > minor.volatility_multiple

    def test_every_category_has_a_declared_prior(self) -> None:
        for category in EventCategory:
            assert category in CATEGORY_IMPACT_PRIORS

    def test_estimates_are_bounded_and_sane(self) -> None:
        model = EventImpactModel()
        for category in EventCategory:
            estimate = model.estimate(event(category, 1, importance=1.0))
            assert estimate.volatility_multiple >= 1.0
            assert -1.0 <= estimate.direction_bias <= 1.0
            assert 0.0 <= estimate.confidence <= 1.0
            assert estimate.expected_duration_hours > 0


class TestBaselineComparison:
    def test_elevated_rate_is_measured_against_the_unconditional_rate(self) -> None:
        """Volatility clusters, so some fraction of all moments show a rising ratio
        whether or not anything was reported. Without that baseline the event rate is
        uninterpretable."""
        total = 1400
        spots = list(range(200, 1300, 30))
        candles = series(total, volatile_at=set(spots), volatile_bars=8)
        results = ImpactValidator(horizons_hours=(6,)).validate(
            events_at(spots, total, EventCategory.ETF), candles, "BTC"
        )
        assert results
        estimate = results[0].elevated
        assert estimate is not None
        assert 0.0 < estimate.baseline < 1.0
        assert estimate.trials == results[0].events
