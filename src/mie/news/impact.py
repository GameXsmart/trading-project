"""Event impact: estimated, then measured.

Requirement §9 asks what a news event could realistically affect — magnitude,
direction, affected assets, expected duration, uncertainty. This module provides two
things that must not be confused:

* :class:`EventImpactModel` produces an **estimate** from priors. It is a hypothesis.
* :class:`ImpactValidator` **measures** what actually happened to prices after events
  of each kind, and reports whether the estimate has any support.

The gate for this phase is explicitly that impact estimates are *validated against
realised post-event volatility, not asserted*. Priors alone would be the assertion.
The validator is the phase.

**The measurement is volatility, not direction.** Whether news moves prices *at all*
is a far easier question than which way, it needs a much smaller sample, and it is the
one worth answering first. A story that reliably precedes elevated volatility is
useful even when its direction is unpredictable — it says *the next few hours are
unusually uncertain*, which is a legitimate input to a confidence estimate. Directional
impact is measured too, but with the expectation that it will be weak: everything
Phase 4 measured points that way.

**The comparison is against the asset's own recent volatility**, not against a fixed
threshold. Crypto volatility varies by an order of magnitude between regimes, so "2%
in an hour" means completely different things in different months. The ratio of
realised volatility after the event to the volatility that preceded it is the only
comparison that survives regime change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import median

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc
from mie.core.types import Candle
from mie.news.types import EventCategory, NewsEvent
from mie.patterns.statistics import (
    ProportionEstimate,
    benjamini_hochberg,
    compare_to_baseline,
)

log = get_logger(__name__)

__all__ = [
    "CATEGORY_IMPACT_PRIORS",
    "EventImpactModel",
    "ImpactEstimate",
    "ImpactMeasurement",
    "ImpactValidator",
]

#: Prior expectations per category: (volatility multiple, directional bias, hours).
#:
#: These are *hypotheses*, stated in advance so the validator can test them rather
#: than being fitted to the data it is measured on. The direction column is
#: deliberately conservative — even a hack does not reliably send price down over a
#: fixed window, and claiming otherwise before measuring would be exactly the error
#: this module exists to avoid.
CATEGORY_IMPACT_PRIORS: dict[EventCategory, tuple[float, float, float]] = {
    EventCategory.SECURITY_INCIDENT: (1.8, -0.35, 12.0),
    EventCategory.ETF: (1.6, +0.25, 24.0),
    EventCategory.REGULATION: (1.5, 0.0, 24.0),
    EventCategory.ENFORCEMENT: (1.5, -0.25, 18.0),
    EventCategory.MACRO: (1.6, 0.0, 12.0),
    EventCategory.EXCHANGE: (1.4, -0.15, 12.0),
    EventCategory.PROTOCOL: (1.2, +0.10, 24.0),
    EventCategory.LISTING: (1.3, +0.20, 6.0),
    EventCategory.ADOPTION: (1.1, +0.10, 24.0),
    EventCategory.PARTNERSHIP: (1.1, +0.10, 12.0),
    EventCategory.FUNDING: (1.0, 0.0, 12.0),
    EventCategory.MARKET_MOVE: (1.2, 0.0, 6.0),
    EventCategory.OTHER: (1.0, 0.0, 6.0),
}

#: Realised volatility above this multiple of the pre-event level counts as "elevated".
_ELEVATED_MULTIPLE = 1.5

#: Bars of pre-event history used as each event's own volatility baseline.
_BASELINE_BARS = 48

#: Below this many usable events, no claim is made about a category.
_MIN_EVENTS = 25


@dataclass(frozen=True, slots=True)
class ImpactEstimate:
    """What an event might do, before measurement. A hypothesis, not a finding."""

    event_cluster_id: str
    category: EventCategory
    assets: tuple[str, ...]
    #: Expected realised volatility as a multiple of the recent level.
    volatility_multiple: float
    #: Directional lean in [-1, 1]. Zero means "movement, direction unknown".
    direction_bias: float
    expected_duration_hours: float
    #: How much to trust this estimate. Low until the category has been validated.
    confidence: float
    grounded_in_measurement: bool = False
    note: str = ""

    def summary(self) -> str:
        basis = "measured" if self.grounded_in_measurement else "prior only — unvalidated"
        direction = (
            "no directional claim"
            if abs(self.direction_bias) < 0.05
            else f"lean {self.direction_bias:+.2f}"
        )
        return (
            f"{self.category}: ~{self.volatility_multiple:.1f}x volatility for "
            f"~{self.expected_duration_hours:.0f}h, {direction}, "
            f"confidence {self.confidence:.2f} ({basis})"
        )


@dataclass(slots=True)
class ImpactMeasurement:
    """What actually happened after events of one category."""

    category: EventCategory
    asset: str
    horizon_hours: int
    events: int
    #: Median realised-volatility ratio (post-event / pre-event).
    median_volatility_ratio: float
    mean_volatility_ratio: float
    #: Rate of "elevated volatility" outcomes, against the unconditional rate.
    elevated: ProportionEstimate | None
    #: Directional outcome rate, tested the same way.
    directional: ProportionEstimate | None
    median_return_pct: float
    sample_start: datetime | None = None
    sample_end: datetime | None = None

    @property
    def has_evidence(self) -> bool:
        return self.events >= _MIN_EVENTS and self.elevated is not None

    @property
    def moves_volatility(self) -> bool:
        """Whether events of this kind demonstrably precede *elevated* volatility.

        The rate must exceed the baseline, not merely differ from it. The underlying
        test is two-sided, and treating "significantly different" as "elevated" is a
        real error: a category followed by *less* volatility than usual would be
        certified as a volatility signal pointing the wrong way.
        """
        return (
            self.has_evidence
            and self.elevated is not None
            and self.elevated.significant
            and self.elevated.interval_excludes_baseline
            and self.elevated.rate > self.elevated.baseline
        )

    @property
    def suppresses_volatility(self) -> bool:
        """Significantly *calmer* than usual after these events.

        A real finding rather than a null one — it says the market treats this
        category as resolving uncertainty rather than creating it — so it is named
        instead of being folded into "no impact".
        """
        return (
            self.has_evidence
            and self.elevated is not None
            and self.elevated.significant
            and self.elevated.interval_excludes_baseline
            and self.elevated.rate < self.elevated.baseline
        )

    @property
    def moves_direction(self) -> bool:
        """Whether price moved the way the category's prior expects, more often than
        the market does anyway. One-sided, for the same reason as above."""
        return (
            self.has_evidence
            and self.directional is not None
            and self.directional.significant
            and self.directional.interval_excludes_baseline
            and self.directional.rate > self.directional.baseline
        )

    @property
    def verdict(self) -> str:
        if not self.has_evidence:
            return f"insufficient evidence ({self.events} events, need {_MIN_EVENTS})"
        if self.moves_direction:
            return "moves price directionally"
        if self.moves_volatility:
            return "precedes elevated volatility, direction unpredictable"
        if self.suppresses_volatility:
            return "followed by calmer-than-usual conditions"
        return "no measurable impact"

    def summary(self) -> str:
        elevated = (
            f"{self.elevated.rate:.0%} vs {self.elevated.baseline:.0%} baseline"
            if self.elevated
            else "n/a"
        )
        return (
            f"{self.category} {self.asset} +{self.horizon_hours}h: n={self.events}, "
            f"vol ratio {self.median_volatility_ratio:.2f}x, elevated {elevated} "
            f"-> {self.verdict}"
        )


class EventImpactModel:
    """Estimates the potential impact of an event.

    Uses measured statistics where they exist and priors where they do not — and says
    which, every time. An estimate resting on an untested prior is reported with low
    confidence and ``grounded_in_measurement=False`` so no consumer can mistake a
    hypothesis for a finding.
    """

    def __init__(self, measurements: Sequence[ImpactMeasurement] = ()) -> None:
        self._measured: dict[tuple[EventCategory, str], ImpactMeasurement] = {}
        for measurement in measurements:
            if measurement.has_evidence:
                self._measured[(measurement.category, measurement.asset)] = measurement

    def estimate(self, event: NewsEvent, asset: str | None = None) -> ImpactEstimate:
        prior_vol, prior_direction, prior_hours = CATEGORY_IMPACT_PRIORS.get(
            event.category, (1.0, 0.0, 6.0)
        )
        assets = tuple(event.assets) or ("MARKET",)
        target = (asset or (assets[0] if assets else "MARKET")).upper()
        measurement = self._measured.get((event.category, target)) or self._measured.get(
            (event.category, "ANY")
        )

        # Importance scales the expected magnitude: a story on one outlet is not the
        # same event as one on seven, even within a category.
        scale = 0.6 + 0.8 * event.importance

        if measurement is None:
            return ImpactEstimate(
                event_cluster_id=event.cluster_id,
                category=event.category,
                assets=assets,
                volatility_multiple=round(1.0 + (prior_vol - 1.0) * scale, 3),
                direction_bias=round(prior_direction * event.importance, 3),
                expected_duration_hours=prior_hours,
                # Deliberately low: nothing here has been tested against prices.
                confidence=round(min(0.35, 0.15 + 0.2 * event.confidence), 3),
                grounded_in_measurement=False,
                note="no validated measurement for this category; prior only",
            )

        # Measured: use the observed ratio, and only claim direction if measured.
        directional = (
            prior_direction if measurement.moves_direction else 0.0
        )
        return ImpactEstimate(
            event_cluster_id=event.cluster_id,
            category=event.category,
            assets=assets,
            volatility_multiple=round(
                1.0 + (measurement.median_volatility_ratio - 1.0) * scale, 3
            ),
            direction_bias=round(directional * event.importance, 3),
            expected_duration_hours=measurement.horizon_hours,
            confidence=round(min(0.8, 0.3 + 0.5 * event.confidence), 3),
            grounded_in_measurement=True,
            note=measurement.verdict,
        )


class ImpactValidator:
    """Measures what actually followed events of each category.

    This is the phase's gate. Everything it reports is derived from price data; where
    the sample is too thin it says so rather than producing a number.
    """

    def __init__(
        self,
        horizons_hours: Sequence[int] = (6, 24),
        baseline_bars: int = _BASELINE_BARS,
        elevated_multiple: float = _ELEVATED_MULTIPLE,
        min_events: int = _MIN_EVENTS,
        false_discovery_rate: float = 0.05,
    ) -> None:
        self.horizons_hours = tuple(horizons_hours)
        self.baseline_bars = baseline_bars
        self.elevated_multiple = elevated_multiple
        self.min_events = min_events
        self.false_discovery_rate = false_discovery_rate

    def validate(
        self,
        events: Sequence[NewsEvent],
        candles: Sequence[Candle],
        asset: str,
        timeframe: Timeframe = Timeframe.H1,
    ) -> list[ImpactMeasurement]:
        """Measure post-event volatility and direction, by category.

        Only events genuinely about ``asset`` are used, and only those with price data
        on both sides — a baseline window before and a full horizon after.
        """
        final = [c for c in candles if c.is_final]
        if len(final) < self.baseline_bars + max(self.horizons_hours) + 2:
            return []

        index_of = _bar_index(final, timeframe)
        relevant = [e for e in events if e.relevance_for(asset) >= 0.5 and not e.is_recycled]

        raw: list[ImpactMeasurement] = []
        for category in EventCategory:
            in_category = [e for e in relevant if e.category is category]
            if not in_category:
                continue
            for horizon in self.horizons_hours:
                measurement = self._measure(
                    in_category, final, index_of, category, asset, horizon, timeframe
                )
                if measurement is not None:
                    raw.append(measurement)

        return self._apply_fdr(raw)

    def _measure(
        self,
        events: Sequence[NewsEvent],
        candles: Sequence[Candle],
        index_of: dict[datetime, int],
        category: EventCategory,
        asset: str,
        horizon_hours: int,
        timeframe: Timeframe,
    ) -> ImpactMeasurement | None:
        bars = max(1, int(horizon_hours * 3600 / timeframe.seconds))
        ratios: list[float] = []
        returns: list[float] = []
        elevated_hits = 0
        used: list[datetime] = []

        for event in sorted(events, key=lambda e: e.published_at):
            index = _locate(index_of, candles, event.published_at, timeframe)
            if index is None:
                continue
            if index < self.baseline_bars or index + bars >= len(candles):
                continue
            # Thin overlapping events: two stories an hour apart share almost their
            # whole forward window and are not independent observations.
            if used and (event.published_at - used[-1]).total_seconds() < horizon_hours * 3600:
                continue

            before = _realised_volatility(candles[index - self.baseline_bars : index])
            after = _realised_volatility(candles[index : index + bars + 1])
            if before <= 0 or after <= 0:
                continue

            ratio = after / before
            ratios.append(ratio)
            elevated_hits += int(ratio >= self.elevated_multiple)
            entry = candles[index].close
            exit_close = candles[index + bars].close
            if entry > 0:
                returns.append((exit_close - entry) / entry * 100.0)
            used.append(event.published_at)

        if len(ratios) < 5:
            return None

        baseline_elevated, baseline_total = self._baseline_elevated(candles, bars)
        elevated = compare_to_baseline(
            elevated_hits, len(ratios), baseline_elevated, baseline_total
        )

        # Directional test: did price move the way the category's prior expects?
        prior_direction = CATEGORY_IMPACT_PRIORS.get(category, (1.0, 0.0, 6.0))[1]
        directional: ProportionEstimate | None = None
        if abs(prior_direction) > 0.05 and returns:
            expected = sum(
                1 for r in returns if (r > 0) == (prior_direction > 0)
            )
            base_up, base_total = _baseline_direction(candles, bars, prior_direction > 0)
            directional = compare_to_baseline(expected, len(returns), base_up, base_total)

        return ImpactMeasurement(
            category=category,
            asset=asset.upper(),
            horizon_hours=horizon_hours,
            events=len(ratios),
            median_volatility_ratio=round(median(ratios), 3),
            mean_volatility_ratio=round(sum(ratios) / len(ratios), 3),
            elevated=elevated,
            directional=directional,
            median_return_pct=round(median(returns), 4) if returns else 0.0,
            sample_start=used[0] if used else None,
            sample_end=used[-1] if used else None,
        )

    def _baseline_elevated(
        self, candles: Sequence[Candle], bars: int
    ) -> tuple[int, int]:
        """How often volatility is 'elevated' at a randomly chosen moment.

        Without this the elevated-volatility rate is uninterpretable: volatility
        clusters, so *some* fraction of all moments show a rising ratio regardless of
        whether anything was reported.
        """
        hits = total = 0
        for index in range(self.baseline_bars, len(candles) - bars, max(1, bars)):
            before = _realised_volatility(candles[index - self.baseline_bars : index])
            after = _realised_volatility(candles[index : index + bars + 1])
            if before <= 0 or after <= 0:
                continue
            total += 1
            hits += int(after / before >= self.elevated_multiple)
        return hits, total

    def _apply_fdr(self, measurements: Sequence[ImpactMeasurement]) -> list[ImpactMeasurement]:
        """Correct across every category and horizon tested at once."""
        tests: list[tuple[int, str, float]] = []
        for position, measurement in enumerate(measurements):
            if measurement.elevated is not None:
                tests.append((position, "elevated", measurement.elevated.p_value))
            if measurement.directional is not None:
                tests.append((position, "directional", measurement.directional.p_value))
        if not tests:
            return list(measurements)

        flags = benjamini_hochberg([p for _, _, p in tests], self.false_discovery_rate)
        from dataclasses import replace

        result = list(measurements)
        for (position, kind, _), significant in zip(tests, flags, strict=True):
            current = result[position]
            if kind == "elevated" and current.elevated is not None:
                current.elevated = replace(current.elevated, significant=significant)
            elif kind == "directional" and current.directional is not None:
                current.directional = replace(current.directional, significant=significant)
        return result


# ---------------------------------------------------------------------- helpers


def _realised_volatility(candles: Sequence[Candle]) -> float:
    """Standard deviation of log returns over a window, in percent."""
    closes = [c.close for c in candles if c.close > 0]
    if len(closes) < 3:
        return 0.0
    returns = [
        math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
    ]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance) * 100.0


def _bar_index(candles: Sequence[Candle], timeframe: Timeframe) -> dict[datetime, int]:
    return {c.open_time: i for i, c in enumerate(candles)}


def _locate(
    index_of: dict[datetime, int],
    candles: Sequence[Candle],
    moment: datetime,
    timeframe: Timeframe,
) -> int | None:
    """Index of the first bar that opens at or after ``moment``.

    Deliberately *after*, not the containing bar: a story published mid-bar cannot
    have influenced the part of that bar which already happened, and starting the
    measurement inside it would credit the event with price action that preceded it.
    """
    aligned = timeframe.ceil(ensure_utc(moment))
    for offset in range(0, 8):
        candidate = index_of.get(aligned + timeframe.delta * offset)
        if candidate is not None:
            return candidate
    return None


def _baseline_direction(
    candles: Sequence[Candle], bars: int, upward: bool
) -> tuple[int, int]:
    """Unconditional rate of the expected direction over non-overlapping windows."""
    hits = total = 0
    for index in range(0, len(candles) - bars, max(1, bars)):
        entry = candles[index].close
        if entry <= 0:
            continue
        total += 1
        moved_up = candles[index + bars].close > entry
        hits += int(moved_up == upward)
    return hits, total
