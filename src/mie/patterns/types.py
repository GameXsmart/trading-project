"""Pattern vocabulary.

A *pattern* is a named, mechanically-detectable market configuration. A *detection* is
one occurrence of it. *Statistics* are what happened afterwards across every occurrence
in the sample.

The distinction matters because the first two are cheap and the third is the only one
with any evidential value. "This looks like a breakout" costs nothing to say; "this
configuration resolved upward 58% of the time against a 52% baseline, n=214,
CI 51-65%" is a claim that can be checked and can be wrong.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mie.core.timeframes import Timeframe, utcnow
from mie.patterns.statistics import ProportionEstimate

__all__ = [
    "Detection",
    "Outcome",
    "PatternDirection",
    "PatternKind",
    "PatternStats",
]


class PatternKind(StrEnum):
    """Detectable configurations, from requirement §5."""

    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    FAKEOUT_UP = "fakeout_up"
    FAKEOUT_DOWN = "fakeout_down"
    LIQUIDITY_SWEEP_HIGH = "liquidity_sweep_high"
    LIQUIDITY_SWEEP_LOW = "liquidity_sweep_low"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    TREND_CONTINUATION_UP = "trend_continuation_up"
    TREND_CONTINUATION_DOWN = "trend_continuation_down"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    MOMENTUM_EXHAUSTION_UP = "momentum_exhaustion_up"
    MOMENTUM_EXHAUSTION_DOWN = "momentum_exhaustion_down"
    VOLUME_ANOMALY = "volume_anomaly"
    STRUCTURE_BREAK_UP = "structure_break_up"
    STRUCTURE_BREAK_DOWN = "structure_break_down"


class PatternDirection(StrEnum):
    """What the pattern would imply if it carried information.

    Recorded *before* any measurement, so the test is a genuine prediction rather than
    a label fitted after seeing which way price went. A pattern whose conventional
    reading is bullish gets scored on whether price rose — even when measurement says
    it reliably falls, which is itself a finding worth keeping.
    """

    BULLISH = "bullish"
    BEARISH = "bearish"
    #: Implies movement without implying direction (compression, volume anomaly).
    NEUTRAL = "neutral"


#: The conventional reading of each pattern. Fixed in advance, never tuned to results.
PATTERN_DIRECTIONS: dict[PatternKind, PatternDirection] = {
    PatternKind.BREAKOUT_UP: PatternDirection.BULLISH,
    PatternKind.BREAKOUT_DOWN: PatternDirection.BEARISH,
    PatternKind.FAKEOUT_UP: PatternDirection.BEARISH,
    PatternKind.FAKEOUT_DOWN: PatternDirection.BULLISH,
    PatternKind.LIQUIDITY_SWEEP_HIGH: PatternDirection.BEARISH,
    PatternKind.LIQUIDITY_SWEEP_LOW: PatternDirection.BULLISH,
    PatternKind.COMPRESSION: PatternDirection.NEUTRAL,
    PatternKind.EXPANSION: PatternDirection.NEUTRAL,
    PatternKind.ACCUMULATION: PatternDirection.BULLISH,
    PatternKind.DISTRIBUTION: PatternDirection.BEARISH,
    PatternKind.TREND_CONTINUATION_UP: PatternDirection.BULLISH,
    PatternKind.TREND_CONTINUATION_DOWN: PatternDirection.BEARISH,
    PatternKind.BULLISH_DIVERGENCE: PatternDirection.BULLISH,
    PatternKind.BEARISH_DIVERGENCE: PatternDirection.BEARISH,
    PatternKind.MOMENTUM_EXHAUSTION_UP: PatternDirection.BEARISH,
    PatternKind.MOMENTUM_EXHAUSTION_DOWN: PatternDirection.BULLISH,
    PatternKind.VOLUME_ANOMALY: PatternDirection.NEUTRAL,
    PatternKind.STRUCTURE_BREAK_UP: PatternDirection.BULLISH,
    PatternKind.STRUCTURE_BREAK_DOWN: PatternDirection.BEARISH,
}


class Detection(BaseModel):
    """One occurrence of a pattern at one bar."""

    model_config = ConfigDict(frozen=True)

    kind: PatternKind
    asset: str
    timeframe: Timeframe
    #: Open time of the bar on which the pattern *completed*. Never a bar that had not
    #: closed yet — a pattern confirmed by a forming bar is a guess about the future.
    at: datetime
    direction: PatternDirection
    #: Detector's own view of how cleanly the conditions were met, in [0, 1]. Not a
    #: probability of success — that comes only from measurement.
    quality: float = 0.5
    close: float = 0.0
    detail: str = ""
    context: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display affordance
        return f"{self.kind} {self.asset} {self.timeframe} @ {self.at:%Y-%m-%d %H:%M}"


class Outcome(BaseModel):
    """What actually happened after a detection, over one forward horizon."""

    model_config = ConfigDict(frozen=True)

    detection_at: datetime
    horizon_bars: int
    entry_close: float
    exit_close: float
    return_pct: float
    max_favourable_pct: float
    max_adverse_pct: float
    #: Whether price moved in the direction the pattern implied. Undefined for
    #: direction-neutral patterns, which are scored on movement instead.
    resolved_as_expected: bool | None = None


class PatternStats(BaseModel):
    """Measured historical behaviour of one pattern on one asset and timeframe.

    This is the object the Phase 4 gate is written against. A pattern without one of
    these has not earned the right to influence anything downstream.
    """

    model_config = ConfigDict(frozen=True)

    kind: PatternKind
    asset: str
    timeframe: Timeframe
    horizon_bars: int
    direction: PatternDirection
    occurrences: int
    estimate: ProportionEstimate
    mean_return_pct: float
    median_return_pct: float
    mean_favourable_pct: float
    mean_adverse_pct: float
    sample_start: datetime | None = None
    sample_end: datetime | None = None
    computed_at: datetime = Field(default_factory=utcnow)

    @property
    def is_informative(self) -> bool:
        """Whether this pattern survived measurement.

        Requires all three: enough samples to say anything, statistical significance
        after multiple-comparison correction, and a confidence interval that clears
        the baseline. Any one alone is easy to achieve by accident.
        """
        return (
            self.occurrences >= 30
            and self.estimate.significant
            and self.estimate.interval_excludes_baseline
        )

    @property
    def verdict(self) -> str:
        if self.occurrences < 30:
            return "insufficient samples"
        if not self.estimate.significant:
            return "indistinguishable from chance"
        if not self.estimate.interval_excludes_baseline:
            return "interval overlaps baseline"
        return "informative"

    def summary(self) -> str:
        return (
            f"{self.kind} {self.asset} {self.timeframe} +{self.horizon_bars}: "
            f"{self.estimate.summary()} -> {self.verdict}"
        )
