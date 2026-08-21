"""Pattern evaluation.

Scans history, records every detection, measures what actually happened afterwards,
and decides — statistically — whether each pattern carries information.

This module is where the Phase 4 gate is enforced, so it is worth being explicit about
the ways this measurement could lie, and what stops each:

* **Comparing to a coin flip.** Crypto drifts. Every pattern is scored against the
  *unconditional* outcome rate over the identical sample, so "56% of the time price
  rose" is only an edge if price rose less often than that in general.
* **Look-ahead.** Detectors see `candles[:i + 1]`; outcomes read `candles[i + horizon]`.
  The two never overlap, and the slice boundary makes that structural.
* **Survivorship of thresholds.** Detector thresholds are conventional and fixed
  before measurement. Tuning them until a pattern looks good would guarantee it looks
  good, and guarantee nothing else.
* **Multiple comparisons.** A sweep of eleven detectors across assets, timeframes and
  horizons is hundreds of hypotheses. Benjamini-Hochberg is applied across the whole
  family at once, not per-test.
* **Overlapping samples.** Consecutive detections of the same pattern share most of
  their forward window, so their outcomes are not independent and naive counting
  overstates the sample size. Detections are thinned to be at least one horizon apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.core.types import Candle
from mie.patterns.detectors import detect_all
from mie.patterns.statistics import benjamini_hochberg, compare_to_baseline
from mie.patterns.types import (
    Detection,
    Outcome,
    PatternDirection,
    PatternKind,
    PatternStats,
)

log = get_logger(__name__)

__all__ = ["DEFAULT_HORIZONS", "PatternEvaluator", "ScanResult"]

#: Forward horizons in bars. Several, because a pattern can be informative over one
#: horizon and useless over another, and reporting only the flattering one is the
#: oldest trick in backtesting.
DEFAULT_HORIZONS: tuple[int, ...] = (3, 12, 48)

#: Minimum detections before a rate is worth reporting at all.
_MIN_OCCURRENCES = 30

#: A "movement" outcome for direction-neutral patterns: did the bar range expand
#: beyond this multiple of the pre-detection average?
_MOVEMENT_MULTIPLE = 1.5


@dataclass(slots=True)
class ScanResult:
    """Everything one scan produced."""

    asset: str
    timeframe: Timeframe
    bars_scanned: int
    detections: list[Detection] = field(default_factory=list)
    stats: list[PatternStats] = field(default_factory=list)

    @property
    def informative(self) -> list[PatternStats]:
        return [s for s in self.stats if s.is_informative]

    @property
    def rejected(self) -> list[PatternStats]:
        return [s for s in self.stats if not s.is_informative]

    def by_kind(self, kind: PatternKind) -> list[Detection]:
        return [d for d in self.detections if d.kind is kind]


class PatternEvaluator:
    """Detects patterns across history and measures whether they mean anything."""

    def __init__(
        self,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        min_occurrences: int = _MIN_OCCURRENCES,
        false_discovery_rate: float = 0.05,
    ) -> None:
        self.horizons = tuple(horizons)
        self.min_occurrences = min_occurrences
        self.false_discovery_rate = false_discovery_rate

    # --------------------------------------------------------------- detection

    def scan(
        self,
        candles: Sequence[Candle],
        asset: str,
        timeframe: Timeframe,
        warmup: int = 80,
    ) -> list[Detection]:
        """Run every detector at every bar, oldest first.

        Each detector receives a slice ending at the bar under test, so a detector
        physically cannot see a future bar even if it tried.
        """
        final = [c for c in candles if c.is_final]
        detections: list[Detection] = []
        for index in range(warmup, len(final)):
            detections.extend(detect_all(final[: index + 1], asset, timeframe))
        return detections

    # ---------------------------------------------------------------- outcomes

    def outcome_for(
        self, candles: Sequence[Candle], index: int, horizon: int, direction: PatternDirection
    ) -> Outcome | None:
        """What happened over the ``horizon`` bars after ``index``."""
        if index + horizon >= len(candles):
            return None
        entry = candles[index]
        exit_bar = candles[index + horizon]
        forward = candles[index + 1 : index + horizon + 1]
        if entry.close <= 0 or not forward:
            return None

        return_pct = (exit_bar.close - entry.close) / entry.close * 100.0
        best = max(c.high for c in forward)
        worst = min(c.low for c in forward)
        favourable = (best - entry.close) / entry.close * 100.0
        adverse = (worst - entry.close) / entry.close * 100.0

        if direction is PatternDirection.BULLISH:
            resolved: bool | None = return_pct > 0
        elif direction is PatternDirection.BEARISH:
            resolved = return_pct < 0
        else:
            # Direction-neutral patterns claim movement, not direction, so they are
            # scored on whether the move exceeded the market's own recent range.
            reference = _average_range(candles, index)
            resolved = (
                None
                if reference <= 0
                else max(favourable, abs(adverse)) > reference / entry.close * 100.0 * _MOVEMENT_MULTIPLE
            )

        return Outcome(
            detection_at=entry.open_time,
            horizon_bars=horizon,
            entry_close=entry.close,
            exit_close=exit_bar.close,
            return_pct=return_pct,
            max_favourable_pct=favourable,
            max_adverse_pct=adverse,
            resolved_as_expected=resolved,
        )

    # -------------------------------------------------------------- evaluation

    def evaluate(
        self, candles: Sequence[Candle], asset: str, timeframe: Timeframe, warmup: int = 80
    ) -> ScanResult:
        """Full pipeline: detect, measure outcomes, test against baseline."""
        final = [c for c in candles if c.is_final]
        result = ScanResult(asset=asset.upper(), timeframe=timeframe, bars_scanned=len(final))
        if len(final) < warmup + max(self.horizons) + self.min_occurrences:
            log.debug(
                "pattern_scan_too_short",
                asset=asset,
                timeframe=str(timeframe),
                bars=len(final),
            )
            return result

        result.detections = self.scan(final, asset, timeframe, warmup)
        index_of = {c.open_time: i for i, c in enumerate(final)}

        raw_stats: list[PatternStats] = []
        for kind in PatternKind:
            matching = [d for d in result.detections if d.kind is kind]
            if not matching:
                continue
            for horizon in self.horizons:
                stats = self._stats_for(final, index_of, matching, kind, horizon, asset, timeframe)
                if stats is not None:
                    raw_stats.append(stats)

        result.stats = self._apply_fdr(raw_stats)
        return result

    def _stats_for(
        self,
        candles: Sequence[Candle],
        index_of: dict[datetime, int],
        detections: Sequence[Detection],
        kind: PatternKind,
        horizon: int,
        asset: str,
        timeframe: Timeframe,
    ) -> PatternStats | None:
        direction = detections[0].direction

        # Thin overlapping detections: consecutive hits share nearly all of their
        # forward window, so counting each as an independent trial inflates n and
        # shrinks the confidence interval to a width the evidence does not support.
        thinned: list[int] = []
        for detection in detections:
            index = index_of.get(detection.at)
            if index is None:
                continue
            if thinned and index - thinned[-1] < horizon:
                continue
            thinned.append(index)

        outcomes = [
            outcome
            for index in thinned
            if (outcome := self.outcome_for(candles, index, horizon, direction)) is not None
            and outcome.resolved_as_expected is not None
        ]
        if len(outcomes) < 5:
            return None  # nothing meaningful to say, not even "insufficient"

        successes = sum(1 for o in outcomes if o.resolved_as_expected)
        baseline_successes, baseline_trials = self._baseline(
            candles, horizon, direction
        )
        estimate = compare_to_baseline(
            successes, len(outcomes), baseline_successes, baseline_trials
        )

        returns = [o.return_pct for o in outcomes]
        return PatternStats(
            kind=kind,
            asset=asset.upper(),
            timeframe=timeframe,
            horizon_bars=horizon,
            direction=direction,
            occurrences=len(outcomes),
            estimate=estimate,
            mean_return_pct=sum(returns) / len(returns),
            median_return_pct=median(returns),
            mean_favourable_pct=sum(o.max_favourable_pct for o in outcomes) / len(outcomes),
            mean_adverse_pct=sum(o.max_adverse_pct for o in outcomes) / len(outcomes),
            sample_start=candles[0].open_time,
            sample_end=candles[-1].open_time,
        )

    def _baseline(
        self, candles: Sequence[Candle], horizon: int, direction: PatternDirection
    ) -> tuple[int, int]:
        """Unconditional outcome rate over the same sample and horizon.

        The comparison that makes an "edge" meaningful. Sampled every ``horizon`` bars
        so the baseline windows are non-overlapping too — comparing a thinned pattern
        sample against an overlapping baseline would bias the test.
        """
        successes = trials = 0
        for index in range(0, len(candles) - horizon, horizon):
            outcome = self.outcome_for(candles, index, horizon, direction)
            if outcome is None or outcome.resolved_as_expected is None:
                continue
            trials += 1
            successes += int(outcome.resolved_as_expected)
        return successes, trials

    def _apply_fdr(self, stats: Sequence[PatternStats]) -> list[PatternStats]:
        """Apply Benjamini-Hochberg across the whole family of tests at once.

        Per-test correction would be no correction at all: the false discoveries come
        from the size of the sweep, so the sweep is what has to be corrected.
        """
        if not stats:
            return []
        rejected = benjamini_hochberg(
            [s.estimate.p_value for s in stats], self.false_discovery_rate
        )
        corrected: list[PatternStats] = []
        for stat, is_significant in zip(stats, rejected, strict=True):
            estimate = stat.estimate
            corrected.append(
                stat.model_copy(
                    update={
                        "estimate": type(estimate)(
                            successes=estimate.successes,
                            trials=estimate.trials,
                            rate=estimate.rate,
                            low=estimate.low,
                            high=estimate.high,
                            baseline=estimate.baseline,
                            edge=estimate.edge,
                            p_value=estimate.p_value,
                            significant=is_significant,
                        )
                    }
                )
            )
        return corrected


def _average_range(candles: Sequence[Candle], index: int, period: int = 14) -> float:
    """Average true range over the bars *preceding* ``index``."""
    start = max(1, index - period + 1)
    if start > index:
        return 0.0
    ranges = []
    for i in range(start, index + 1):
        previous = candles[i - 1].close
        bar = candles[i]
        ranges.append(
            max(bar.high - bar.low, abs(bar.high - previous), abs(bar.low - previous))
        )
    return sum(ranges) / len(ranges) if ranges else 0.0
