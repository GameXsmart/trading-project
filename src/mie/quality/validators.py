"""Validation and anomaly detection for incoming market data.

Requirement §20: the system must never blindly trust incoming data. Everything a
provider returns passes through :class:`CandleValidator` before it is persisted.

The design has one important property — **it separates rejection from flagging**:

* Structurally impossible data (``high < low``, a timestamp off the grid, a negative
  volume) is *rejected*. Persisting it would corrupt every downstream calculation.
* Suspicious-but-possible data (a large move, a gap, a flat run) is *flagged*. Crypto
  really does move 20% in an hour and exchanges really do halt. Discarding those bars
  would fabricate a calmer market than the one that exists, which is a worse error
  than recording them with a warning attached.

Flags accumulate into the rolling quality score, which reduces published confidence.
That is the whole mechanism: degrade, don't pretend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from statistics import median

from mie.config.settings import QualityConfig
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc, grid, utcnow
from mie.core.types import Candle, QualityEvent, QualityEventType, QualitySeverity

log = get_logger(__name__)

__all__ = ["CandleValidator", "ValidationOutcome"]


@dataclass(slots=True)
class ValidationOutcome:
    """What survived validation, what did not, and why."""

    candles: list[Candle] = field(default_factory=list)
    rejected: list[tuple[Candle, str]] = field(default_factory=list)
    events: list[QualityEvent] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.candles)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def has_errors(self) -> bool:
        return any(e.severity is QualitySeverity.ERROR for e in self.events)

    def events_of(self, event_type: QualityEventType) -> list[QualityEvent]:
        return [e for e in self.events if e.event_type is event_type]


class CandleValidator:
    """Runs the check suite over one batch of candles from one source."""

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def validate(
        self,
        candles: list[Candle],
        asset: str,
        timeframe: Timeframe,
        source: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        context: list[Candle] | None = None,
        now: datetime | None = None,
    ) -> ValidationOutcome:
        """Validate a batch.

        ``window_start``/``window_end`` enable gap detection: without a stated
        expectation there is no way to distinguish "the provider returned less" from
        "less exists". ``context`` supplies previously-stored candles so that outlier
        statistics have history to work with on small incremental batches.
        """
        now = now or utcnow()
        outcome = ValidationOutcome()
        scope = {"asset": asset.upper(), "timeframe": timeframe, "source": source}

        if not candles:
            if window_start is not None and window_end is not None:
                outcome.events.append(
                    QualityEvent(
                        event_type=QualityEventType.EMPTY_RESPONSE,
                        severity=QualitySeverity.INFO,
                        window_start=window_start,
                        window_end=window_end,
                        message="no candles returned for the requested window",
                        **scope,
                    )
                )
            return outcome

        # Order matters: the ordering check must run before deduplication, because
        # deduplication sorts as a side effect and would otherwise hide the very
        # condition the ordering check exists to report.
        surviving = self._check_shape(candles, outcome, scope)
        surviving = self._check_alignment(surviving, timeframe, outcome, scope)
        surviving = self._check_ordering(surviving, outcome, scope)
        surviving = self._check_duplicates(surviving, outcome, scope)

        if surviving:
            history = (context or []) + surviving
            self._check_moves(surviving, history, timeframe, outcome, scope)
            self._check_flatline(surviving, outcome, scope)
            # Staleness is a question about *liveness*, so it only applies to a batch
            # that claims to be current. A backfill page covering last March is
            # supposed to be old; flagging it would bury the real signal under one
            # error per page and drag the trust score down for healthy history.
            if window_end is None or (
                (now - ensure_utc(window_end)).total_seconds() <= timeframe.seconds * 2
            ):
                self._check_staleness(surviving, timeframe, now, outcome, scope)

        if window_start is not None and window_end is not None:
            self._check_gaps(surviving, timeframe, window_start, window_end, now, outcome, scope)

        outcome.candles = surviving
        if outcome.events:
            log.debug(
                "validation_complete",
                asset=asset,
                timeframe=str(timeframe),
                source=source,
                accepted=len(surviving),
                rejected=len(outcome.rejected),
                events=len(outcome.events),
            )
        return outcome

    # ------------------------------------------------------------------ rejecting

    def _check_shape(
        self, candles: list[Candle], outcome: ValidationOutcome, scope: dict
    ) -> list[Candle]:
        """Reject bars that cannot describe a real market."""
        kept: list[Candle] = []
        for candle in candles:
            problem = _shape_problem(candle)
            if problem is None:
                kept.append(candle)
                continue
            outcome.rejected.append((candle, problem))
            outcome.events.append(
                QualityEvent(
                    event_type=QualityEventType.SHAPE_INVALID,
                    severity=QualitySeverity.ERROR,
                    window_start=candle.open_time,
                    window_end=candle.close_time,
                    message=problem,
                    details={
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    },
                    **scope,
                )
            )
        return kept

    def _check_alignment(
        self, candles: list[Candle], timeframe: Timeframe, outcome: ValidationOutcome, scope: dict
    ) -> list[Candle]:
        """Reject timestamps that are not on the timeframe grid.

        A misaligned bar silently breaks every join and rollup downstream, and there
        is no safe way to guess which bucket it belonged to.
        """
        kept: list[Candle] = []
        for candle in candles:
            if timeframe.is_aligned(candle.open_time):
                kept.append(candle)
                continue
            reason = f"open_time {candle.open_time.isoformat()} is not on the {timeframe} grid"
            outcome.rejected.append((candle, reason))
            outcome.events.append(
                QualityEvent(
                    event_type=QualityEventType.GRID_MISALIGNED,
                    severity=QualitySeverity.ERROR,
                    window_start=candle.open_time,
                    message=reason,
                    details={"expected": timeframe.floor(candle.open_time).isoformat()},
                    **scope,
                )
            )
        return kept

    # ------------------------------------------------------------------ repairing

    def _check_duplicates(
        self, candles: list[Candle], outcome: ValidationOutcome, scope: dict
    ) -> list[Candle]:
        """Collapse repeated open times, keeping the last occurrence.

        Providers repeat bars during pagination overlaps. Last-wins is right because a
        later copy in the same response is the more recently computed one, and Python's
        stable sort means the preceding ordering pass preserves arrival order among
        bars sharing a timestamp.
        """
        seen: dict[datetime, Candle] = {}
        duplicates: list[datetime] = []
        for candle in candles:
            if candle.open_time in seen:
                duplicates.append(candle.open_time)
            seen[candle.open_time] = candle
        if duplicates:
            outcome.events.append(
                QualityEvent(
                    event_type=QualityEventType.DUPLICATE,
                    severity=QualitySeverity.WARNING,
                    window_start=min(duplicates),
                    window_end=max(duplicates),
                    message=f"{len(duplicates)} duplicate candle(s) collapsed",
                    details={"timestamps": [d.isoformat() for d in duplicates[:20]]},
                    **scope,
                )
            )
        return [seen[key] for key in sorted(seen)]

    def _check_ordering(
        self, candles: list[Candle], outcome: ValidationOutcome, scope: dict
    ) -> list[Candle]:
        """Sort out-of-order batches, and say so."""
        times = [c.open_time for c in candles]
        if times == sorted(times):
            return candles
        outcome.events.append(
            QualityEvent(
                event_type=QualityEventType.OUT_OF_ORDER,
                severity=QualitySeverity.WARNING,
                window_start=min(times),
                window_end=max(times),
                message="candles arrived out of chronological order and were sorted",
                **scope,
            )
        )
        return sorted(candles, key=lambda c: c.open_time)

    # ------------------------------------------------------------------- flagging

    def _check_gaps(
        self,
        candles: list[Candle],
        timeframe: Timeframe,
        window_start: datetime,
        window_end: datetime,
        now: datetime,
        outcome: ValidationOutcome,
        scope: dict,
    ) -> None:
        """Compare against the expected grid and report contiguous missing runs.

        The window is truncated at the last *closed* bar: a bar that has not finished
        forming is not missing, and counting it as a gap would raise an alarm on every
        single poll.
        """
        effective_end = min(ensure_utc(window_end), timeframe.floor(now))
        if effective_end <= ensure_utc(window_start):
            return

        present = {c.open_time for c in candles}
        expected = list(grid(window_start, effective_end, timeframe))
        if not expected:
            return
        missing = [ts for ts in expected if ts not in present]
        if not missing:
            return

        ratio = len(missing) / len(expected)
        severity = (
            QualitySeverity.ERROR
            if ratio >= self.config.gap_error_ratio
            else QualitySeverity.WARNING
            if ratio >= self.config.gap_warning_ratio
            else QualitySeverity.INFO
        )
        runs = _contiguous_runs(missing, timeframe)
        outcome.events.append(
            QualityEvent(
                event_type=QualityEventType.GAP,
                severity=severity,
                window_start=missing[0],
                window_end=timeframe.close_time(missing[-1]),
                message=(
                    f"{len(missing)} of {len(expected)} expected candles missing "
                    f"({ratio:.1%}) across {len(runs)} gap(s)"
                ),
                details={
                    "missing": len(missing),
                    "expected": len(expected),
                    "ratio": round(ratio, 4),
                    "runs": [
                        {"from": a.isoformat(), "to": b.isoformat(), "candles": n}
                        for a, b, n in runs[:20]
                    ],
                },
                **scope,
            )
        )

    def _check_moves(
        self,
        candles: list[Candle],
        history: list[Candle],
        timeframe: Timeframe,
        outcome: ValidationOutcome,
        scope: dict,
    ) -> None:
        """Flag implausible single-bar returns, two ways.

        A hard percentage cap catches unambiguous corruption (a decimal-point error,
        a wrong-symbol response). A robust z-score against recent volatility catches
        bars that are merely extreme *for this market right now* — the MAD is used
        rather than the standard deviation precisely because a single 50-sigma print
        would inflate an SD-based threshold enough to hide itself.
        """
        cap = self.config.max_move_pct.get(timeframe, 100.0)
        returns = _returns(history)
        # Only bars from this batch are reportable; history is context for the
        # statistics. Membership is tested on open_time so the check stays O(n).
        batch_times = {c.open_time for c in candles}

        scale = 0.0
        if len(returns) >= self.config.outlier_min_samples:
            centre = median(returns)
            deviations = [abs(r - centre) for r in returns]
            # 1.4826 rescales the MAD to be a consistent estimator of sigma for
            # normally distributed data.
            scale = median(deviations) * 1.4826
        else:
            centre = 0.0

        for previous, candle in pairwise(history):
            if candle.open_time not in batch_times or previous.close <= 0:
                continue
            change = (candle.close - previous.close) / previous.close * 100.0
            magnitude = abs(change)

            if magnitude > cap:
                outcome.events.append(
                    QualityEvent(
                        event_type=QualityEventType.IMPOSSIBLE_MOVE,
                        severity=QualitySeverity.ERROR,
                        window_start=candle.open_time,
                        window_end=candle.close_time,
                        message=(
                            f"{change:+.2f}% move on one {timeframe} bar exceeds the "
                            f"{cap:.0f}% plausibility cap"
                        ),
                        details={
                            "change_pct": round(change, 4),
                            "cap_pct": cap,
                            "previous_close": previous.close,
                            "close": candle.close,
                        },
                        **scope,
                    )
                )
                continue

            if scale > 0:
                z = abs(change - centre) / scale
                if z > self.config.outlier_mad_threshold:
                    outcome.events.append(
                        QualityEvent(
                            event_type=QualityEventType.OUTLIER,
                            severity=QualitySeverity.WARNING,
                            window_start=candle.open_time,
                            window_end=candle.close_time,
                            message=(
                                f"{change:+.2f}% move is {z:.1f} robust sigma from recent "
                                f"behaviour — extreme but not impossible"
                            ),
                            details={
                                "change_pct": round(change, 4),
                                "robust_z": round(z, 2),
                                "mad_sigma_pct": round(scale, 4),
                                "samples": len(returns),
                            },
                            **scope,
                        )
                    )

    def _check_flatline(
        self, candles: list[Candle], outcome: ValidationOutcome, scope: dict
    ) -> None:
        """Detect runs of zero volume or zero range — the signature of a dead feed.

        Genuinely illiquid assets do print empty bars, so this is a warning, not a
        rejection; the run-length threshold keeps ordinary quiet periods quiet.
        """
        threshold = self.config.flatline_run_length
        run: list[Candle] = []
        flagged: list[tuple[Candle, Candle, int]] = []

        for candle in candles:
            dead = candle.volume <= 0 or candle.high == candle.low
            if dead:
                run.append(candle)
                continue
            if len(run) >= threshold:
                flagged.append((run[0], run[-1], len(run)))
            run = []
        if len(run) >= threshold:
            flagged.append((run[0], run[-1], len(run)))

        for first, last, length in flagged:
            outcome.events.append(
                QualityEvent(
                    event_type=QualityEventType.FLATLINE,
                    severity=QualitySeverity.WARNING,
                    window_start=first.open_time,
                    window_end=last.close_time,
                    message=f"{length} consecutive candles with no volume or no range",
                    details={"run_length": length},
                    **scope,
                )
            )

    def _check_staleness(
        self,
        candles: list[Candle],
        timeframe: Timeframe,
        now: datetime,
        outcome: ValidationOutcome,
        scope: dict,
    ) -> None:
        """Flag a feed whose newest bar is too old to be live.

        Measured from the newest bar's *close*, so a just-opened bar is never called
        stale. The multiplier allows for normal provider publication lag.
        """
        newest = max(candles, key=lambda c: c.open_time)
        age = (now - newest.close_time).total_seconds()
        allowance = timeframe.seconds * self.config.staleness_multiplier
        if age <= allowance:
            return
        outcome.events.append(
            QualityEvent(
                event_type=QualityEventType.STALE_FEED,
                severity=(
                    QualitySeverity.ERROR
                    if age > allowance * 3
                    else QualitySeverity.WARNING
                ),
                window_start=newest.open_time,
                window_end=newest.close_time,
                message=(
                    f"newest candle closed {age / 60:.1f} minutes ago; expected within "
                    f"{allowance / 60:.1f} minutes"
                ),
                details={"age_s": round(age), "allowance_s": round(allowance)},
                **scope,
            )
        )


# ---------------------------------------------------------------------- helpers


def _shape_problem(candle: Candle) -> str | None:
    """Return a description of why this bar is impossible, or None if it is fine."""
    values = {
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
    }
    for name, value in values.items():
        if value != value:  # NaN is the only value not equal to itself
            return f"{name} is NaN"
        if value in (float("inf"), float("-inf")):
            return f"{name} is infinite"
        if value <= 0:
            return f"{name} is non-positive ({value})"
    if candle.volume < 0 or candle.volume != candle.volume:
        return f"volume is negative or NaN ({candle.volume})"
    if candle.high < candle.low:
        return f"high {candle.high} is below low {candle.low}"
    if candle.high < max(candle.open, candle.close):
        return f"high {candle.high} is below open/close {max(candle.open, candle.close)}"
    if candle.low > min(candle.open, candle.close):
        return f"low {candle.low} is above open/close {min(candle.open, candle.close)}"
    return None


def _returns(candles: list[Candle]) -> list[float]:
    """Close-to-close percentage returns over an ordered series."""
    return [
        (right.close - left.close) / left.close * 100.0
        for left, right in pairwise(candles)
        if left.close > 0
    ]


def _contiguous_runs(
    missing: list[datetime], timeframe: Timeframe
) -> list[tuple[datetime, datetime, int]]:
    """Collapse missing timestamps into ``(start, end, count)`` runs.

    One event describing a six-hour outage is actionable; 360 events describing each
    missing minute is noise that buries everything else in the log.
    """
    if not missing:
        return []
    runs: list[tuple[datetime, datetime, int]] = []
    start = previous = missing[0]
    count = 1
    for timestamp in missing[1:]:
        if (timestamp - previous).total_seconds() == timeframe.seconds:
            previous = timestamp
            count += 1
            continue
        runs.append((start, previous, count))
        start = previous = timestamp
        count = 1
    runs.append((start, previous, count))
    return runs
