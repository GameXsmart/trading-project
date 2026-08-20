"""Timeframe algebra.

Every timestamp in the system lives on a timeframe grid. Getting this wrong is the
single most common source of silent data corruption in market-data pipelines
(off-by-one candles, misaligned joins, look-ahead), so the grid logic lives in one
place, is pure, and is unit-tested.

All datetimes are timezone-aware UTC. Naive datetimes are rejected, not coerced.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from enum import StrEnum

__all__ = ["UTC", "Timeframe", "ensure_utc", "expected_count", "floor_to", "grid", "utcnow"]


class Timeframe(StrEnum):
    """Supported analysis timeframes, ordered from fastest to slowest."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        return _SECONDS[self]

    @property
    def delta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    @property
    def rank(self) -> int:
        """Position in the hierarchy; higher means slower/more macro."""
        return _ORDER.index(self)

    def floor(self, moment: datetime) -> datetime:
        """Start of the candle that contains ``moment``."""
        return floor_to(moment, self)

    def ceil(self, moment: datetime) -> datetime:
        """Start of the next candle boundary at or after ``moment``."""
        floored = self.floor(moment)
        return floored if floored == moment else floored + self.delta

    def is_aligned(self, moment: datetime) -> bool:
        return self.floor(moment) == ensure_utc(moment)

    def close_time(self, open_time: datetime) -> datetime:
        """Exclusive end of the candle beginning at ``open_time``."""
        return ensure_utc(open_time) + self.delta

    @classmethod
    def parse(cls, value: str | Timeframe) -> Timeframe:
        if isinstance(value, Timeframe):
            return value
        try:
            return cls(value.strip().lower())
        except ValueError as exc:  # pragma: no cover - message clarity only
            raise ValueError(
                f"unknown timeframe {value!r}; supported: {', '.join(t.value for t in cls)}"
            ) from exc


_SECONDS: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3_600,
    Timeframe.H4: 14_400,
    Timeframe.H12: 43_200,
    Timeframe.D1: 86_400,
    Timeframe.W1: 604_800,
}

_ORDER: list[Timeframe] = sorted(_SECONDS, key=lambda tf: _SECONDS[tf])

# The Unix epoch is a Thursday. Weekly candles conventionally open on Monday, so the
# weekly grid is anchored to the first Monday of the epoch rather than to t=0.
_WEEK_ANCHOR = datetime(1970, 1, 5, tzinfo=UTC)


def utcnow() -> datetime:
    """Current time, timezone-aware UTC. The only clock the system reads."""
    return datetime.now(UTC)


def ensure_utc(moment: datetime) -> datetime:
    """Reject naive datetimes; normalise aware ones to UTC.

    Coercing naive datetimes to UTC silently is how timezone bugs become data bugs,
    so a naive input is an error rather than an assumption.
    """
    if moment.tzinfo is None:
        raise ValueError(f"naive datetime {moment!r}: all timestamps must be tz-aware UTC")
    return moment.astimezone(UTC)


def floor_to(moment: datetime, timeframe: Timeframe) -> datetime:
    """Round ``moment`` down to the start of its candle on ``timeframe``."""
    moment = ensure_utc(moment)
    if timeframe is Timeframe.W1:
        elapsed = (moment - _WEEK_ANCHOR).total_seconds()
        buckets = int(elapsed // timeframe.seconds)
        return _WEEK_ANCHOR + timedelta(seconds=buckets * timeframe.seconds)
    epoch_seconds = moment.timestamp()
    return datetime.fromtimestamp(
        (int(epoch_seconds) // timeframe.seconds) * timeframe.seconds, tz=UTC
    )


def grid(start: datetime, end: datetime, timeframe: Timeframe) -> Iterator[datetime]:
    """Yield every expected candle open time in ``[start, end)``.

    Both bounds are floored to the grid first, so callers may pass arbitrary times.
    This is the reference the gap detector compares observed data against.
    """
    cursor = floor_to(start, timeframe)
    limit = ensure_utc(end)
    step = timeframe.delta
    while cursor < limit:
        yield cursor
        cursor += step


def expected_count(start: datetime, end: datetime, timeframe: Timeframe) -> int:
    """Number of candles a complete series would contain over ``[start, end)``."""
    first = floor_to(start, timeframe)
    last = ensure_utc(end)
    if last <= first:
        return 0
    return int((last - first).total_seconds() // timeframe.seconds)
