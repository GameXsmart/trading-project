"""Timeframe grid algebra.

The grid underpins gap detection, rollups, and every join in the system, so it is
tested harder than its size suggests. A bug here is invisible until it corrupts
months of history.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mie.core.timeframes import (
    UTC,
    Timeframe,
    ensure_utc,
    expected_count,
    floor_to,
    grid,
)


class TestParsing:
    def test_parses_strings_and_passes_through_instances(self) -> None:
        assert Timeframe.parse("1h") is Timeframe.H1
        assert Timeframe.parse(" 4H ") is Timeframe.H4
        assert Timeframe.parse(Timeframe.D1) is Timeframe.D1

    def test_rejects_unknown_timeframe_with_a_useful_message(self) -> None:
        with pytest.raises(ValueError, match="unknown timeframe"):
            Timeframe.parse("3h")

    def test_seconds_and_ordering_are_consistent(self) -> None:
        frames = list(Timeframe)
        seconds = [f.seconds for f in frames]
        assert seconds == sorted(seconds), "enum declaration order must be fastest-first"
        assert Timeframe.H1.rank < Timeframe.D1.rank


class TestFlooring:
    @pytest.mark.parametrize(
        ("timeframe", "moment", "expected"),
        [
            (Timeframe.M1, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 13, 47, tzinfo=UTC)),
            (Timeframe.M5, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 13, 45, tzinfo=UTC)),
            (Timeframe.M15, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 13, 45, tzinfo=UTC)),
            (Timeframe.M30, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 13, 30, tzinfo=UTC)),
            (Timeframe.H1, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 13, 0, tzinfo=UTC)),
            (Timeframe.H4, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 12, 0, tzinfo=UTC)),
            (Timeframe.H12, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 12, 0, tzinfo=UTC)),
            (Timeframe.D1, datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC), datetime(2025, 6, 2, 0, 0, tzinfo=UTC)),
        ],
    )
    def test_floors_to_the_containing_bucket(
        self, timeframe: Timeframe, moment: datetime, expected: datetime
    ) -> None:
        assert floor_to(moment, timeframe) == expected

    def test_weekly_candles_open_on_monday(self) -> None:
        """The epoch is a Thursday, so a naive modulus would put weeks on Thursdays."""
        wednesday = datetime(2025, 6, 4, 9, 30, tzinfo=UTC)
        floored = floor_to(wednesday, Timeframe.W1)
        assert floored.weekday() == 0, "weekly bars must open on Monday"
        assert floored == datetime(2025, 6, 2, tzinfo=UTC)

    def test_flooring_is_idempotent(self) -> None:
        moment = datetime(2025, 6, 2, 13, 47, 33, tzinfo=UTC)
        for timeframe in Timeframe:
            once = floor_to(moment, timeframe)
            assert floor_to(once, timeframe) == once

    def test_ceil_leaves_aligned_moments_untouched(self) -> None:
        aligned = datetime(2025, 6, 2, 13, 0, tzinfo=UTC)
        assert Timeframe.H1.ceil(aligned) == aligned
        assert Timeframe.H1.ceil(aligned + timedelta(seconds=1)) == aligned + timedelta(hours=1)

    def test_alignment_check_matches_flooring(self) -> None:
        assert Timeframe.H4.is_aligned(datetime(2025, 6, 2, 12, tzinfo=UTC))
        assert not Timeframe.H4.is_aligned(datetime(2025, 6, 2, 13, tzinfo=UTC))


class TestTimezoneDiscipline:
    def test_naive_datetimes_are_rejected_not_assumed(self) -> None:
        """Silently treating a naive datetime as UTC is how timezone bugs get in."""
        with pytest.raises(ValueError, match="naive datetime"):
            ensure_utc(datetime(2025, 6, 2, 13, 0))

    def test_non_utc_offsets_are_normalised(self) -> None:
        tokyo = datetime(2025, 6, 2, 22, 0, tzinfo=timezone_offset(9))
        assert ensure_utc(tokyo) == datetime(2025, 6, 2, 13, 0, tzinfo=UTC)

    def test_flooring_respects_the_original_offset(self) -> None:
        """13:30 in UTC+9 is 04:30 UTC, so the daily bar is the previous day."""
        tokyo = datetime(2025, 6, 2, 13, 30, tzinfo=timezone_offset(9))
        assert floor_to(tokyo, Timeframe.D1) == datetime(2025, 6, 2, 0, 0, tzinfo=UTC)


class TestGrid:
    def test_grid_is_half_open(self) -> None:
        start = datetime(2025, 6, 2, tzinfo=UTC)
        end = start + timedelta(hours=3)
        points = list(grid(start, end, Timeframe.H1))
        assert points == [start, start + timedelta(hours=1), start + timedelta(hours=2)]
        assert end not in points

    def test_grid_floors_unaligned_bounds(self) -> None:
        start = datetime(2025, 6, 2, 0, 37, tzinfo=UTC)
        points = list(grid(start, start + timedelta(hours=2), Timeframe.H1))
        assert points[0] == datetime(2025, 6, 2, 0, 0, tzinfo=UTC)

    def test_empty_and_inverted_ranges_yield_nothing(self) -> None:
        moment = datetime(2025, 6, 2, tzinfo=UTC)
        assert list(grid(moment, moment, Timeframe.H1)) == []
        assert list(grid(moment, moment - timedelta(days=1), Timeframe.H1)) == []

    def test_expected_count_matches_the_grid_length(self) -> None:
        start = datetime(2025, 6, 2, tzinfo=UTC)
        for timeframe in (Timeframe.M1, Timeframe.M15, Timeframe.H4, Timeframe.D1):
            end = start + timedelta(days=3)
            assert expected_count(start, end, timeframe) == len(list(grid(start, end, timeframe)))

    def test_close_time_is_exclusive_end_of_bar(self) -> None:
        open_time = datetime(2025, 6, 2, 13, tzinfo=UTC)
        assert Timeframe.H1.close_time(open_time) == datetime(2025, 6, 2, 14, tzinfo=UTC)


def timezone_offset(hours: int):
    from datetime import timedelta, timezone

    return timezone(timedelta(hours=hours))
