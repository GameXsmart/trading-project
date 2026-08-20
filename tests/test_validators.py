"""Data-quality validation.

Each test injects one specific defect into an otherwise clean series and asserts both
halves of the contract: the right event is raised, *and* the right thing happens to
the data — rejected when it is impossible, kept when it is merely unusual.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW, make_candle, series

from mie.config.settings import QualityConfig
from mie.core.timeframes import Timeframe
from mie.core.types import QualityEventType, QualitySeverity
from mie.quality.validators import CandleValidator


@pytest.fixture
def validator() -> CandleValidator:
    return CandleValidator(QualityConfig())


def _validate(validator: CandleValidator, candles, **kwargs):
    kwargs.setdefault("asset", "BTC")
    kwargs.setdefault("timeframe", Timeframe.H1)
    kwargs.setdefault("source", "fake")
    kwargs.setdefault("now", FIXED_NOW)
    return validator.validate(candles, **kwargs)


class TestCleanData:
    def test_clean_series_produces_no_events(self, validator: CandleValidator) -> None:
        outcome = _validate(validator, series(60))
        assert outcome.accepted_count == 60
        assert outcome.rejected_count == 0
        assert outcome.events == []

    def test_empty_batch_without_a_window_is_silent(self, validator: CandleValidator) -> None:
        """No expectation was stated, so there is nothing to report as missing."""
        assert _validate(validator, []).events == []

    def test_empty_batch_within_a_stated_window_is_reported(
        self, validator: CandleValidator
    ) -> None:
        outcome = _validate(
            validator,
            [],
            window_start=FIXED_NOW - timedelta(hours=5),
            window_end=FIXED_NOW,
        )
        assert outcome.events_of(QualityEventType.EMPTY_RESPONSE)


class TestRejection:
    """Structurally impossible bars must never reach storage."""

    def test_high_below_low_is_rejected(self, validator: CandleValidator) -> None:
        candles = series(10)
        candles[5] = make_candle(candles[5].open_time, close=100, high=90, low=110)
        outcome = _validate(validator, candles)

        assert outcome.accepted_count == 9
        assert outcome.rejected_count == 1
        event = outcome.events_of(QualityEventType.SHAPE_INVALID)[0]
        assert event.severity is QualitySeverity.ERROR
        assert "below low" in event.message

    def test_high_below_close_is_rejected(self, validator: CandleValidator) -> None:
        candles = series(5)
        candles[2] = make_candle(candles[2].open_time, close=120, high=100, low=90)
        outcome = _validate(validator, candles)
        assert outcome.rejected_count == 1
        assert outcome.events_of(QualityEventType.SHAPE_INVALID)

    def test_non_positive_price_is_rejected(self, validator: CandleValidator) -> None:
        candles = series(5)
        candles[1] = make_candle(candles[1].open_time, close=0.0, open_=0.0, high=0.0, low=0.0)
        outcome = _validate(validator, candles)
        assert outcome.rejected_count == 1
        assert "non-positive" in outcome.events_of(QualityEventType.SHAPE_INVALID)[0].message

    def test_nan_price_is_rejected(self, validator: CandleValidator) -> None:
        candles = series(5)
        candles[3] = make_candle(
            candles[3].open_time, close=float("nan"), open_=100, high=200, low=50
        )
        outcome = _validate(validator, candles)
        assert outcome.rejected_count == 1
        assert "NaN" in outcome.events_of(QualityEventType.SHAPE_INVALID)[0].message

    def test_negative_volume_is_rejected(self, validator: CandleValidator) -> None:
        candles = series(5)
        candles[0] = make_candle(candles[0].open_time, close=100, volume=-5.0)
        outcome = _validate(validator, candles)
        assert outcome.rejected_count == 1

    def test_misaligned_timestamp_is_rejected(self, validator: CandleValidator) -> None:
        """A bar off the grid cannot be assigned to a bucket, so it cannot be kept."""
        candles = series(5)
        candles[2] = make_candle(candles[2].open_time + timedelta(minutes=7), close=100)
        outcome = _validate(validator, candles)

        assert outcome.rejected_count == 1
        event = outcome.events_of(QualityEventType.GRID_MISALIGNED)[0]
        assert event.severity is QualitySeverity.ERROR
        assert "expected" in event.details


class TestRepair:
    def test_duplicates_are_collapsed_last_wins(self, validator: CandleValidator) -> None:
        candles = series(5)
        # A later copy of the same bar is the more recently computed one.
        revised = make_candle(candles[2].open_time, close=999.0, high=999.0, low=1.0)
        outcome = _validate(validator, [*candles, revised])

        assert outcome.accepted_count == 5
        assert outcome.events_of(QualityEventType.DUPLICATE)
        kept = next(c for c in outcome.candles if c.open_time == candles[2].open_time)
        assert kept.close == 999.0

    def test_out_of_order_batches_are_sorted_and_flagged(
        self, validator: CandleValidator
    ) -> None:
        candles = series(6)
        shuffled = [candles[3], candles[0], candles[5], candles[1], candles[4], candles[2]]
        outcome = _validate(validator, shuffled)

        times = [c.open_time for c in outcome.candles]
        assert times == sorted(times)
        assert outcome.events_of(QualityEventType.OUT_OF_ORDER)


class TestGaps:
    def test_missing_candles_are_detected_against_the_grid(
        self, validator: CandleValidator
    ) -> None:
        candles = series(24, start=FIXED_NOW - timedelta(hours=24))
        del candles[10:13]  # a three-hour hole in the middle

        outcome = _validate(
            validator,
            candles,
            window_start=FIXED_NOW - timedelta(hours=24),
            window_end=FIXED_NOW,
        )
        event = outcome.events_of(QualityEventType.GAP)[0]
        assert event.details["missing"] == 3
        assert event.details["runs"][0]["candles"] == 3

    def test_separate_gaps_are_reported_as_separate_runs(
        self, validator: CandleValidator
    ) -> None:
        """One event per outage, not one per missing bar — otherwise the log is noise."""
        candles = series(24, start=FIXED_NOW - timedelta(hours=24))
        del candles[15]
        del candles[5]

        outcome = _validate(
            validator,
            candles,
            window_start=FIXED_NOW - timedelta(hours=24),
            window_end=FIXED_NOW,
        )
        event = outcome.events_of(QualityEventType.GAP)[0]
        assert len(event.details["runs"]) == 2

    def test_the_forming_bar_is_not_counted_as_a_gap(
        self, validator: CandleValidator
    ) -> None:
        """Otherwise every single poll would raise a gap alarm for the current bar."""
        candles = series(5, start=FIXED_NOW - timedelta(hours=5))
        outcome = _validate(
            validator,
            candles,
            window_start=FIXED_NOW - timedelta(hours=5),
            window_end=FIXED_NOW + timedelta(hours=1),
            now=FIXED_NOW + timedelta(minutes=20),
        )
        assert not outcome.events_of(QualityEventType.GAP)

    def test_large_gap_ratio_escalates_to_error(self, validator: CandleValidator) -> None:
        candles = series(20, start=FIXED_NOW - timedelta(hours=20))
        del candles[2:12]  # half the window missing

        outcome = _validate(
            validator,
            candles,
            window_start=FIXED_NOW - timedelta(hours=20),
            window_end=FIXED_NOW,
        )
        assert outcome.events_of(QualityEventType.GAP)[0].severity is QualitySeverity.ERROR


class TestAnomalies:
    def test_impossible_move_is_flagged_but_kept(self, validator: CandleValidator) -> None:
        """A corrupt-looking bar is recorded, not discarded: we do not rewrite history."""
        candles = series(40, base=100.0, step=0.0)
        spiked = make_candle(candles[20].open_time, close=100_000.0, open_=100.0, low=100.0)
        candles[20] = spiked

        outcome = _validate(validator, candles)
        event = outcome.events_of(QualityEventType.IMPOSSIBLE_MOVE)[0]
        assert event.severity is QualitySeverity.ERROR
        assert outcome.accepted_count == 40, "flagged, not rejected"
        assert event.details["cap_pct"] == 35.0

    def test_outlier_uses_a_robust_scale(self, validator: CandleValidator) -> None:
        """MAD, not standard deviation: an SD wide enough to contain the spike would
        hide it."""
        candles = series(60, base=100.0, step=0.0)
        for index in range(1, 60):
            drift = 100.0 * (1 + 0.001 * (1 if index % 2 else -1))
            candles[index] = make_candle(candles[index].open_time, close=drift)
        candles[50] = make_candle(candles[50].open_time, close=115.0, open_=100.0, low=100.0)

        outcome = _validate(validator, candles)
        assert outcome.events_of(QualityEventType.OUTLIER)

    def test_ordinary_volatility_is_not_flagged(self, validator: CandleValidator) -> None:
        """Crypto moves. A validator that cries wolf on normal action is useless."""
        candles = series(60, base=100.0, step=0.0)
        for index in range(1, 60):
            level = 100.0 * (1 + 0.02 * (1 if index % 3 else -1))
            candles[index] = make_candle(candles[index].open_time, close=level)

        outcome = _validate(validator, candles)
        assert not outcome.events_of(QualityEventType.IMPOSSIBLE_MOVE)

    def test_flatline_run_is_flagged(self, validator: CandleValidator) -> None:
        candles = series(30)
        for index in range(10, 22):
            candles[index] = make_candle(
                candles[index].open_time, close=100.0, open_=100.0, high=100.0, low=100.0, volume=0.0
            )
        outcome = _validate(validator, candles)
        event = outcome.events_of(QualityEventType.FLATLINE)[0]
        assert event.details["run_length"] >= 12

    def test_short_quiet_period_is_not_a_flatline(self, validator: CandleValidator) -> None:
        candles = series(30)
        for index in range(10, 13):
            candles[index] = make_candle(
                candles[index].open_time, close=100.0, open_=100.0, high=100.0, low=100.0, volume=0.0
            )
        assert not _validate(validator, candles).events_of(QualityEventType.FLATLINE)


class TestStaleness:
    def test_stale_feed_is_flagged(self, validator: CandleValidator) -> None:
        candles = series(5, start=FIXED_NOW - timedelta(hours=30))
        outcome = _validate(validator, candles, now=FIXED_NOW)
        event = outcome.events_of(QualityEventType.STALE_FEED)[0]
        assert event.severity is QualitySeverity.ERROR
        assert event.details["age_s"] > event.details["allowance_s"]

    def test_a_just_closed_bar_is_not_stale(self, validator: CandleValidator) -> None:
        candles = series(5, start=FIXED_NOW - timedelta(hours=5))
        outcome = _validate(validator, candles, now=FIXED_NOW + timedelta(minutes=1))
        assert not outcome.events_of(QualityEventType.STALE_FEED)

    def test_staleness_scales_with_the_timeframe(self, validator: CandleValidator) -> None:
        """Two hours old is dead on a 1m feed and perfectly normal on a daily one."""
        minute_series = series(5, start=FIXED_NOW - timedelta(hours=2), timeframe=Timeframe.M1)
        daily_series = series(5, start=FIXED_NOW - timedelta(days=5), timeframe=Timeframe.D1)

        stale = _validate(
            validator, minute_series, timeframe=Timeframe.M1, now=FIXED_NOW
        )
        fresh = _validate(
            validator, daily_series, timeframe=Timeframe.D1, now=FIXED_NOW
        )
        assert stale.events_of(QualityEventType.STALE_FEED)
        assert not fresh.events_of(QualityEventType.STALE_FEED)


class TestContext:
    def test_context_supplies_statistics_for_small_batches(
        self, validator: CandleValidator
    ) -> None:
        """A two-bar poll has no history of its own; stored bars provide the baseline."""
        history = series(60, base=100.0, step=0.0)
        incoming = [make_candle(FIXED_NOW, close=100_000.0, open_=100.0, low=100.0)]

        without = _validate(validator, incoming, now=FIXED_NOW + timedelta(hours=1))
        with_context = _validate(
            validator, incoming, context=history, now=FIXED_NOW + timedelta(hours=1)
        )

        assert not without.events_of(QualityEventType.IMPOSSIBLE_MOVE)
        assert with_context.events_of(QualityEventType.IMPOSSIBLE_MOVE)

    def test_only_batch_candles_are_reported_not_context(
        self, validator: CandleValidator
    ) -> None:
        """Context is there to inform the statistics, not to be re-flagged every poll."""
        history = series(40, base=100.0, step=0.0)
        history[20] = make_candle(history[20].open_time, close=100_000.0, open_=100.0, low=100.0)
        incoming = series(2, start=FIXED_NOW, base=100.0, step=0.0)

        outcome = _validate(
            validator, incoming, context=history, now=FIXED_NOW + timedelta(hours=3)
        )
        assert not outcome.events_of(QualityEventType.IMPOSSIBLE_MOVE)


class TestHistoricalWindows:
    """Staleness is a liveness question, so it must not fire on historical pages."""

    def test_a_stated_historical_window_is_not_flagged_as_stale(
        self, validator: CandleValidator
    ) -> None:
        candles = series(50, start=FIXED_NOW - timedelta(days=60))
        outcome = _validate(
            validator,
            candles,
            window_start=FIXED_NOW - timedelta(days=60),
            window_end=FIXED_NOW - timedelta(days=58),
            now=FIXED_NOW,
        )
        assert not outcome.events_of(QualityEventType.STALE_FEED)

    def test_a_current_window_still_reports_staleness(
        self, validator: CandleValidator
    ) -> None:
        candles = series(5, start=FIXED_NOW - timedelta(hours=40))
        outcome = _validate(
            validator,
            candles,
            window_start=FIXED_NOW - timedelta(hours=40),
            window_end=FIXED_NOW,
            now=FIXED_NOW,
        )
        assert outcome.events_of(QualityEventType.STALE_FEED)

    def test_ordinary_crypto_volatility_does_not_trip_the_outlier_check(
        self, validator: CandleValidator
    ) -> None:
        """Calibrated against real BTC returns: a 4% hourly move is ~20 robust sigma
        and is entirely normal, so the threshold must sit above it."""
        candles = series(200, base=100.0, step=0.0)
        for index in range(1, 200):
            level = 100.0 * (1 + 0.002 * (1 if index % 2 else -1))
            candles[index] = make_candle(candles[index].open_time, close=level)
        candles[150] = make_candle(candles[150].open_time, close=104.0, open_=100.0, low=100.0)

        outcome = _validate(validator, candles)
        assert not outcome.events_of(QualityEventType.OUTLIER)
