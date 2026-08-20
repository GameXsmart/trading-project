"""Quality scoring.

The score is what makes requirement §20 real: it is the number later phases multiply
into published confidence. These tests pin the properties that matter — that it
discriminates, that it is a rate rather than a count, that it recovers, and that it
never silently rates a dead feed as healthy.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW

from mie.config.settings import QualityConfig
from mie.core.timeframes import Timeframe
from mie.core.types import QualityEvent, QualityEventType, QualitySeverity
from mie.quality.scoring import QualityScorer


@pytest.fixture
def scorer() -> QualityScorer:
    return QualityScorer(QualityConfig())


def event(
    severity: QualitySeverity = QualitySeverity.WARNING,
    event_type: QualityEventType = QualityEventType.OUTLIER,
    age_hours: float = 0.0,
) -> QualityEvent:
    return QualityEvent(
        event_type=event_type,
        severity=severity,
        source="fake",
        asset="BTC",
        timeframe=Timeframe.H1,
        message="synthetic",
        detected_at=FIXED_NOW - timedelta(hours=age_hours),
    )


def _score(scorer: QualityScorer, events, assessed=1000, last_candle_offset_h=0.5):
    return scorer.score(
        source="fake",
        asset="BTC",
        timeframe=Timeframe.H1,
        events=events,
        last_candle_at=FIXED_NOW - timedelta(hours=last_candle_offset_h),
        candles_assessed=assessed,
        now=FIXED_NOW,
    )


class TestBaseline:
    def test_clean_feed_scores_one(self, scorer: QualityScorer) -> None:
        result = _score(scorer, [])
        assert result.score == 1.0
        assert not result.is_degraded
        assert "no quality issues" in result.explain()

    def test_score_is_always_in_range(self, scorer: QualityScorer) -> None:
        catastrophic = [event(QualitySeverity.ERROR) for _ in range(500)]
        result = _score(scorer, catastrophic, assessed=100)
        assert 0.0 <= result.score <= 1.0
        assert result.score >= QualityConfig().min_score


class TestRateNotCount:
    def test_the_same_events_score_differently_by_exposure(
        self, scorer: QualityScorer
    ) -> None:
        """The property the count-based formula could not express: forty warnings
        across a year of history is a healthy feed; forty in an hour is not."""
        events = [event() for _ in range(40)]
        sparse = _score(scorer, events, assessed=8760)
        dense = _score(scorer, events, assessed=60)
        assert sparse.score > 0.9
        assert dense.score < 0.3

    def test_a_few_warnings_do_not_saturate_the_score(
        self, scorer: QualityScorer
    ) -> None:
        """A score that pins to the floor on ordinary noise carries no information."""
        result = _score(scorer, [event() for _ in range(5)], assessed=2000)
        assert result.score > 0.95

    def test_missing_exposure_is_treated_conservatively(
        self, scorer: QualityScorer
    ) -> None:
        """An unmeasurable feed must not score as a clean one."""
        events = [event(QualitySeverity.ERROR) for _ in range(5)]
        unknown = _score(scorer, events, assessed=None)
        known = _score(scorer, events, assessed=5000)
        assert unknown.score < known.score


class TestSeverityAndRecency:
    def test_errors_cost_more_than_warnings(self, scorer: QualityScorer) -> None:
        warnings = _score(scorer, [event(QualitySeverity.WARNING) for _ in range(10)], 500)
        errors = _score(scorer, [event(QualitySeverity.ERROR) for _ in range(10)], 500)
        assert errors.score < warnings.score

    def test_info_events_are_free(self, scorer: QualityScorer) -> None:
        """Informational events are context, not defects."""
        result = _score(scorer, [event(QualitySeverity.INFO) for _ in range(50)], 500)
        assert result.score == 1.0

    def test_recent_events_weigh_more_than_old_ones(self, scorer: QualityScorer) -> None:
        fresh = _score(scorer, [event(age_hours=0) for _ in range(10)], 500)
        stale = _score(scorer, [event(age_hours=20) for _ in range(10)], 500)
        assert stale.score > fresh.score

    def test_events_outside_the_window_are_ignored(self, scorer: QualityScorer) -> None:
        """The score must recover on its own as incidents age out."""
        result = _score(scorer, [event(age_hours=48) for _ in range(50)], 500)
        assert result.score == 1.0
        assert result.events_in_window == 0


class TestStaleness:
    def test_a_silent_feed_is_penalised_without_any_events(
        self, scorer: QualityScorer
    ) -> None:
        """A feed that simply stops produces no events at all — an event-only score
        would rate it perfect."""
        result = _score(scorer, [], last_candle_offset_h=12)
        assert result.score < 1.0
        assert any("behind" in reason for reason in result.reasons)

    def test_no_data_at_all_is_heavily_penalised(self, scorer: QualityScorer) -> None:
        result = scorer.score(
            source="fake",
            asset="BTC",
            timeframe=Timeframe.H1,
            events=[],
            last_candle_at=None,
            now=FIXED_NOW,
        )
        assert result.score <= 0.5
        assert "no data observed" in result.reasons

    def test_freshness_is_judged_against_the_timeframe(
        self, scorer: QualityScorer
    ) -> None:
        """Six hours old is dead on a 1m feed and perfectly normal on a daily one."""
        minute = scorer.score(
            "fake", "BTC", Timeframe.M1, [], FIXED_NOW - timedelta(hours=6), 500, FIXED_NOW
        )
        daily = scorer.score(
            "fake", "BTC", Timeframe.D1, [], FIXED_NOW - timedelta(hours=6), 500, FIXED_NOW
        )
        assert minute.score < daily.score
        assert daily.score == 1.0


class TestExplainability:
    def test_reasons_name_the_dominant_defect(self, scorer: QualityScorer) -> None:
        """An operator must be able to look at a low score and see why."""
        events = [event(event_type=QualityEventType.GAP) for _ in range(20)]
        events += [event(event_type=QualityEventType.OUTLIER) for _ in range(2)]
        result = _score(scorer, events, assessed=200)
        assert any("gap" in reason for reason in result.reasons)
        assert "events/1k bars" in result.explain()

    def test_details_are_serialisable_for_storage(self, scorer: QualityScorer) -> None:
        details = _score(scorer, [event() for _ in range(3)], 500).as_details()
        assert set(details) == {"components", "reasons", "events_in_window"}
        assert isinstance(details["components"]["events"], float)


class TestThresholds:
    def test_degraded_and_unusable_bands(self, scorer: QualityScorer) -> None:
        healthy = _score(scorer, [])
        degraded = _score(scorer, [event() for _ in range(12)], assessed=300)
        broken = _score(scorer, [event(QualitySeverity.ERROR) for _ in range(80)], assessed=200)

        assert not healthy.is_degraded
        assert degraded.is_degraded
        assert broken.is_unusable, "below this, later phases publish nothing at all"

    def test_summarise_flags_problem_scopes(self, scorer: QualityScorer) -> None:
        scores = [
            _score(scorer, []),
            _score(scorer, [event(QualitySeverity.ERROR) for _ in range(80)], assessed=200),
        ]
        summary = QualityScorer.summarise(scores)
        assert summary["count"] == 2
        assert summary["unusable"] == ["fake/BTC/1h"]

    def test_summarise_handles_no_scores(self) -> None:
        assert QualityScorer.summarise([])["count"] == 0
