"""Storage layer: schema, upserts, and the queries analytics will depend on."""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW, make_candle, series

from mie.core.timeframes import UTC, Timeframe
from mie.core.types import (
    FundingRate,
    GlobalMetricsPoint,
    IngestResult,
    IngestStatus,
    MarketType,
    QualityEvent,
    QualityEventType,
    QualitySeverity,
)
from mie.storage.db import Database
from mie.storage.repositories import (
    DerivativesRepository,
    GlobalMetricsRepository,
    IngestRunRepository,
    OHLCVRepository,
    QualityRepository,
    ReferenceRepository,
)


class TestReference:
    async def test_ensure_asset_is_idempotent(self, database: Database) -> None:
        async with database.session() as session:
            repo = ReferenceRepository(session)
            first = await repo.ensure_asset("btc", "Bitcoin", tier=1)
            second = await repo.ensure_asset("BTC")
            assert first.id == second.id
            assert first.symbol == "BTC", "symbols are normalised to upper case"

    async def test_instruments_separate_sources_and_market_types(
        self, database: Database
    ) -> None:
        """The (asset, source, market) triple is what makes failover and cross-source
        comparison possible; collapsing it would merge two different venues' prices."""
        async with database.session() as session:
            repo = ReferenceRepository(session)
            binance_spot = await repo.ensure_instrument("BTC", "binance", "BTCUSDT")
            kraken_spot = await repo.ensure_instrument("BTC", "kraken", "XBTUSD")
            binance_perp = await repo.ensure_instrument(
                "BTC", "binance", "BTCUSDT", MarketType.PERP
            )
            assert len({binance_spot, kraken_spot, binance_perp}) == 3

    async def test_provider_symbol_changes_are_applied(self, database: Database) -> None:
        async with database.session() as session:
            repo = ReferenceRepository(session)
            first = await repo.ensure_instrument("BTC", "binance", "BTCUSD")
        ReferenceRepository.clear_cache()
        async with database.session() as session:
            repo = ReferenceRepository(session)
            second = await repo.ensure_instrument("BTC", "binance", "BTCUSDT")
            assert first == second


class TestOHLCVUpsert:
    async def test_round_trip_preserves_utc(self, database: Database) -> None:
        """SQLite has no timezone type; a naive value coming back would silently
        shift every timestamp in the system."""
        candles = series(5)
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            stored = await OHLCVRepository(session).fetch("BTC", Timeframe.H1)

        assert len(stored) == 5
        assert all(row.open_time.tzinfo is not None for row in stored)
        assert stored[0].open_time == candles[0].open_time
        assert stored[0].open_time.utcoffset() == timedelta(0)

    async def test_upsert_is_idempotent(self, database: Database) -> None:
        candles = series(10)
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            assert await OHLCVRepository(session).count("BTC", Timeframe.H1) == 10

    async def test_final_candle_replaces_provisional_one(self, database: Database) -> None:
        provisional = make_candle(FIXED_NOW, close=100.0, is_final=False)
        final = make_candle(FIXED_NOW, close=105.0, is_final=True)

        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles([provisional])
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles([final])
        async with database.session() as session:
            stored = await OHLCVRepository(session).fetch("BTC", Timeframe.H1, final_only=False)

        assert len(stored) == 1
        assert stored[0].close == 105.0
        assert stored[0].is_final is True
        assert stored[0].revision == 1, "rewrites are visible, not silent"

    async def test_provisional_candle_cannot_overwrite_a_final_one(
        self, database: Database
    ) -> None:
        """A late provisional bar must not reopen history the analytics already used."""
        final = make_candle(FIXED_NOW, close=105.0, is_final=True)
        stale_provisional = make_candle(FIXED_NOW, close=1.0, is_final=False)

        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles([final])
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles([stale_provisional])
        async with database.session() as session:
            stored = await OHLCVRepository(session).fetch("BTC", Timeframe.H1, final_only=False)

        assert stored[0].close == 105.0
        assert stored[0].is_final is True

    async def test_fetch_excludes_the_forming_bar_by_default(
        self, database: Database
    ) -> None:
        """Look-ahead defence: consumers must opt in to provisional data."""
        candles = [*series(3), make_candle(FIXED_NOW, close=200.0, is_final=False)]
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            repo = OHLCVRepository(session)
            assert len(await repo.fetch("BTC", Timeframe.H1)) == 3
            assert len(await repo.fetch("BTC", Timeframe.H1, final_only=False)) == 4

    async def test_sources_are_stored_independently(self, database: Database) -> None:
        left = make_candle(FIXED_NOW, close=100.0, source="binance")
        right = make_candle(FIXED_NOW, close=101.0, source="kraken")
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles([left, right])
        async with database.session() as session:
            repo = OHLCVRepository(session)
            assert await repo.count("BTC", Timeframe.H1) == 2
            assert await repo.count("BTC", Timeframe.H1, source="binance") == 1


class TestQueries:
    async def test_latest_and_earliest_open_time(self, database: Database) -> None:
        candles = series(20)
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            repo = OHLCVRepository(session)
            assert await repo.latest_open_time("BTC", Timeframe.H1) == candles[-1].open_time
            assert await repo.earliest_open_time("BTC", Timeframe.H1) == candles[0].open_time

    async def test_latest_open_time_ignores_the_forming_bar(
        self, database: Database
    ) -> None:
        """Backfill resumes from this value; including a provisional bar would skip it."""
        candles = series(5)
        provisional = make_candle(candles[-1].open_time + timedelta(hours=1), is_final=False)
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles([*candles, provisional])
        async with database.session() as session:
            latest = await OHLCVRepository(session).latest_open_time("BTC", Timeframe.H1)
        assert latest == candles[-1].open_time

    async def test_missing_windows_finds_interior_gaps(self, database: Database) -> None:
        candles = series(24, start=FIXED_NOW - timedelta(hours=24))
        kept = candles[:8] + candles[12:]
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(kept)
        async with database.session() as session:
            gaps = await OHLCVRepository(session).missing_windows(
                "BTC", Timeframe.H1, FIXED_NOW - timedelta(hours=24), FIXED_NOW
            )
        assert len(gaps) == 1
        assert gaps[0][0] == candles[8].open_time
        assert gaps[0][1] == candles[11].open_time

    async def test_missing_windows_is_empty_for_complete_history(
        self, database: Database
    ) -> None:
        candles = series(24, start=FIXED_NOW - timedelta(hours=24))
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            gaps = await OHLCVRepository(session).missing_windows(
                "BTC", Timeframe.H1, FIXED_NOW - timedelta(hours=24), FIXED_NOW
            )
        assert gaps == []

    async def test_coverage_reports_completeness(self, database: Database) -> None:
        candles = series(20, start=FIXED_NOW - timedelta(hours=20))
        del candles[5]
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            info = await OHLCVRepository(session).coverage("BTC", Timeframe.H1)
        assert info["rows"] == 19
        assert info["expected"] == 20
        assert info["completeness"] == pytest.approx(0.95)

    async def test_fetch_respects_the_half_open_range(self, database: Database) -> None:
        candles = series(10, start=FIXED_NOW - timedelta(hours=10))
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(candles)
        async with database.session() as session:
            rows = await OHLCVRepository(session).fetch(
                "BTC",
                Timeframe.H1,
                start=candles[2].open_time,
                end=candles[5].open_time,
            )
        assert [r.open_time for r in rows] == [c.open_time for c in candles[2:5]]


class TestQualityStore:
    async def test_events_persist_and_aggregate(self, database: Database) -> None:
        events = [
            QualityEvent(
                event_type=QualityEventType.GAP,
                severity=QualitySeverity.WARNING,
                source="fake",
                asset="BTC",
                timeframe=Timeframe.H1,
                message="test gap",
            ),
            QualityEvent(
                event_type=QualityEventType.OUTLIER,
                severity=QualitySeverity.WARNING,
                source="fake",
                asset="BTC",
                timeframe=Timeframe.H1,
                message="test outlier",
            ),
        ]
        async with database.session() as session:
            assert await QualityRepository(session).record_events(events) == 2
        async with database.session() as session:
            repo = QualityRepository(session)
            assert await repo.event_counts() == {"gap": 1, "outlier": 1}
            assert len(await repo.recent_events(asset="BTC")) == 2
            assert len(await repo.recent_events(severity="error")) == 0

    async def test_scores_upsert_and_default_optimistically(
        self, database: Database
    ) -> None:
        async with database.session() as session:
            repo = QualityRepository(session)
            # An unmeasured scope is not a bad scope.
            assert await repo.get_score("fake", "BTC", Timeframe.H1) == 1.0
            await repo.set_score("fake", "BTC", Timeframe.H1, 0.42, 7)
        async with database.session() as session:
            repo = QualityRepository(session)
            assert await repo.get_score("fake", "BTC", Timeframe.H1) == pytest.approx(0.42)
            await repo.set_score("fake", "BTC", Timeframe.H1, 0.88, 1)
        async with database.session() as session:
            repo = QualityRepository(session)
            assert await repo.get_score("fake", "BTC", Timeframe.H1) == pytest.approx(0.88)
            assert len(await repo.all_scores()) == 1, "upsert, not insert"


class TestProvenanceAndContext:
    async def test_ingest_runs_are_recorded(self, database: Database) -> None:
        result = IngestResult(
            job="backfill",
            asset="BTC",
            timeframe=Timeframe.H1,
            source="fake",
            status=IngestStatus.PARTIAL,
            rows_written=42,
        )
        async with database.session() as session:
            assert await IngestRunRepository(session).record(result)
        async with database.session() as session:
            runs = await IngestRunRepository(session).recent()
        assert runs[0].asset == "BTC"
        assert runs[0].status == "partial"
        assert runs[0].rows_written == 42

    async def test_funding_and_open_interest_round_trip(self, database: Database) -> None:
        point = FundingRate(asset="BTC", source="fake", ts=FIXED_NOW, rate=0.0001)
        async with database.session() as session:
            assert await DerivativesRepository(session).upsert_funding([point]) == 1
        async with database.session() as session:
            latest = await DerivativesRepository(session).latest_funding("BTC")
        assert latest is not None
        assert latest.rate == pytest.approx(0.0001)
        assert latest.ts == FIXED_NOW

    async def test_global_metrics_round_trip(self, database: Database) -> None:
        point = GlobalMetricsPoint(
            source="fake", ts=FIXED_NOW, btc_dominance=54.3, total_market_cap_usd=2.1e12
        )
        async with database.session() as session:
            await GlobalMetricsRepository(session).upsert(point)
        async with database.session() as session:
            latest = await GlobalMetricsRepository(session).latest()
        assert latest is not None
        assert latest.btc_dominance == pytest.approx(54.3)


class TestSchema:
    async def test_naive_datetimes_are_refused_at_the_boundary(
        self, database: Database
    ) -> None:
        """The type decorator rejects rather than assuming a timezone."""
        from datetime import datetime

        from mie.storage.models import UTCDateTime

        decorator = UTCDateTime()
        with pytest.raises(ValueError, match="naive datetime"):
            decorator.process_bind_param(datetime(2025, 6, 2, 12, 0), None)
        aware = datetime(2025, 6, 2, 12, 0, tzinfo=UTC)
        assert decorator.process_bind_param(aware, None) == aware

    async def test_healthcheck_and_dialect(self, database: Database) -> None:
        assert await database.healthcheck() is True
        assert database.dialect == "sqlite"


class TestQualityScopeIsolation:
    """Scores are per (source, asset, timeframe); events must not leak across scopes."""

    async def test_events_are_filtered_by_timeframe(self, database: Database) -> None:
        events = [
            QualityEvent(
                event_type=QualityEventType.GAP,
                severity=QualitySeverity.WARNING,
                source="fake",
                asset="BTC",
                timeframe=Timeframe.H1,
                message="hourly gap",
            ),
            QualityEvent(
                event_type=QualityEventType.OUTLIER,
                severity=QualitySeverity.WARNING,
                source="fake",
                asset="BTC",
                timeframe=Timeframe.M1,
                message="minute outlier",
            ),
        ]
        async with database.session() as session:
            await QualityRepository(session).record_events(events)
        async with database.session() as session:
            repo = QualityRepository(session)
            hourly = await repo.recent_events(asset="BTC", timeframe=Timeframe.H1)
            minutely = await repo.recent_events(asset="BTC", timeframe=Timeframe.M1)
            both = await repo.recent_events(asset="BTC")

        assert [e.message for e in hourly] == ["hourly gap"]
        assert [e.message for e in minutely] == ["minute outlier"]
        assert len(both) == 2, "omitting the filter still returns every scope"
