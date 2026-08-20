"""Ingestion: backfill, live polling, and the service that runs them.

These are integration tests — real database, real validation, real event bus, with
only the network replaced by the deterministic synthetic provider.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from mie.config.settings import Settings
from mie.core.events import Event, InProcessEventBus, Topics
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import IngestStatus, QualityEventType
from mie.ingestion.backfill import BackfillEngine
from mie.ingestion.live import LivePoller
from mie.ingestion.service import IngestionService
from mie.providers.fake import FakeProvider, fake_config
from mie.providers.manager import ProviderManager
from mie.storage.db import Database
from mie.storage.repositories import (
    IngestRunRepository,
    OHLCVRepository,
    QualityRepository,
)


@pytest.fixture
def bus() -> InProcessEventBus:
    return InProcessEventBus()


@pytest.fixture
def engine(
    database: Database, manager: ProviderManager, settings: Settings, bus: InProcessEventBus
) -> BackfillEngine:
    return BackfillEngine(database, manager, settings, bus)


class TestBackfill:
    async def test_fills_a_requested_window(
        self, engine: BackfillEngine, database: Database
    ) -> None:
        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=48)
        result = await engine.backfill("BTC", Timeframe.H1, start=start)

        assert result.status is IngestStatus.SUCCESS
        assert result.rows_written >= 47
        async with database.session() as session:
            stored = await OHLCVRepository(session).count("BTC", Timeframe.H1)
        assert stored == result.rows_written

    async def test_never_stores_the_forming_bar(
        self, engine: BackfillEngine, database: Database
    ) -> None:
        """History must contain only completed bars — this is the look-ahead guard."""
        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=10)
        await engine.backfill("BTC", Timeframe.H1, start=start)

        async with database.session() as session:
            rows = await OHLCVRepository(session).fetch(
                "BTC", Timeframe.H1, final_only=False
            )
        assert rows
        assert all(row.is_final for row in rows)
        assert all(row.open_time < Timeframe.H1.floor(utcnow()) for row in rows)

    async def test_is_idempotent(self, engine: BackfillEngine, database: Database) -> None:
        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=24)
        await engine.backfill("BTC", Timeframe.H1, start=start)
        async with database.session() as session:
            first = await OHLCVRepository(session).count("BTC", Timeframe.H1)

        await engine.backfill("BTC", Timeframe.H1, start=start, force=True)
        async with database.session() as session:
            second = await OHLCVRepository(session).count("BTC", Timeframe.H1)
        assert first == second

    async def test_resume_does_not_refetch_existing_history(
        self, database: Database, settings: Settings, provider: FakeProvider
    ) -> None:
        """A second run must fetch only what is missing, not re-download everything."""
        settings.ingestion.backfill_days[Timeframe.H1] = 2
        engine = BackfillEngine(database, ProviderManager([provider]), settings)

        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=48)
        await engine.backfill("BTC", Timeframe.H1, start=start)
        calls_after_first = provider.calls

        second = await engine.backfill("BTC", Timeframe.H1)
        assert provider.calls - calls_after_first <= 2
        assert second.status is IngestStatus.SKIPPED
        assert second.rows_written == 0

    async def test_increasing_configured_depth_deepens_history(
        self, database: Database, settings: Settings, provider: FakeProvider
    ) -> None:
        """The trap this guards against: a series that is current at its front edge
        looks 'up to date', so a forward-only resume would silently ignore a raised
        `backfill_days` and never deepen the history the macro models need."""
        settings.ingestion.backfill_days[Timeframe.H1] = 2
        engine = BackfillEngine(database, ProviderManager([provider]), settings)
        await engine.backfill("BTC", Timeframe.H1)

        async with database.session() as session:
            shallow = await OHLCVRepository(session).earliest_open_time("BTC", Timeframe.H1)

        settings.ingestion.backfill_days[Timeframe.H1] = 10
        result = await engine.backfill("BTC", Timeframe.H1)

        async with database.session() as session:
            deep = await OHLCVRepository(session).earliest_open_time("BTC", Timeframe.H1)
        assert result.rows_written > 0
        assert deep < shallow, "the leading segment should have been filled"

    async def test_both_ends_are_filled_in_one_pass(
        self, database: Database, settings: Settings, provider: FakeProvider
    ) -> None:
        """A stale, shallow series needs its head *and* its tail; planning both in one
        run means one command brings a neglected series fully current."""
        settings.ingestion.backfill_days[Timeframe.H1] = 30
        engine = BackfillEngine(database, ProviderManager([provider]), settings)

        now = Timeframe.H1.floor(utcnow())
        await engine.backfill(
            "BTC", Timeframe.H1, start=now - timedelta(hours=100), end=now - timedelta(hours=50)
        )
        result = await engine.backfill("BTC", Timeframe.H1)

        async with database.session() as session:
            repo = OHLCVRepository(session)
            gaps = await repo.missing_windows(
                "BTC", Timeframe.H1, now - timedelta(days=30), now
            )
        assert result.rows_written > 0
        assert gaps == [], "planning both segments should leave no hole"

    async def test_pages_through_windows_larger_than_one_request(
        self, database: Database, settings: Settings, bus: InProcessEventBus
    ) -> None:
        """A 300-candle provider cap must not silently truncate a 500-candle request."""
        provider = FakeProvider(fake_config("small_pages", max_candles=50))
        engine = BackfillEngine(database, ProviderManager([provider]), settings, bus)

        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=200)
        result = await engine.backfill("BTC", Timeframe.H1, start=start)

        assert provider.calls > 1, "the window should have required several pages"
        assert result.rows_written >= 195

    async def test_provider_gaps_are_detected_after_the_fact(
        self, database: Database, settings: Settings
    ) -> None:
        """Verification reads storage rather than trusting the fetch loop."""
        provider = FakeProvider(fake_config("gappy", drop_every=10))
        engine = BackfillEngine(database, ProviderManager([provider]), settings)

        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=100)
        result = await engine.backfill("BTC", Timeframe.H1, start=start)
        assert [e for e in result.quality_events if e.event_type is QualityEventType.GAP]

    async def test_malformed_candles_are_rejected_and_reported(
        self, database: Database, settings: Settings
    ) -> None:
        provider = FakeProvider(fake_config("broken", break_shape_every=8))
        engine = BackfillEngine(database, ProviderManager([provider]), settings)

        start = Timeframe.H1.floor(utcnow()) - timedelta(hours=60)
        result = await engine.backfill("BTC", Timeframe.H1, start=start)

        assert result.rows_rejected > 0
        assert [e for e in result.quality_events if e.event_type is QualityEventType.SHAPE_INVALID]
        async with database.session() as session:
            rows = await OHLCVRepository(session).fetch("BTC", Timeframe.H1)
        assert all(row.high >= row.low for row in rows), "no broken bar reached storage"

    async def test_total_provider_failure_is_reported_not_raised(
        self, database: Database, settings: Settings
    ) -> None:
        provider = FakeProvider(fake_config("dead", always_fail=True))
        engine = BackfillEngine(database, ProviderManager([provider]), settings)

        result = await engine.backfill(
            "BTC", Timeframe.H1, start=utcnow() - timedelta(hours=10)
        )
        assert result.status is IngestStatus.FAILED
        assert result.error

    async def test_runs_are_recorded_for_provenance(
        self, engine: BackfillEngine, database: Database
    ) -> None:
        await engine.backfill("BTC", Timeframe.H1, start=utcnow() - timedelta(hours=12))
        async with database.session() as session:
            runs = await IngestRunRepository(session).recent(job="backfill")
        assert runs
        assert runs[0].asset == "BTC"
        assert runs[0].rows_written > 0

    async def test_quality_events_are_persisted(
        self, database: Database, settings: Settings
    ) -> None:
        provider = FakeProvider(fake_config("gappy", drop_every=6))
        engine = BackfillEngine(database, ProviderManager([provider]), settings)
        await engine.backfill("BTC", Timeframe.H1, start=utcnow() - timedelta(hours=80))

        async with database.session() as session:
            counts = await QualityRepository(session).event_counts()
        assert counts

    async def test_publishes_candles_to_the_bus(
        self, engine: BackfillEngine, bus: InProcessEventBus
    ) -> None:
        """Phase 2's feature engine attaches here; the seam must actually fire."""
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(Topics.CANDLE_CLOSED, handler)
        await engine.backfill("BTC", Timeframe.H1, start=utcnow() - timedelta(hours=12))
        assert received
        assert received[0].meta["asset"] == "BTC"

    async def test_a_failing_subscriber_cannot_break_ingestion(
        self, engine: BackfillEngine, bus: InProcessEventBus, database: Database
    ) -> None:
        async def exploding(event: Event) -> None:
            raise RuntimeError("subscriber is broken")

        bus.subscribe(Topics.CANDLE_CLOSED, exploding)
        result = await engine.backfill(
            "BTC", Timeframe.H1, start=utcnow() - timedelta(hours=12)
        )
        assert result.rows_written > 0
        async with database.session() as session:
            assert await OHLCVRepository(session).count("BTC", Timeframe.H1) > 0


class TestLivePoller:
    @pytest.fixture
    def poller(
        self,
        database: Database,
        manager: ProviderManager,
        settings: Settings,
        bus: InProcessEventBus,
    ) -> LivePoller:
        return LivePoller(database, manager, settings, bus)

    async def test_poll_stores_recent_candles(
        self, poller: LivePoller, database: Database
    ) -> None:
        watch = poller.watch("BTC", Timeframe.H1)
        result = await poller.poll(watch)

        assert result.rows_written > 0
        async with database.session() as session:
            assert await OHLCVRepository(session).count("BTC", Timeframe.H1) > 0

    async def test_forming_bar_is_stored_as_provisional(
        self, poller: LivePoller, database: Database
    ) -> None:
        """The dashboard needs a live price; nothing else may treat it as history."""
        await poller.poll(poller.watch("BTC", Timeframe.H1))
        async with database.session() as session:
            repo = OHLCVRepository(session)
            everything = await repo.fetch("BTC", Timeframe.H1, final_only=False)
            final_only = await repo.fetch("BTC", Timeframe.H1)

        assert any(not row.is_final for row in everything)
        assert all(row.is_final for row in final_only)

    async def test_closes_are_published_exactly_once(
        self, poller: LivePoller, bus: InProcessEventBus
    ) -> None:
        seen: list[Event] = []

        async def handler(event: Event) -> None:
            seen.append(event)

        bus.subscribe(Topics.CANDLE_CLOSED, handler)
        watch = poller.watch("BTC", Timeframe.H1)
        await poller.poll(watch)
        first_count = len(seen)
        await poller.poll(watch)

        assert first_count > 0
        assert len(seen) == first_count, "re-polling the same bars must not re-publish"

    async def test_scheduling_targets_the_bar_boundary(self, poller: LivePoller) -> None:
        """Polling a 1h series every 20s would be ~180× the necessary request volume."""
        now = utcnow()
        watch = poller.watch("BTC", Timeframe.H1)
        await poller.poll(watch, now=now)
        assert watch.next_poll_at > now
        assert watch.next_poll_at <= Timeframe.H1.ceil(now) + timedelta(seconds=10)

    async def test_failures_back_off_and_are_recorded(
        self, database: Database, settings: Settings
    ) -> None:
        provider = FakeProvider(fake_config("dead", always_fail=True))
        poller = LivePoller(database, ProviderManager([provider]), settings)
        watch = poller.watch("BTC", Timeframe.H1)

        result = await poller.poll(watch)
        assert result.status is IngestStatus.FAILED
        assert watch.consecutive_failures == 1
        async with database.session() as session:
            assert await QualityRepository(session).recent_events()

    async def test_tick_only_polls_due_series(self, poller: LivePoller) -> None:
        watch = poller.watch("BTC", Timeframe.H1)
        assert await poller.tick()  # first tick: due immediately
        watch.next_poll_at = utcnow() + timedelta(hours=1)
        assert await poller.tick() == []

    async def test_one_bad_series_does_not_stop_the_others(
        self, database: Database, settings: Settings
    ) -> None:
        provider = FakeProvider(fake_config("partial", unsupported_assets=["SOL"]))
        poller = LivePoller(database, ProviderManager([provider]), settings)
        poller.watch("BTC", Timeframe.H1)
        poller.watch("SOL", Timeframe.H1)

        results = await poller.tick()
        statuses = {r.asset: r.status for r in results}
        assert statuses["BTC"] is not IngestStatus.FAILED
        assert statuses["SOL"] is IngestStatus.FAILED

    async def test_watchlist_economises_on_tier_two_assets(
        self, poller: LivePoller
    ) -> None:
        """Tier-2 assets skip the 1m series: highest cost, weakest signal."""
        poller.watch_universe()
        keys = {w.key for w in poller.watches}
        assert "BTC:1m" in keys
        assert "SOL:1m" not in keys
        assert "SOL:1h" in keys

    async def test_run_loop_stops_promptly(self, poller: LivePoller) -> None:
        poller.watch("BTC", Timeframe.H1)
        stop = asyncio.Event()
        task = asyncio.create_task(poller.run(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)


class TestIngestionService:
    @pytest.fixture
    def service(
        self, settings: Settings, database: Database, manager: ProviderManager
    ) -> IngestionService:
        return IngestionService(settings, database=database, manager=manager)

    async def test_bootstrap_is_idempotent(self, service: IngestionService) -> None:
        await service.bootstrap()
        await service.bootstrap()
        async with service.db.session() as session:
            from mie.storage.repositories import ReferenceRepository

            reference = ReferenceRepository(session)
            assert len(await reference.list_assets()) == 3
            assert len(await reference.list_sources()) == 1

    async def test_backfill_all_covers_the_matrix(self, service: IngestionService) -> None:
        await service.bootstrap()
        results = await service.backfill_all(
            assets=["BTC", "ETH"], timeframes=[Timeframe.H1, Timeframe.D1]
        )
        covered = {(r.asset, str(r.timeframe)) for r in results}
        assert covered == {("BTC", "1h"), ("BTC", "1d"), ("ETH", "1h"), ("ETH", "1d")}

    async def test_backfill_all_prioritises_slower_timeframes(
        self, service: IngestionService
    ) -> None:
        """An interrupted backfill should leave a usable coarse history behind."""
        await service.bootstrap()
        results = await service.backfill_all(
            assets=["BTC"], timeframes=[Timeframe.M1, Timeframe.D1, Timeframe.H1]
        )
        assert [str(r.timeframe) for r in results] == ["1d", "1h", "1m"]

    async def test_quality_scores_are_computed_and_stored(
        self, service: IngestionService
    ) -> None:
        await service.bootstrap()
        await service.backfill_all(assets=["BTC"], timeframes=[Timeframe.H1])
        assert await service.refresh_quality_scores(["BTC"], [Timeframe.H1]) > 0

        async with service.db.session() as session:
            score = await QualityRepository(session).get_score("fake", "BTC", Timeframe.H1)
        assert 0.0 < score <= 1.0

    async def test_clean_data_scores_better_than_broken_data(
        self, database: Database, settings: Settings
    ) -> None:
        """The whole point of the score: bad input must depress downstream confidence."""
        clean = IngestionService(
            settings,
            database=database,
            manager=ProviderManager([FakeProvider(fake_config("fake"))]),
        )
        await clean.bootstrap()
        await clean.backfill_all(assets=["BTC"], timeframes=[Timeframe.H1])
        await clean.refresh_quality_scores(["BTC"], [Timeframe.H1])
        async with database.session() as session:
            clean_score = await QualityRepository(session).get_score(
                "fake", "BTC", Timeframe.H1
            )

        broken = IngestionService(
            settings,
            database=database,
            manager=ProviderManager(
                [FakeProvider(fake_config("fake", drop_every=4, break_shape_every=9))]
            ),
        )
        await broken.backfill_all(assets=["ETH"], timeframes=[Timeframe.H1], force=True)
        await broken.refresh_quality_scores(["ETH"], [Timeframe.H1])
        async with database.session() as session:
            broken_score = await QualityRepository(session).get_score(
                "fake", "ETH", Timeframe.H1
            )

        assert broken_score < clean_score
