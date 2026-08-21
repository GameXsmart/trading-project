"""Ingestion service — the Phase 1 composition root.

Owns the object graph (settings → database → providers → engines → bus) and runs the
concurrent loops that keep the store current:

* **live polling** — recent candles for every watched series;
* **derivatives** — funding and open interest on a slower cadence;
* **global metrics** — dominance and aggregate capital, slower still;
* **quality scoring** — recompute the trust score per series from recent events.

Each loop is independent and failure-isolated. A CoinGecko rate-limit must not stop
candle ingestion, so a crashing loop is logged and restarted rather than taking the
process down with it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

from mie.config.settings import Settings
from mie.core.events import EventBus, InProcessEventBus
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import IngestResult, IngestStatus
from mie.features.engine import FeatureEngine
from mie.ingestion.backfill import BackfillEngine
from mie.ingestion.live import LivePoller
from mie.providers.manager import ProviderManager, build_providers
from mie.quality.scoring import QualityScorer
from mie.storage.db import Database
from mie.storage.repositories import (
    DerivativesRepository,
    GlobalMetricsRepository,
    OHLCVRepository,
    QualityRepository,
    ReferenceRepository,
)

log = get_logger(__name__)

__all__ = ["IngestionService", "ServiceStats"]


@dataclass(slots=True)
class ServiceStats:
    started_at: datetime
    ticks: int = 0
    candles_written: int = 0
    funding_points: int = 0
    oi_points: int = 0
    global_points: int = 0
    errors: int = 0


class IngestionService:
    """Composition root and process supervisor for Phase 1."""

    def __init__(
        self,
        settings: Settings,
        database: Database | None = None,
        manager: ProviderManager | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.settings = settings
        self.db = database or Database(settings)
        self.manager = manager or ProviderManager(build_providers(settings))
        self.bus = bus or InProcessEventBus()
        self.backfill_engine = BackfillEngine(self.db, self.manager, settings, self.bus)
        self.poller = LivePoller(self.db, self.manager, settings, self.bus)
        # Phase 2 attaches to the same event stream ingestion already publishes; the
        # poller does not know it exists.
        self.features = FeatureEngine(self.db, settings, self.bus)
        self.scorer = QualityScorer(settings.quality)
        self.stats = ServiceStats(started_at=utcnow())
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ bootstrap

    async def bootstrap(self) -> None:
        """Create the schema and register the configured universe and sources.

        Idempotent: safe to run on every start, which is what makes a fresh clone
        one command away from working.
        """
        await self.db.create_schema()
        async with self.db.session() as session:
            reference = ReferenceRepository(session)
            for provider in self.manager.providers:
                await reference.ensure_source(
                    provider.name, kind=provider.kind, priority=provider.config.priority
                )
            for asset in self.settings.universe.enabled():
                await reference.ensure_asset(asset.symbol, asset.name, asset.tier)
        log.info(
            "bootstrap_complete",
            assets=len(self.settings.universe.enabled()),
            providers=len(self.manager.providers),
        )

    async def start_features(self, warm: bool = True) -> None:
        """Subscribe the feature engine and prime it from stored history.

        Warming first matters: a cold engine emits nothing until 200+ new bars have
        arrived, which on a daily series is most of a year of silence.
        """
        self.features.subscribe()
        if not warm:
            return
        primary = next(
            (p.name for p in self.manager.providers if p.capabilities.timeframes), None
        )
        if primary is None:
            return
        for asset in self.settings.universe.enabled():
            for timeframe in self.settings.ingestion.live_timeframes:
                await self.features.warmup(asset.symbol, timeframe, primary)

    # ------------------------------------------------------------------- backfill

    async def backfill_all(
        self,
        assets: list[str] | None = None,
        timeframes: list[Timeframe] | None = None,
        force: bool = False,
        source: str | None = None,
    ) -> list[IngestResult]:
        """Backfill the requested matrix, bounded by the configured concurrency.

        Slower timeframes run first: they are cheap, they are what the macro layer of
        the multi-timeframe model needs most, and getting them in place means an
        interrupted backfill still leaves a usable coarse history behind.
        """
        symbols = [s.upper() for s in (assets or self.settings.universe.symbols())]
        frames = timeframes or self.settings.ingestion.timeframes
        ordered = sorted(frames, key=lambda tf: -tf.seconds)

        semaphore = asyncio.Semaphore(self.settings.ingestion.max_concurrency)
        results: list[IngestResult] = []

        async def _one(asset: str, timeframe: Timeframe) -> IngestResult:
            async with semaphore:
                return await self.backfill_engine.backfill(
                    asset, timeframe, force=force, source=source
                )

        for timeframe in ordered:
            batch = await asyncio.gather(
                *(_one(asset, timeframe) for asset in symbols), return_exceptions=True
            )
            for asset, outcome in zip(symbols, batch, strict=True):
                if isinstance(outcome, BaseException):
                    self.stats.errors += 1
                    log.error(
                        "backfill_crashed",
                        asset=asset,
                        timeframe=str(timeframe),
                        error=str(outcome)[:300],
                    )
                    continue
                results.append(outcome)
                self.stats.candles_written += outcome.rows_written
            await self.refresh_quality_scores(symbols, [timeframe])

        succeeded = sum(1 for r in results if r.status is not IngestStatus.FAILED)
        log.info(
            "backfill_all_complete",
            series=len(results),
            succeeded=succeeded,
            rows=sum(r.rows_written for r in results),
        )
        return results

    # ----------------------------------------------------------------------- run

    async def run(self) -> None:
        """Run every loop until stopped."""
        self._stop.clear()
        await self.start_features()
        loops = [
            self._supervise("live", self.poller.run(self._stop)),
            self._supervise("derivatives", self._derivatives_loop()),
            self._supervise("global_metrics", self._global_metrics_loop()),
            self._supervise("quality", self._quality_loop()),
        ]
        log.info("ingestion_service_running", loops=len(loops))
        await asyncio.gather(*loops)

    async def stop(self) -> None:
        self._stop.set()

    async def _supervise(self, name: str, coro) -> None:
        """Run a loop, logging and absorbing its failure rather than propagating it."""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stats.errors += 1
            log.error("loop_failed", loop=name, error=str(exc)[:400], exc_info=True)

    async def _derivatives_loop(self) -> None:
        if not self.settings.ingestion.collect_derivatives:
            return
        interval = self.settings.ingestion.derivatives_interval_s
        while not self._stop.is_set():
            for asset in self.settings.universe.enabled():
                if self._stop.is_set():
                    break
                try:
                    funding = await self.manager.fetch_funding(asset.symbol, limit=50)
                    oi = await self.manager.fetch_open_interest(asset.symbol, limit=50)
                    if funding or oi:
                        async with self.db.session() as session:
                            repo = DerivativesRepository(session)
                            self.stats.funding_points += await repo.upsert_funding(funding)
                            self.stats.oi_points += await repo.upsert_open_interest(oi)
                except Exception as exc:
                    # Derivatives are context, not the backbone: log and continue.
                    log.warning(
                        "derivatives_fetch_failed", asset=asset.symbol, error=str(exc)[:200]
                    )
            await self._sleep(interval)

    async def _global_metrics_loop(self) -> None:
        if not self.settings.ingestion.collect_global_metrics:
            return
        interval = self.settings.ingestion.global_metrics_interval_s
        while not self._stop.is_set():
            try:
                point = await self.manager.fetch_global_metrics()
                if point is not None:
                    async with self.db.session() as session:
                        await GlobalMetricsRepository(session).upsert(point)
                    self.stats.global_points += 1
            except Exception as exc:
                log.warning("global_metrics_failed", error=str(exc)[:200])
            await self._sleep(interval)

    async def _quality_loop(self) -> None:
        """Recompute trust scores on the quality window's cadence."""
        interval = max(60.0, self.settings.quality.score_window_hours * 3600 / 24)
        while not self._stop.is_set():
            try:
                await self.refresh_quality_scores()
            except Exception as exc:
                log.warning("quality_scoring_failed", error=str(exc)[:200])
            await self._sleep(interval)

    async def _sleep(self, seconds: float) -> None:
        """Interruptible sleep: shutdown should not wait out a ten-minute interval."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            return

    # ------------------------------------------------------------------- scoring

    async def refresh_quality_scores(
        self, assets: list[str] | None = None, timeframes: list[Timeframe] | None = None
    ) -> int:
        """Recompute and persist the rolling quality score for each series."""
        symbols = [s.upper() for s in (assets or self.settings.universe.symbols())]
        frames = timeframes or self.settings.ingestion.live_timeframes
        window = self.settings.quality.score_window_hours

        since = utcnow() - timedelta(hours=window)
        updated = 0
        async with self.db.session() as session:
            quality = QualityRepository(session)
            ohlcv = OHLCVRepository(session)
            for provider in self.manager.providers:
                if not provider.capabilities.timeframes:
                    continue  # aggregators serve no candles, so they score nothing
                for asset in symbols:
                    for timeframe in frames:
                        if not provider.supports(asset, timeframe):
                            continue
                        events = await quality.recent_events(
                            hours=window,
                            source=provider.name,
                            asset=asset,
                            timeframe=timeframe,
                            limit=500,
                        )
                        last_candle = await ohlcv.latest_open_time(
                            asset, timeframe, source=provider.name
                        )
                        # No data at all from this source is not a quality failure —
                        # it may simply never have been asked. Skip rather than
                        # manufacture a bad score.
                        if last_candle is None and not events:
                            continue
                        # Exposure: bars this source actually delivered in the window.
                        # Without it the event count has no scale, and a long clean
                        # backfill would score the same as a short broken one.
                        assessed = await ohlcv.count_ingested_since(
                            asset, timeframe, since, source=provider.name
                        )
                        score = self.scorer.score(
                            source=provider.name,
                            asset=asset,
                            timeframe=timeframe,
                            events=[
                                _to_event(row, provider.name, asset, timeframe) for row in events
                            ],
                            last_candle_at=last_candle,
                            candles_assessed=assessed,
                        )
                        await quality.set_score(
                            source=provider.name,
                            asset=asset,
                            timeframe=timeframe,
                            score=score.score,
                            events_in_window=score.events_in_window,
                            last_candle_at=last_candle,
                            details=score.as_details(),
                        )
                        updated += 1
        log.debug("quality_scores_refreshed", scopes=updated)
        return updated

    # --------------------------------------------------------------------- close

    async def close(self) -> None:
        await self.stop()
        await self.manager.close()
        await self.db.dispose()

    async def __aenter__(self) -> IngestionService:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def _to_event(row, source: str, asset: str, timeframe: Timeframe):
    """Rehydrate a stored quality-event row into the domain type the scorer expects."""
    from mie.core.types import QualityEvent, QualityEventType, QualitySeverity

    return QualityEvent(
        event_type=QualityEventType(row.event_type),
        severity=QualitySeverity(row.severity),
        source=source,
        asset=asset,
        timeframe=timeframe,
        window_start=row.window_start,
        window_end=row.window_end,
        message=row.message,
        details=row.details or {},
        detected_at=row.detected_at,
    )
