"""Live polling.

Keeps the recent tail of every watched series current and emits `candle.closed`
events — the seam Phase 2's feature engine will attach to.

Design notes:

* **Poll on the bar boundary, not on a fixed timer.** A 1h series does not need to be
  polled every 20 seconds; each (asset, timeframe) is scheduled for shortly after its
  bar is due to close. This is the difference between ~10 requests/minute and ~500 for
  the same freshness.
* **Overlap on purpose.** Each poll re-fetches the last few bars rather than only the
  newest. Exchanges revise recently-closed bars, and the overlap is also what lets the
  poller heal a short outage without a separate repair job. Upserts make it free.
* **Publish once.** A bar becoming final is an event that fires exactly once, tracked
  per series, so a subscriber never sees the same close twice.

This is polling, not streaming. WebSocket feeds are Phase 12: they are strictly better
for latency and strictly worse for reliability, and the correct thing to build first is
the fallback path that has to exist anyway.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from mie.config.settings import Settings
from mie.core.events import Event, EventBus, Topics
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import Candle, IngestResult, IngestStatus
from mie.providers.manager import ProviderManager
from mie.quality.validators import CandleValidator
from mie.storage.db import Database
from mie.storage.repositories import OHLCVRepository, QualityRepository

log = get_logger(__name__)

__all__ = ["LivePoller", "SeriesWatch"]

#: Grace period after a bar's close before polling for it. Exchanges do not publish
#: the closed bar the instant the clock ticks over.
_PUBLICATION_LAG_S = 2.0


@dataclass(slots=True)
class SeriesWatch:
    """Polling state for one (asset, timeframe) pair."""

    asset: str
    timeframe: Timeframe
    next_poll_at: datetime
    last_final_open_time: datetime | None = None
    consecutive_failures: int = 0
    polls: int = 0
    candles_written: int = 0
    context: list[Candle] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.asset}:{self.timeframe}"


class LivePoller:
    """Keeps watched series current and publishes bar closes."""

    def __init__(
        self,
        database: Database,
        manager: ProviderManager,
        settings: Settings,
        bus: EventBus | None = None,
    ) -> None:
        self.db = database
        self.manager = manager
        self.settings = settings
        self.bus = bus
        self.validator = CandleValidator(settings.quality)
        self._watches: dict[str, SeriesWatch] = {}
        self._running = False
        self._semaphore = asyncio.Semaphore(settings.ingestion.max_concurrency)

    # ------------------------------------------------------------------ watchlist

    def watch(self, asset: str, timeframe: Timeframe) -> SeriesWatch:
        watch = SeriesWatch(
            asset=asset.upper(),
            timeframe=timeframe,
            # Poll immediately on registration so a fresh start is not idle for an
            # entire bar before producing anything.
            next_poll_at=utcnow(),
        )
        self._watches[watch.key] = watch
        return watch

    def watch_universe(self) -> list[SeriesWatch]:
        """Register the configured universe.

        Tier-1 assets get every live timeframe; tier-2 assets skip the 1m series. The
        fastest resolution is where request volume explodes and where the signal on a
        thinner market is weakest, so that is the right place to economise.
        """
        watches: list[SeriesWatch] = []
        for asset in self.settings.universe.enabled():
            for timeframe in self.settings.ingestion.live_timeframes:
                if asset.tier > 1 and timeframe is Timeframe.M1:
                    continue
                watches.append(self.watch(asset.symbol, timeframe))
        log.info("live_watchlist_registered", series=len(watches))
        return watches

    @property
    def watches(self) -> list[SeriesWatch]:
        return list(self._watches.values())

    # --------------------------------------------------------------------- runloop

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Poll due series until stopped."""
        if not self._watches:
            self.watch_universe()
        self._running = True
        stop = stop or asyncio.Event()
        interval = self.settings.ingestion.poll_interval_s
        log.info("live_poller_started", series=len(self._watches), tick_s=interval)

        try:
            while not stop.is_set():
                await self.tick()
                try:
                    # Wait on the stop event rather than sleeping, so shutdown is
                    # immediate instead of taking up to a full tick.
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    continue
        finally:
            self._running = False
            log.info("live_poller_stopped")

    async def tick(self, now: datetime | None = None) -> list[IngestResult]:
        """Poll every series whose next bar should have closed by now."""
        now = now or utcnow()
        due = [w for w in self._watches.values() if w.next_poll_at <= now]
        if not due:
            return []

        async def _guarded(watch: SeriesWatch) -> IngestResult:
            async with self._semaphore:
                return await self.poll(watch, now=now)

        results = await asyncio.gather(*(_guarded(w) for w in due), return_exceptions=True)
        collected: list[IngestResult] = []
        for watch, result in zip(due, results, strict=True):
            if isinstance(result, BaseException):
                # One bad series must not stop the others; record and keep going.
                watch.consecutive_failures += 1
                log.error("poll_crashed", series=watch.key, error=str(result)[:300])
                self._reschedule(watch, now)
                continue
            collected.append(result)
        return collected

    async def poll(self, watch: SeriesWatch, now: datetime | None = None) -> IngestResult:
        """Fetch and store the recent tail of one series."""
        now = now or utcnow()
        watch.polls += 1
        result = IngestResult(
            job="live", asset=watch.asset, timeframe=watch.timeframe, started_at=now
        )

        lookback = max(2, self.settings.ingestion.lookback_candles_on_poll)
        window_start = watch.timeframe.floor(now) - watch.timeframe.delta * (lookback - 1)

        outcome = await self.manager.fetch_ohlcv(
            watch.asset,
            watch.timeframe,
            start=window_start,
            limit=lookback + 1,
            quote=self.settings.universe.default_quote,
        )
        result.quality_events.extend(outcome.events)
        result.source = outcome.provider

        if not outcome.ok:
            watch.consecutive_failures += 1
            result.status = IngestStatus.FAILED
            result.error = outcome.error
            result.finished_at = utcnow()
            await self._persist_events(result)
            self._reschedule(watch, now)
            return result

        watch.consecutive_failures = 0
        result.rows_fetched = len(outcome.candles)

        validation = self.validator.validate(
            outcome.candles,
            asset=watch.asset,
            timeframe=watch.timeframe,
            source=outcome.provider or "unknown",
            context=watch.context,
            now=now,
        )
        result.quality_events.extend(validation.events)
        result.rows_rejected = validation.rejected_count

        if validation.candles:
            async with self.db.session() as session:
                # The forming bar is stored too, flagged provisional, so the dashboard
                # can show a live price without any consumer mistaking it for history.
                result.rows_written = await OHLCVRepository(session).upsert_candles(
                    validation.candles
                )
            watch.candles_written += result.rows_written
            result.covered_start = validation.candles[0].open_time
            result.covered_end = validation.candles[-1].open_time

            final = [c for c in validation.candles if c.is_final]
            watch.context = (watch.context + final)[-200:]
            await self._publish_closes(watch, final)
        else:
            result.status = IngestStatus.SKIPPED

        result.finished_at = utcnow()
        await self._persist_events(result)
        self._reschedule(watch, now)
        return result

    # ------------------------------------------------------------------ internals

    async def _publish_closes(self, watch: SeriesWatch, final: list[Candle]) -> None:
        """Emit `candle.closed` exactly once per newly-completed bar."""
        if not final:
            return
        fresh = [
            c
            for c in final
            if watch.last_final_open_time is None or c.open_time > watch.last_final_open_time
        ]
        if not fresh:
            return
        watch.last_final_open_time = max(c.open_time for c in fresh)

        if self.bus is None:
            return
        for candle in fresh:
            await self.bus.publish(
                Event(
                    topic=Topics.CANDLE_CLOSED,
                    payload=candle,
                    meta={
                        "asset": candle.asset,
                        "timeframe": str(candle.timeframe),
                        "source": candle.source,
                        "origin": "live",
                    },
                )
            )

    async def _persist_events(self, result: IngestResult) -> None:
        if not result.quality_events:
            return
        try:
            async with self.db.session() as session:
                await QualityRepository(session).record_events(result.quality_events)
        except Exception as exc:
            log.error("quality_event_persist_failed", error=str(exc)[:300])

    def _reschedule(self, watch: SeriesWatch, now: datetime) -> None:
        """Schedule the next poll just after the current bar is due to close.

        Repeated failures back off exponentially (capped at one bar), so a broken
        series stops competing for the request budget with healthy ones.
        """
        next_close = watch.timeframe.ceil(now)
        if next_close <= now:
            next_close = next_close + watch.timeframe.delta
        target = next_close.timestamp() + _PUBLICATION_LAG_S

        if watch.consecutive_failures:
            backoff = min(
                watch.timeframe.seconds,
                self.settings.ingestion.poll_interval_s * (2 ** min(watch.consecutive_failures, 6)),
            )
            target = min(target, now.timestamp() + backoff)

        watch.next_poll_at = datetime.fromtimestamp(target, tz=now.tzinfo)

    def stats(self) -> dict[str, object]:
        return {
            "running": self._running,
            "series": len(self._watches),
            "polls": sum(w.polls for w in self._watches.values()),
            "candles_written": sum(w.candles_written for w in self._watches.values()),
            "failing": [w.key for w in self._watches.values() if w.consecutive_failures],
        }
