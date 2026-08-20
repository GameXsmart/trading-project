"""Historical backfill.

Walks a provider's pagination forward in time, validating and persisting each page,
then verifies the result against the expected grid. Three properties matter:

* **Resumable in both directions.** It plans *segments* rather than one window: the
  tail after the newest stored bar, and — when the configured depth reaches further
  back than the oldest stored bar — the leading segment before it. Extending
  ``backfill_days`` in config therefore actually deepens history, instead of silently
  doing nothing because the series looks up to date. ``force`` re-fetches outright.
* **Loop-safe.** A provider that returns nothing, or returns bars that do not advance
  the cursor, must not spin forever. Every path through the page loop either advances
  the cursor or terminates.
* **Honest about what it got.** The result reports the range actually covered and the
  gaps that remain, rather than reporting success because the requests returned 200.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from mie.config.settings import Settings
from mie.core.events import Event, EventBus, Topics
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc, utcnow
from mie.core.types import (
    Candle,
    IngestResult,
    IngestStatus,
    QualityEvent,
    QualityEventType,
    QualitySeverity,
)
from mie.providers.manager import ProviderManager
from mie.quality.validators import CandleValidator
from mie.storage.db import Database
from mie.storage.repositories import (
    IngestRunRepository,
    OHLCVRepository,
    QualityRepository,
)

log = get_logger(__name__)

__all__ = ["BackfillEngine"]

#: How many previously-seen candles to keep as statistical context for the outlier
#: detector. Enough for a stable MAD estimate, small enough to stay cheap.
_CONTEXT_CANDLES = 200

#: Hard ceiling on pages per backfill call. A misbehaving provider that always
#: returns one candle would otherwise page forever; this turns that into a reported
#: partial result instead of a hung process.
_MAX_PAGES = 5_000


class BackfillEngine:
    """Fetches and stores historical candles for one (asset, timeframe) at a time."""

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

    async def backfill(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
        force: bool = False,
        quote: str | None = None,
    ) -> IngestResult:
        """Backfill one series. Never raises for data problems — it reports them."""
        asset = asset.upper()
        quote = quote or self.settings.universe.default_quote
        result = IngestResult(
            job="backfill", asset=asset, timeframe=timeframe, source=source, started_at=utcnow()
        )

        try:
            segments, window_end = await self._plan_segments(
                asset, timeframe, start, end, source, force
            )
        except Exception as exc:
            result.status = IngestStatus.FAILED
            result.error = f"could not resolve window: {exc}"
            result.finished_at = utcnow()
            return result

        if not segments:
            result.requested_end = window_end
            result.status = IngestStatus.SKIPPED
            result.finished_at = utcnow()
            log.debug("backfill_up_to_date", asset=asset, timeframe=str(timeframe))
            await self._record(result)
            return result

        result.requested_start = min(a for a, _ in segments)
        result.requested_end = max(b for _, b in segments)

        log.info(
            "backfill_started",
            asset=asset,
            timeframe=str(timeframe),
            segments=[f"{a.isoformat()}..{b.isoformat()}" for a, b in segments],
            source=source or "auto",
        )

        pages = 0
        first_written: datetime | None = None
        last_written: datetime | None = None
        served_by: set[str] = set()

        for segment_start, segment_end in segments:
            # Context is reloaded per segment: the bars preceding a leading segment are
            # a different population from those preceding the tail, and stale context
            # would skew the outlier statistics.
            context: list[Candle] = await self._load_context(
                asset, timeframe, segment_start, source
            )
            cursor = segment_start
            stop_all = False

            while cursor < segment_end and pages < _MAX_PAGES:
                pages += 1
                page_limit = self._page_limit(asset, timeframe, cursor, segment_end, source)
                page_end = min(segment_end, cursor + timeframe.delta * page_limit)

                outcome = await self.manager.fetch_ohlcv(
                    asset,
                    timeframe,
                    start=cursor,
                    end=page_end,
                    limit=page_limit,
                    quote=quote,
                    preferred=source,
                )
                result.quality_events.extend(outcome.events)

                if not outcome.ok:
                    # Distinguish "this provider has nothing here" from "everything is
                    # broken": an empty page is usually the former — most commonly the
                    # window predates the asset's listing.
                    if outcome.attempts and all(kind == "empty" for _, kind in outcome.attempts):
                        result.quality_events.append(
                            self._event(
                                QualityEventType.GAP,
                                QualitySeverity.INFO,
                                asset,
                                timeframe,
                                source or "manager",
                                f"no data available from {cursor.isoformat()} "
                                f"to {page_end.isoformat()}",
                                window_start=cursor,
                                window_end=page_end,
                            )
                        )
                        cursor = page_end
                        continue
                    result.status = (
                        IngestStatus.PARTIAL if result.rows_written else IngestStatus.FAILED
                    )
                    result.error = outcome.error
                    stop_all = True
                    break

                served_by.add(outcome.provider or "unknown")
                result.rows_fetched += len(outcome.candles)

                validation = self.validator.validate(
                    outcome.candles,
                    asset=asset,
                    timeframe=timeframe,
                    source=outcome.provider or "unknown",
                    window_start=cursor,
                    window_end=page_end,
                    context=context,
                )
                result.quality_events.extend(validation.events)
                result.rows_rejected += validation.rejected_count

                # Only completed bars belong in history. The forming bar is picked up
                # by the live poller, which knows to mark it provisional.
                final_candles = [c for c in validation.candles if c.is_final]

                if final_candles:
                    async with self.db.session() as session:
                        written = await OHLCVRepository(session).upsert_candles(final_candles)
                    result.rows_written += written
                    first = final_candles[0].open_time
                    last = final_candles[-1].open_time
                    first_written = first if first_written is None else min(first_written, first)
                    last_written = last if last_written is None else max(last_written, last)
                    context = (context + final_candles)[-_CONTEXT_CANDLES:]
                    await self._publish(final_candles)

                # Advance past the newest bar this page produced, whether or not it was
                # stored — otherwise a page of rejected bars would be re-fetched forever.
                newest = max((c.open_time for c in validation.candles), default=None)
                raw_newest = max((c.open_time for c in outcome.candles), default=None)
                advance_from = newest or raw_newest
                cursor = (
                    ensure_utc(advance_from) + timeframe.delta
                    if advance_from is not None
                    and ensure_utc(advance_from) + timeframe.delta > cursor
                    else page_end
                )

            if stop_all:
                break

        if pages >= _MAX_PAGES:
            result.status = IngestStatus.PARTIAL
            result.error = f"stopped after {_MAX_PAGES} pages without reaching the window end"
            log.warning("backfill_page_limit", asset=asset, timeframe=str(timeframe))

        result.covered_start = first_written
        result.covered_end = last_written
        result.source = ", ".join(sorted(served_by)) or source

        # Verify against what is actually stored rather than trusting the loop, over
        # the union of everything that was planned.
        await self._verify_coverage(
            asset,
            timeframe,
            min(a for a, _ in segments),
            max(b for _, b in segments),
            source,
            result,
        )

        if result.status is IngestStatus.SUCCESS and result.rows_written == 0:
            result.status = IngestStatus.SKIPPED

        result.finished_at = utcnow()
        await self._record(result)
        log.info("backfill_finished", summary=result.summary())
        return result

    # ----------------------------------------------------------------- internals

    async def _plan_segments(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None,
        end: datetime | None,
        source: str | None,
        force: bool,
    ) -> tuple[list[tuple[datetime, datetime]], datetime]:
        """Decide which ranges to fetch, in chronological order.

        The end of the overall window is the last *closed* bar: pulling the forming bar
        into history would store an incomplete candle as if it were final.

        When resuming, up to two segments are planned:

        * a **leading** segment, when the configured depth reaches further back than the
          oldest stored bar — without this, raising ``backfill_days`` would appear to do
          nothing, because the series already looks up to date at its front edge;
        * a **trailing** segment, from the newest stored bar to now.

        An explicit ``start`` or ``force`` collapses this to the single requested range.
        """
        now = utcnow()
        window_end = timeframe.floor(ensure_utc(end)) if end else timeframe.floor(now)

        if start is not None:
            requested = timeframe.floor(ensure_utc(start))
            return ([(requested, window_end)] if requested < window_end else []), window_end

        depth_days = self.settings.ingestion.backfill_days.get(timeframe, 30)
        default_start = timeframe.floor(now - timedelta(days=depth_days))

        if force:
            return ([(default_start, window_end)] if default_start < window_end else []), window_end

        async with self.db.session() as session:
            repo = OHLCVRepository(session)
            earliest = await repo.earliest_open_time(asset, timeframe, source)
            latest = await repo.latest_open_time(asset, timeframe, source)

        if earliest is None or latest is None:
            return ([(default_start, window_end)] if default_start < window_end else []), window_end

        segments: list[tuple[datetime, datetime]] = []
        if default_start < earliest:
            segments.append((default_start, earliest))
        tail_start = latest + timeframe.delta
        if tail_start < window_end:
            segments.append((tail_start, window_end))
        return segments, window_end

    def _page_limit(
        self,
        asset: str,
        timeframe: Timeframe,
        cursor: datetime,
        window_end: datetime,
        source: str | None,
    ) -> int:
        """Largest page the chosen provider will serve, capped by config and need.

        Sized against the providers that can actually serve *this* asset: Coinbase's
        300-candle cap must shrink the page when it is the one answering, or every
        request would come back truncated and leave a gap behind.
        """
        configured = self.settings.ingestion.batch_limit
        capable = self.manager.candidates(asset=asset, timeframe=timeframe, preferred=source)
        provider_cap = min(
            (p.capabilities.max_candles_per_request for p in capable if p.capabilities.max_candles_per_request),
            default=configured,
        )
        remaining = max(1, int((window_end - cursor).total_seconds() // timeframe.seconds))
        return max(1, min(configured, provider_cap, remaining))

    async def _load_context(
        self, asset: str, timeframe: Timeframe, before: datetime, source: str | None
    ) -> list[Candle]:
        """Load stored bars preceding the window so outlier stats start warm."""
        async with self.db.session() as session:
            rows = await OHLCVRepository(session).fetch(
                asset,
                timeframe,
                source=source,
                end=before,
                limit=_CONTEXT_CANDLES,
                final_only=True,
            )
        return [
            Candle(
                asset=asset,
                source=source or "stored",
                timeframe=timeframe,
                open_time=row.open_time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                quote_volume=row.quote_volume,
                trades=row.trades,
                is_final=row.is_final,
            )
            for row in rows
        ]

    async def _verify_coverage(
        self,
        asset: str,
        timeframe: Timeframe,
        window_start: datetime,
        window_end: datetime,
        source: str | None,
        result: IngestResult,
    ) -> None:
        """Re-read storage and report any grid positions still missing.

        Checking the database rather than the fetch loop is the point: this catches
        bars that were fetched but rejected, and pages that were skipped entirely.
        """
        async with self.db.session() as session:
            gaps = await OHLCVRepository(session).missing_windows(
                asset, timeframe, window_start, window_end, source
            )
        if not gaps:
            return

        missing = sum(int((b - a).total_seconds() // timeframe.seconds) + 1 for a, b in gaps)
        expected = max(1, int((window_end - window_start).total_seconds() // timeframe.seconds))
        ratio = missing / expected
        severity = (
            QualitySeverity.ERROR
            if ratio >= self.settings.quality.gap_error_ratio
            else QualitySeverity.WARNING
        )
        result.quality_events.append(
            self._event(
                QualityEventType.GAP,
                severity,
                asset,
                timeframe,
                result.source or source or "manager",
                f"{missing}/{expected} candles ({ratio:.1%}) still missing after backfill",
                window_start=gaps[0][0],
                window_end=gaps[-1][1],
                details={
                    "gaps": [
                        {"from": a.isoformat(), "to": b.isoformat()} for a, b in gaps[:20]
                    ],
                    "gap_count": len(gaps),
                },
            )
        )
        if severity is QualitySeverity.ERROR and result.status is IngestStatus.SUCCESS:
            result.status = IngestStatus.PARTIAL

    async def _record(self, result: IngestResult) -> None:
        """Persist provenance and quality events for this run."""
        try:
            async with self.db.session() as session:
                await IngestRunRepository(session).record(result)
                if result.quality_events:
                    await QualityRepository(session).record_events(result.quality_events)
        except Exception as exc:
            # Losing the audit trail must not lose the data that was already written.
            log.error("ingest_run_record_failed", error=str(exc)[:300])

    async def _publish(self, candles: list[Candle]) -> None:
        if self.bus is None:
            return
        await self.bus.publish(
            Event(
                topic=Topics.CANDLE_CLOSED,
                payload=candles,
                meta={
                    "asset": candles[0].asset,
                    "timeframe": str(candles[0].timeframe),
                    "count": len(candles),
                    "origin": "backfill",
                },
            )
        )

    @staticmethod
    def _event(
        event_type: QualityEventType,
        severity: QualitySeverity,
        asset: str,
        timeframe: Timeframe,
        source: str,
        message: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        details: dict | None = None,
    ) -> QualityEvent:
        return QualityEvent(
            event_type=event_type,
            severity=severity,
            source=source,
            asset=asset,
            timeframe=timeframe,
            window_start=window_start,
            window_end=window_end,
            message=message,
            details=details or {},
        )
