"""Repositories — the only code that speaks SQL.

Keeping persistence behind these classes means the ingestion and analytical layers
never build queries, and the dialect-specific upsert lives in exactly one place.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc, grid, utcnow
from mie.core.types import (
    Candle,
    FundingRate,
    GlobalMetricsPoint,
    IngestResult,
    MarketType,
    OpenInterestPoint,
    QualityEvent,
)
from mie.storage.models import (
    OHLCV,
    Asset,
    DataQualityEventRow,
    DataSource,
    FeatureRow,
    FundingRateRow,
    GlobalMetricsRow,
    IngestRunRow,
    Instrument,
    MarketStateRow,
    ModelWeightRow,
    NewsEventRow,
    OpenInterestRow,
    PatternStatsRow,
    PredictionOutcomeRow,
    PredictionRow,
    SourceQualityScore,
)

log = get_logger(__name__)

__all__ = [
    "DerivativesRepository",
    "GlobalMetricsRepository",
    "IngestRunRepository",
    "OHLCVRepository",
    "QualityRepository",
    "ReferenceRepository",
]


#: Every database caps the bound parameters in one statement — SQLite at 32,766 and
#: PostgreSQL at 65,535. A multi-row INSERT spends (rows x columns) of that budget, so
#: a large batch silently becomes "too many SQL variables" at some row count that
#: depends on the table's width. Batching against a conservative budget makes the
#: write size independent of both the caller's batch size and the table's shape.
_MAX_BIND_PARAMS = 20_000


def _chunks(rows: Sequence[dict[str, Any]], columns: int) -> Iterable[Sequence[dict[str, Any]]]:
    """Split rows so no single statement exceeds the parameter budget."""
    size = max(1, _MAX_BIND_PARAMS // max(1, columns))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _upsert(session: AsyncSession, table: Any) -> Any:
    """Return the dialect-appropriate INSERT construct.

    Both SQLite and Postgres implement ``ON CONFLICT DO UPDATE`` with the same
    SQLAlchemy surface, so this is the whole of the portability shim.
    """
    dialect = session.get_bind().dialect.name
    return pg_insert(table) if dialect == "postgresql" else sqlite_insert(table)


class ReferenceRepository:
    """Assets, sources and instruments.

    Instrument lookups happen on every ingest batch, so resolved ids are cached in
    process — the reference tables change on the order of once per deployment.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    _instrument_cache: dict[tuple[str, str, str], int] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._instrument_cache.clear()

    async def ensure_asset(
        self, symbol: str, name: str = "", tier: int = 2, meta: dict[str, Any] | None = None
    ) -> Asset:
        symbol = symbol.upper()
        existing = await self.session.scalar(select(Asset).where(Asset.symbol == symbol))
        if existing:
            # Config is the source of truth for descriptive fields; refresh them.
            if name and existing.name != name:
                existing.name = name
            if existing.tier != tier:
                existing.tier = tier
            return existing
        asset = Asset(symbol=symbol, name=name or symbol, tier=tier, meta=meta or {})
        self.session.add(asset)
        await self.session.flush()
        return asset

    async def ensure_source(
        self, name: str, kind: str = "exchange", priority: int = 100, enabled: bool = True
    ) -> DataSource:
        existing = await self.session.scalar(select(DataSource).where(DataSource.name == name))
        if existing:
            existing.priority = priority
            existing.enabled = enabled
            return existing
        source = DataSource(name=name, kind=kind, priority=priority, enabled=enabled)
        self.session.add(source)
        await self.session.flush()
        return source

    async def ensure_instrument(
        self,
        asset_symbol: str,
        source_name: str,
        provider_symbol: str,
        market_type: MarketType = MarketType.SPOT,
        quote: str = "USDT",
    ) -> int:
        key = (asset_symbol.upper(), source_name, str(market_type))
        cached = self._instrument_cache.get(key)
        if cached is not None:
            return cached

        asset = await self.ensure_asset(asset_symbol)
        source = await self.ensure_source(source_name)
        stmt = select(Instrument).where(
            Instrument.asset_id == asset.id,
            Instrument.source_id == source.id,
            Instrument.market_type == str(market_type),
        )
        instrument = await self.session.scalar(stmt)
        if instrument is None:
            instrument = Instrument(
                asset_id=asset.id,
                source_id=source.id,
                provider_symbol=provider_symbol,
                market_type=str(market_type),
                quote=quote,
            )
            self.session.add(instrument)
            await self.session.flush()
        elif instrument.provider_symbol != provider_symbol:
            instrument.provider_symbol = provider_symbol
            await self.session.flush()

        self._instrument_cache[key] = instrument.id
        return instrument.id

    async def instrument_id(
        self, asset_symbol: str, source_name: str, market_type: MarketType = MarketType.SPOT
    ) -> int | None:
        key = (asset_symbol.upper(), source_name, str(market_type))
        if key in self._instrument_cache:
            return self._instrument_cache[key]
        stmt = (
            select(Instrument.id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .join(DataSource, DataSource.id == Instrument.source_id)
            .where(
                Asset.symbol == asset_symbol.upper(),
                DataSource.name == source_name,
                Instrument.market_type == str(market_type),
            )
        )
        found = await self.session.scalar(stmt)
        if found is not None:
            self._instrument_cache[key] = found
        return found

    async def list_assets(self, active_only: bool = True) -> Sequence[Asset]:
        stmt = select(Asset).order_by(Asset.tier, Asset.symbol)
        if active_only:
            stmt = stmt.where(Asset.is_active.is_(True))
        return (await self.session.scalars(stmt)).all()

    async def list_sources(self) -> Sequence[DataSource]:
        return (await self.session.scalars(select(DataSource).order_by(DataSource.priority))).all()


class OHLCVRepository:
    """Candle persistence and retrieval."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.reference = ReferenceRepository(session)

    async def upsert_candles(self, candles: Iterable[Candle]) -> int:
        """Idempotently write candles. Returns the number of rows sent.

        Re-ingesting the same window is a normal occurrence (polling overlap,
        backfill retries), so writes must be idempotent. A stored bar is overwritten
        only when the incoming one is at least as authoritative: a final candle
        replaces a provisional one, but a provisional candle never overwrites a final
        one — that would reintroduce an unfinished bar the analytics already consumed.
        """
        rows: list[dict[str, Any]] = []
        for candle in candles:
            instrument_id = await self.reference.ensure_instrument(
                candle.asset, candle.source, _default_symbol(candle), candle.market_type, candle.quote
            )
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "timeframe": str(candle.timeframe),
                    "open_time": candle.open_time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "quote_volume": candle.quote_volume,
                    "trades": candle.trades,
                    "is_final": candle.is_final,
                    "revision": 0,
                    "ingested_at": candle.ingested_at,
                }
            )
        if not rows:
            return 0

        for chunk in _chunks(rows, columns=len(rows[0])):
            await self._upsert_ohlcv_chunk(chunk)
        return len(rows)

    async def _upsert_ohlcv_chunk(self, rows: Sequence[dict[str, Any]]) -> None:
        stmt = _upsert(self.session, OHLCV).values(list(rows))
        stmt = stmt.on_conflict_do_update(
            index_elements=[OHLCV.instrument_id, OHLCV.timeframe, OHLCV.open_time],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "quote_volume": stmt.excluded.quote_volume,
                "trades": stmt.excluded.trades,
                "is_final": stmt.excluded.is_final,
                "revision": OHLCV.revision + 1,
                "ingested_at": stmt.excluded.ingested_at,
            },
            where=(OHLCV.is_final.is_(False)) | (stmt.excluded.is_final.is_(True)),
        )
        await self.session.execute(stmt)

    async def fetch(
        self,
        asset: str,
        timeframe: Timeframe,
        source: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        final_only: bool = True,
        market_type: MarketType = MarketType.SPOT,
    ) -> list[OHLCV]:
        """Read candles ascending by time.

        ``final_only`` defaults to True: the forming bar is display-only, and making
        callers opt in to it is the cheap structural guard against look-ahead.
        """
        stmt = (
            select(OHLCV)
            .join(Instrument, Instrument.id == OHLCV.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(
                Asset.symbol == asset.upper(),
                OHLCV.timeframe == str(timeframe),
                Instrument.market_type == str(market_type),
            )
            .order_by(OHLCV.open_time)
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        if start:
            stmt = stmt.where(OHLCV.open_time >= ensure_utc(start))
        if end:
            stmt = stmt.where(OHLCV.open_time < ensure_utc(end))
        if final_only:
            stmt = stmt.where(OHLCV.is_final.is_(True))
        if limit:
            stmt = stmt.limit(limit)
        return list((await self.session.scalars(stmt)).all())

    async def fetch_recent(
        self,
        asset: str,
        timeframe: Timeframe,
        source: str | None = None,
        limit: int = 300,
        final_only: bool = True,
        market_type: MarketType = MarketType.SPOT,
    ) -> list[OHLCV]:
        """The newest ``limit`` bars, returned oldest-first.

        ``fetch`` with a limit takes the *earliest* rows, which is the wrong end for
        warming an indicator. This selects from the recent end and then reverses, so
        the caller can replay the result straight into recursive state.
        """
        stmt = (
            select(OHLCV)
            .join(Instrument, Instrument.id == OHLCV.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(
                Asset.symbol == asset.upper(),
                OHLCV.timeframe == str(timeframe),
                Instrument.market_type == str(market_type),
            )
            .order_by(OHLCV.open_time.desc())
            .limit(limit)
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        if final_only:
            stmt = stmt.where(OHLCV.is_final.is_(True))
        rows = list((await self.session.scalars(stmt)).all())
        rows.reverse()
        return rows

    async def latest_open_time(
        self,
        asset: str,
        timeframe: Timeframe,
        source: str | None = None,
        final_only: bool = True,
    ) -> datetime | None:
        """Newest stored candle — the resume point for incremental backfill."""
        stmt = (
            select(func.max(OHLCV.open_time))
            .select_from(OHLCV)
            .join(Instrument, Instrument.id == OHLCV.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper(), OHLCV.timeframe == str(timeframe))
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        if final_only:
            stmt = stmt.where(OHLCV.is_final.is_(True))
        result = await self.session.scalar(stmt)
        return ensure_utc(result) if result is not None else None

    async def earliest_open_time(
        self, asset: str, timeframe: Timeframe, source: str | None = None
    ) -> datetime | None:
        stmt = (
            select(func.min(OHLCV.open_time))
            .select_from(OHLCV)
            .join(Instrument, Instrument.id == OHLCV.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper(), OHLCV.timeframe == str(timeframe))
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        result = await self.session.scalar(stmt)
        return ensure_utc(result) if result is not None else None

    async def count(
        self, asset: str, timeframe: Timeframe, source: str | None = None
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(OHLCV)
            .join(Instrument, Instrument.id == OHLCV.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper(), OHLCV.timeframe == str(timeframe))
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        return int(await self.session.scalar(stmt) or 0)

    async def count_ingested_since(
        self, asset: str, timeframe: Timeframe, since: datetime, source: str | None = None
    ) -> int:
        """How many candles were *written* since ``since``.

        This is the exposure denominator for quality scoring: it measures how much
        data was actually assessed in the window, which is what makes an event count
        interpretable. Keyed on ``ingested_at``, not ``open_time`` — a backfill of a
        year of history is one recent assessment of 8,760 bars, not a year-old event.
        """
        stmt = (
            select(func.count())
            .select_from(OHLCV)
            .join(Instrument, Instrument.id == OHLCV.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(
                Asset.symbol == asset.upper(),
                OHLCV.timeframe == str(timeframe),
                OHLCV.ingested_at >= ensure_utc(since),
            )
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        return int(await self.session.scalar(stmt) or 0)

    async def missing_windows(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        source: str | None = None,
    ) -> list[tuple[datetime, datetime]]:
        """Contiguous gaps in stored history, as inclusive ``[from, to]`` ranges.

        Compares what is stored against the expected grid rather than trusting row
        counts, so a gap in the middle of a dense range is found as reliably as a
        missing tail.
        """
        stored = {
            row.open_time
            for row in await self.fetch(
                asset, timeframe, source=source, start=start, end=end, final_only=True
            )
        }
        stored = {ensure_utc(ts) for ts in stored}
        gaps: list[tuple[datetime, datetime]] = []
        run_start: datetime | None = None
        previous: datetime | None = None
        for expected in grid(start, end, timeframe):
            if expected in stored:
                if run_start is not None and previous is not None:
                    gaps.append((run_start, previous))
                    run_start = None
            else:
                if run_start is None:
                    run_start = expected
                previous = expected
        if run_start is not None and previous is not None:
            gaps.append((run_start, previous))
        return gaps

    async def coverage(
        self, asset: str, timeframe: Timeframe, source: str | None = None
    ) -> dict[str, Any]:
        """Completeness summary used by ``mie status`` and the quality report."""
        first = await self.earliest_open_time(asset, timeframe, source)
        last = await self.latest_open_time(asset, timeframe, source)
        stored = await self.count(asset, timeframe, source)
        if first is None or last is None:
            return {"asset": asset, "timeframe": str(timeframe), "rows": 0, "completeness": None}
        expected = int((last - first).total_seconds() // timeframe.seconds) + 1
        return {
            "asset": asset,
            "timeframe": str(timeframe),
            "rows": stored,
            "first": first,
            "last": last,
            "expected": expected,
            "completeness": round(stored / expected, 4) if expected else None,
            "age_s": round((utcnow() - last).total_seconds()),
        }

    async def delete_range(
        self, asset: str, timeframe: Timeframe, start: datetime, end: datetime, source: str
    ) -> int:
        instrument_id = await self.reference.instrument_id(asset, source)
        if instrument_id is None:
            return 0
        stmt = delete(OHLCV).where(
            OHLCV.instrument_id == instrument_id,
            OHLCV.timeframe == str(timeframe),
            OHLCV.open_time >= ensure_utc(start),
            OHLCV.open_time < ensure_utc(end),
        )
        result = await self.session.execute(stmt)
        # DELETE returns a CursorResult, which is where rowcount lives; the generic
        # Result protocol does not declare it.
        return int(cast("CursorResult[Any]", result).rowcount or 0)


class DerivativesRepository:
    """Funding rates and open interest."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.reference = ReferenceRepository(session)

    async def upsert_funding(self, points: Iterable[FundingRate]) -> int:
        rows = []
        for point in points:
            instrument_id = await self.reference.ensure_instrument(
                point.asset, point.source, f"{point.asset}USDT", MarketType.PERP, "USDT"
            )
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "ts": point.ts,
                    "rate": point.rate,
                    "interval_hours": point.interval_hours,
                    "mark_price": point.mark_price,
                    "ingested_at": point.ingested_at,
                }
            )
        if not rows:
            return 0
        stmt = _upsert(self.session, FundingRateRow).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[FundingRateRow.instrument_id, FundingRateRow.ts],
            set_={"rate": stmt.excluded.rate, "mark_price": stmt.excluded.mark_price},
        )
        await self.session.execute(stmt)
        return len(rows)

    async def upsert_open_interest(self, points: Iterable[OpenInterestPoint]) -> int:
        rows = []
        for point in points:
            instrument_id = await self.reference.ensure_instrument(
                point.asset, point.source, f"{point.asset}USDT", MarketType.PERP, "USDT"
            )
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "ts": point.ts,
                    "open_interest": point.open_interest,
                    "open_interest_value": point.open_interest_value,
                    "ingested_at": point.ingested_at,
                }
            )
        if not rows:
            return 0
        stmt = _upsert(self.session, OpenInterestRow).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[OpenInterestRow.instrument_id, OpenInterestRow.ts],
            set_={
                "open_interest": stmt.excluded.open_interest,
                "open_interest_value": stmt.excluded.open_interest_value,
            },
        )
        await self.session.execute(stmt)
        return len(rows)

    async def latest_funding(self, asset: str) -> FundingRateRow | None:
        stmt = (
            select(FundingRateRow)
            .join(Instrument, Instrument.id == FundingRateRow.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper())
            .order_by(FundingRateRow.ts.desc())
            .limit(1)
        )
        return await self.session.scalar(stmt)


class GlobalMetricsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.reference = ReferenceRepository(session)

    async def upsert(self, point: GlobalMetricsPoint) -> None:
        source = await self.reference.ensure_source(point.source, kind="aggregator")
        stmt = _upsert(self.session, GlobalMetricsRow).values(
            source_id=source.id,
            ts=point.ts,
            btc_dominance=point.btc_dominance,
            eth_dominance=point.eth_dominance,
            total_market_cap_usd=point.total_market_cap_usd,
            total_volume_24h_usd=point.total_volume_24h_usd,
            stablecoin_share=point.stablecoin_share,
            ingested_at=point.ingested_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[GlobalMetricsRow.source_id, GlobalMetricsRow.ts],
            set_={
                "btc_dominance": stmt.excluded.btc_dominance,
                "eth_dominance": stmt.excluded.eth_dominance,
                "total_market_cap_usd": stmt.excluded.total_market_cap_usd,
                "total_volume_24h_usd": stmt.excluded.total_volume_24h_usd,
                "stablecoin_share": stmt.excluded.stablecoin_share,
            },
        )
        await self.session.execute(stmt)

    async def latest(self) -> GlobalMetricsRow | None:
        stmt = select(GlobalMetricsRow).order_by(GlobalMetricsRow.ts.desc()).limit(1)
        return await self.session.scalar(stmt)


class QualityRepository:
    """Quality events and the rolling trust score derived from them."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_events(self, events: Iterable[QualityEvent]) -> int:
        rows = [
            DataQualityEventRow(
                event_type=str(e.event_type),
                severity=str(e.severity),
                source=e.source,
                asset=e.asset,
                timeframe=str(e.timeframe) if e.timeframe else None,
                window_start=e.window_start,
                window_end=e.window_end,
                message=e.message[:2000],
                details=e.details,
                detected_at=e.detected_at,
            )
            for e in events
        ]
        if not rows:
            return 0
        self.session.add_all(rows)
        await self.session.flush()
        return len(rows)

    async def recent_events(
        self,
        hours: int = 24,
        source: str | None = None,
        asset: str | None = None,
        timeframe: Timeframe | None = None,
        severity: str | None = None,
        limit: int = 200,
    ) -> list[DataQualityEventRow]:
        """Recent events, optionally narrowed to one scope.

        The ``timeframe`` filter matters more than it looks: scoring a series means
        counting the events *for that series*. Without it, a gap on BTC 1h would drag
        down the score of BTC 1m, which is a different feed with different behaviour.
        """
        since = utcnow() - timedelta(hours=hours)
        stmt = (
            select(DataQualityEventRow)
            .where(DataQualityEventRow.detected_at >= since)
            .order_by(DataQualityEventRow.detected_at.desc())
            .limit(limit)
        )
        if source:
            stmt = stmt.where(DataQualityEventRow.source == source)
        if asset:
            stmt = stmt.where(DataQualityEventRow.asset == asset.upper())
        if timeframe:
            stmt = stmt.where(DataQualityEventRow.timeframe == str(timeframe))
        if severity:
            stmt = stmt.where(DataQualityEventRow.severity == severity)
        return list((await self.session.scalars(stmt)).all())

    async def event_counts(self, hours: int = 24) -> dict[str, int]:
        since = utcnow() - timedelta(hours=hours)
        stmt = (
            select(DataQualityEventRow.event_type, func.count())
            .where(DataQualityEventRow.detected_at >= since)
            .group_by(DataQualityEventRow.event_type)
        )
        return {row[0]: int(row[1]) for row in (await self.session.execute(stmt)).all()}

    async def set_score(
        self,
        source: str,
        asset: str,
        timeframe: Timeframe,
        score: float,
        events_in_window: int,
        last_candle_at: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        stmt = _upsert(self.session, SourceQualityScore).values(
            source=source,
            asset=asset.upper(),
            timeframe=str(timeframe),
            score=score,
            events_in_window=events_in_window,
            last_candle_at=last_candle_at,
            details=details or {},
            updated_at=utcnow(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                SourceQualityScore.source,
                SourceQualityScore.asset,
                SourceQualityScore.timeframe,
            ],
            set_={
                "score": stmt.excluded.score,
                "events_in_window": stmt.excluded.events_in_window,
                "last_candle_at": stmt.excluded.last_candle_at,
                "details": stmt.excluded.details,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        await self.session.execute(stmt)

    async def get_score(self, source: str, asset: str, timeframe: Timeframe) -> float:
        """Trust score for a scope. Unknown scopes are optimistic (1.0) by default —
        absence of evidence of a problem, not evidence of a problem."""
        stmt = select(SourceQualityScore.score).where(
            SourceQualityScore.source == source,
            SourceQualityScore.asset == asset.upper(),
            SourceQualityScore.timeframe == str(timeframe),
        )
        score = await self.session.scalar(stmt)
        return float(score) if score is not None else 1.0

    async def all_scores(self) -> list[SourceQualityScore]:
        stmt = select(SourceQualityScore).order_by(SourceQualityScore.score)
        return list((await self.session.scalars(stmt)).all())


class IngestRunRepository:
    """Append-only provenance log for ingest jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(self, result: IngestResult) -> int:
        row = IngestRunRow(
            job=result.job,
            asset=result.asset,
            timeframe=str(result.timeframe) if result.timeframe else None,
            source=result.source,
            status=str(result.status),
            requested_start=result.requested_start,
            requested_end=result.requested_end,
            covered_start=result.covered_start,
            covered_end=result.covered_end,
            rows_fetched=result.rows_fetched,
            rows_written=result.rows_written,
            rows_rejected=result.rows_rejected,
            quality_event_count=len(result.quality_events),
            error=result.error[:2000] if result.error else None,
            started_at=result.started_at,
            finished_at=result.finished_at or utcnow(),
            duration_s=result.duration_s,
        )
        self.session.add(row)
        await self.session.flush()
        return row.id

    async def recent(self, limit: int = 50, job: str | None = None) -> list[IngestRunRow]:
        stmt = select(IngestRunRow).order_by(IngestRunRow.started_at.desc()).limit(limit)
        if job:
            stmt = stmt.where(IngestRunRow.job == job)
        return list((await self.session.scalars(stmt)).all())


def _default_symbol(candle: Candle) -> str:
    """Fallback provider symbol when a candle arrives without instrument context.

    Real ingestion paths register the instrument explicitly with the provider's own
    symbol; this only covers direct writes (tests, imports) so they cannot create an
    instrument row with an empty symbol.
    """
    return f"{candle.asset}{candle.quote}"


class FeatureRepository:
    """Computed feature vectors."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.reference = ReferenceRepository(session)

    async def upsert(
        self,
        asset: str,
        source: str,
        market_type: MarketType,
        timeframe: Timeframe,
        open_time: datetime,
        values: dict[str, Any],
        version: int = 1,
    ) -> None:
        await self.upsert_many(
            asset=asset,
            source=source,
            market_type=market_type,
            timeframe=timeframe,
            rows=[{"open_time": open_time, "values": values}],
            version=version,
        )

    async def upsert_many(
        self,
        asset: str,
        source: str,
        market_type: MarketType,
        timeframe: Timeframe,
        rows: Sequence[dict[str, Any]],
        version: int = 1,
    ) -> int:
        """Idempotently write feature vectors for one series.

        Recomputation is expected — a corrected bar, a new feature definition — so a
        rewrite overwrites in place and carries the version that produced it.
        """
        if not rows:
            return 0
        instrument_id = await self.reference.ensure_instrument(
            asset, source, f"{asset.upper()}USDT", market_type
        )
        now = utcnow()
        payload = [
            {
                "instrument_id": instrument_id,
                "timeframe": str(timeframe),
                "open_time": ensure_utc(row["open_time"]),
                "version": version,
                "payload": row["values"],
                "computed_at": now,
            }
            for row in rows
        ]
        for chunk in _chunks(payload, columns=len(payload[0])):
            await self._upsert_feature_chunk(chunk)
        return len(payload)

    async def _upsert_feature_chunk(self, rows: Sequence[dict[str, Any]]) -> None:
        stmt = _upsert(self.session, FeatureRow).values(list(rows))
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                FeatureRow.instrument_id,
                FeatureRow.timeframe,
                FeatureRow.open_time,
            ],
            set_={
                "payload": stmt.excluded.payload,
                "version": stmt.excluded.version,
                "computed_at": stmt.excluded.computed_at,
            },
        )
        await self.session.execute(stmt)

    async def fetch(
        self,
        asset: str,
        timeframe: Timeframe,
        source: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[FeatureRow]:
        stmt = (
            select(FeatureRow)
            .join(Instrument, Instrument.id == FeatureRow.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper(), FeatureRow.timeframe == str(timeframe))
            .order_by(FeatureRow.open_time)
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        if start:
            stmt = stmt.where(FeatureRow.open_time >= ensure_utc(start))
        if end:
            stmt = stmt.where(FeatureRow.open_time < ensure_utc(end))
        if limit:
            stmt = stmt.limit(limit)
        return list((await self.session.scalars(stmt)).all())

    async def latest(
        self, asset: str, timeframe: Timeframe, source: str | None = None
    ) -> FeatureRow | None:
        """Most recent feature vector — what a live consumer asks for."""
        stmt = (
            select(FeatureRow)
            .join(Instrument, Instrument.id == FeatureRow.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper(), FeatureRow.timeframe == str(timeframe))
            .order_by(FeatureRow.open_time.desc())
            .limit(1)
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        return await self.session.scalar(stmt)

    async def count(self, asset: str, timeframe: Timeframe, source: str | None = None) -> int:
        stmt = (
            select(func.count())
            .select_from(FeatureRow)
            .join(Instrument, Instrument.id == FeatureRow.instrument_id)
            .join(Asset, Asset.id == Instrument.asset_id)
            .where(Asset.symbol == asset.upper(), FeatureRow.timeframe == str(timeframe))
        )
        if source:
            stmt = stmt.join(DataSource, DataSource.id == Instrument.source_id).where(
                DataSource.name == source
            )
        return int(await self.session.scalar(stmt) or 0)


class MarketStateRepository:
    """Persisted multi-timeframe market state."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, state: Any) -> None:
        """Store one state. Recomputing the same moment overwrites in place."""
        stmt = _upsert(self.session, MarketStateRow).values(
            asset=state.asset,
            as_of=ensure_utc(state.as_of),
            bias=str(state.bias),
            alignment=str(state.alignment),
            regime=str(state.regime),
            agreement=state.agreement,
            confidence=state.confidence,
            data_quality=state.data_quality,
            interpretation=state.interpretation,
            levels={
                str(level.timeframe): {
                    "direction": str(level.direction),
                    "strength": level.strength,
                    "confidence": level.confidence,
                    "score": level.score,
                    "as_of": level.as_of.isoformat(),
                    "close": level.close,
                    "volatility_pct": level.volatility_pct,
                    "evidence": [e.model_dump() for e in level.evidence],
                    "counter_evidence": [e.model_dump() for e in level.counter_evidence],
                }
                for level in state.timeframes
            },
            evidence={
                "supporting": [e.model_dump() for e in state.evidence],
                "conflicts": state.conflicts,
                "details": state.details,
            },
            computed_at=utcnow(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[MarketStateRow.asset, MarketStateRow.as_of],
            set_={
                "bias": stmt.excluded.bias,
                "alignment": stmt.excluded.alignment,
                "regime": stmt.excluded.regime,
                "agreement": stmt.excluded.agreement,
                "confidence": stmt.excluded.confidence,
                "data_quality": stmt.excluded.data_quality,
                "interpretation": stmt.excluded.interpretation,
                "levels": stmt.excluded.levels,
                "evidence": stmt.excluded.evidence,
                "computed_at": stmt.excluded.computed_at,
            },
        )
        await self.session.execute(stmt)

    async def latest(self, asset: str) -> MarketStateRow | None:
        stmt = (
            select(MarketStateRow)
            .where(MarketStateRow.asset == asset.upper())
            .order_by(MarketStateRow.as_of.desc())
            .limit(1)
        )
        return await self.session.scalar(stmt)

    async def history(
        self,
        asset: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[MarketStateRow]:
        stmt = (
            select(MarketStateRow)
            .where(MarketStateRow.asset == asset.upper())
            .order_by(MarketStateRow.as_of)
            .limit(limit)
        )
        if start:
            stmt = stmt.where(MarketStateRow.as_of >= ensure_utc(start))
        if end:
            stmt = stmt.where(MarketStateRow.as_of < ensure_utc(end))
        return list((await self.session.scalars(stmt)).all())

    async def regime_counts(self, hours: int = 168) -> dict[str, int]:
        """How much time each regime has accounted for recently."""
        since = utcnow() - timedelta(hours=hours)
        stmt = (
            select(MarketStateRow.regime, func.count())
            .where(MarketStateRow.as_of >= since)
            .group_by(MarketStateRow.regime)
        )
        return {row[0]: int(row[1]) for row in (await self.session.execute(stmt)).all()}


class PatternStatsRepository:
    """The evidence base behind the Phase 4 pattern gate."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_many(self, stats: Sequence[Any]) -> int:
        """Store measured pattern statistics, replacing any prior measurement.

        Re-measurement is expected — more history, or a corrected detector — and the
        latest measurement always wins. Keeping stale statistics would let a finding
        that a later fix invalidated go on influencing predictions.
        """
        if not stats:
            return 0
        now = utcnow()
        rows = [
            {
                "kind": str(s.kind),
                "asset": s.asset.upper(),
                "timeframe": str(s.timeframe),
                "horizon_bars": s.horizon_bars,
                "direction": str(s.direction),
                "occurrences": s.occurrences,
                "rate": s.estimate.rate,
                "interval_low": s.estimate.low,
                "interval_high": s.estimate.high,
                "baseline": s.estimate.baseline,
                "edge": s.estimate.edge,
                "p_value": s.estimate.p_value,
                "significant": s.estimate.significant,
                "informative": s.is_informative,
                "verdict": s.verdict,
                "mean_return_pct": s.mean_return_pct,
                "median_return_pct": s.median_return_pct,
                "mean_favourable_pct": s.mean_favourable_pct,
                "mean_adverse_pct": s.mean_adverse_pct,
                "sample_start": s.sample_start,
                "sample_end": s.sample_end,
                "computed_at": now,
            }
            for s in stats
        ]
        updatable = (
            "direction", "occurrences", "rate", "interval_low", "interval_high",
            "baseline", "edge", "p_value", "significant", "informative", "verdict",
            "mean_return_pct", "median_return_pct", "mean_favourable_pct",
            "mean_adverse_pct", "sample_start", "sample_end", "computed_at",
        )
        for chunk in _chunks(rows, columns=len(rows[0])):
            stmt = _upsert(self.session, PatternStatsRow).values(list(chunk))
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    PatternStatsRow.kind,
                    PatternStatsRow.asset,
                    PatternStatsRow.timeframe,
                    PatternStatsRow.horizon_bars,
                ],
                set_={column: getattr(stmt.excluded, column) for column in updatable},
            )
            await self.session.execute(stmt)
        return len(rows)

    async def all_stats(self, informative_only: bool = False) -> list[PatternStatsRow]:
        stmt = select(PatternStatsRow).order_by(PatternStatsRow.p_value)
        if informative_only:
            stmt = stmt.where(PatternStatsRow.informative.is_(True))
        return list((await self.session.scalars(stmt)).all())

    async def for_asset(
        self, asset: str, timeframe: Timeframe | None = None
    ) -> list[PatternStatsRow]:
        stmt = select(PatternStatsRow).where(PatternStatsRow.asset == asset.upper())
        if timeframe:
            stmt = stmt.where(PatternStatsRow.timeframe == str(timeframe))
        return list((await self.session.scalars(stmt.order_by(PatternStatsRow.p_value))).all())


class NewsEventRepository:
    """Persisted news stories, so history accumulates across fetches."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_many(self, events: Sequence[Any]) -> int:
        """Store events, updating coverage for stories already seen.

        Re-fetching is the normal case: a story stays in the feed for days while more
        outlets pick it up. The update path keeps the *earliest* publication time —
        when the story broke is what matters for pairing it with price action, and
        letting a later re-run overwrite it would move the event forward in time and
        quietly invalidate every impact measurement that used it.
        """
        if not events:
            return 0
        now = utcnow()
        rows = [
            {
                "cluster_id": e.cluster_id,
                "title": e.title[:2000],
                "url": e.url[:2000],
                "published_at": ensure_utc(e.published_at),
                "sources": list(e.sources),
                "category": str(e.category),
                "sentiment": str(e.sentiment),
                "sentiment_score": e.sentiment_score,
                "relevance": {r.asset: r.score for r in e.relevance},
                "importance": e.importance,
                "confidence": e.confidence,
                "coverage": e.coverage,
                "article_count": e.article_count,
                "is_recycled": e.is_recycled,
                "recycled_from": e.recycled_from,
                "first_seen_at": now,
                "updated_at": now,
            }
            for e in events
        ]
        for chunk in _chunks(rows, columns=len(rows[0])):
            stmt = _upsert(self.session, NewsEventRow).values(list(chunk))
            stmt = stmt.on_conflict_do_update(
                index_elements=[NewsEventRow.cluster_id],
                set_={
                    # published_at is deliberately absent: the break time is fixed.
                    "title": stmt.excluded.title,
                    "sources": stmt.excluded.sources,
                    "coverage": stmt.excluded.coverage,
                    "article_count": stmt.excluded.article_count,
                    "importance": stmt.excluded.importance,
                    "sentiment": stmt.excluded.sentiment,
                    "sentiment_score": stmt.excluded.sentiment_score,
                    "relevance": stmt.excluded.relevance,
                    "confidence": stmt.excluded.confidence,
                    "is_recycled": stmt.excluded.is_recycled,
                    "recycled_from": stmt.excluded.recycled_from,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            await self.session.execute(stmt)
        return len(rows)

    async def recent(
        self,
        hours: int = 168,
        category: str | None = None,
        asset: str | None = None,
        exclude_recycled: bool = True,
        limit: int = 1000,
    ) -> list[NewsEventRow]:
        since = utcnow() - timedelta(hours=hours)
        stmt = (
            select(NewsEventRow)
            .where(NewsEventRow.published_at >= since)
            .order_by(NewsEventRow.published_at.desc())
            .limit(limit)
        )
        if category:
            stmt = stmt.where(NewsEventRow.category == category)
        if exclude_recycled:
            stmt = stmt.where(NewsEventRow.is_recycled.is_(False))
        rows = list((await self.session.scalars(stmt)).all())
        if asset:
            key = asset.upper()
            rows = [r for r in rows if (r.relevance or {}).get(key, 0.0) >= 0.5]
        return rows

    async def all_events(
        self, exclude_recycled: bool = True, limit: int = 10_000
    ) -> list[NewsEventRow]:
        """Every stored story, oldest first — the sample impact validation runs on."""
        stmt = select(NewsEventRow).order_by(NewsEventRow.published_at).limit(limit)
        if exclude_recycled:
            stmt = stmt.where(NewsEventRow.is_recycled.is_(False))
        return list((await self.session.scalars(stmt)).all())

    async def count(self) -> int:
        return int(await self.session.scalar(select(func.count()).select_from(NewsEventRow)) or 0)

    async def category_counts(self, hours: int = 168) -> dict[str, int]:
        since = utcnow() - timedelta(hours=hours)
        stmt = (
            select(NewsEventRow.category, func.count())
            .where(NewsEventRow.published_at >= since)
            .group_by(NewsEventRow.category)
        )
        return {row[0]: int(row[1]) for row in (await self.session.execute(stmt)).all()}


class PredictionRepository:
    """Append-only prediction storage, with outcomes and weights alongside.

    There is no update path for a prediction, and that is the point. The insert uses
    ``ON CONFLICT DO NOTHING``: re-running the same prediction point collides on its
    derived id and is dropped, so a re-run can neither duplicate the sample nor revise
    what was said. The only field ever mutated on a prediction row is ``resolved``,
    which carries no information about the outcome.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def append(self, records: Sequence[Any]) -> int:
        """Store predictions, ignoring any that already exist. Returns rows offered."""
        if not records:
            return 0
        rows = [
            {
                "prediction_id": r.prediction_id,
                "content_hash": r.content_hash,
                "model_id": r.model_id,
                "model_version": r.model_version,
                "asset": r.asset.upper(),
                "timeframe": str(r.timeframe),
                "horizon_bars": r.horizon_bars,
                "as_of": ensure_utc(r.as_of),
                "resolves_at": ensure_utc(r.resolves_at),
                "prob_up": r.distribution.up,
                "prob_flat": r.distribution.flat,
                "prob_down": r.distribution.down,
                "confidence": r.confidence,
                "move_threshold_pct": r.move_threshold_pct,
                "reference_price": r.reference_price,
                "regime": r.regime,
                "volatility_bucket": r.volatility_bucket,
                "data_quality": r.data_quality,
                "is_actionable": r.is_actionable,
                "evidence": r.evidence,
                "resolved": False,
                "created_at": ensure_utc(r.created_at),
            }
            for r in records
        ]
        for chunk in _chunks(rows, columns=len(rows[0])):
            stmt = _upsert(self.session, PredictionRow).values(list(chunk))
            await self.session.execute(
                stmt.on_conflict_do_nothing(index_elements=[PredictionRow.prediction_id])
            )
        return len(rows)

    async def due(self, now: datetime | None = None, limit: int = 5000) -> list[PredictionRow]:
        """Unresolved predictions whose horizon has elapsed."""
        moment = now or utcnow()
        stmt = (
            select(PredictionRow)
            .where(PredictionRow.resolved.is_(False), PredictionRow.resolves_at <= moment)
            .order_by(PredictionRow.resolves_at)
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def unresolved(self, limit: int = 20000) -> list[PredictionRow]:
        stmt = (
            select(PredictionRow)
            .where(PredictionRow.resolved.is_(False))
            .order_by(PredictionRow.as_of)
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def records(self, limit: int = 100000) -> list[PredictionRow]:
        """Every stored prediction, resolved or not.

        Recalibration needs the *resolved* ones: a curve is fitted from what a model
        said paired with what happened, and the prediction row is the only place the
        distribution it said is kept. Loading only unresolved rows — the obvious thing,
        since those are what the resolver wants — leaves the calibrator with nothing
        the moment the backlog clears, and it fails silently rather than loudly.
        """
        stmt = select(PredictionRow).order_by(PredictionRow.as_of).limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())

    async def mark_resolved(self, prediction_ids: Sequence[str]) -> int:
        if not prediction_ids:
            return 0
        total = 0
        for start in range(0, len(prediction_ids), 500):
            chunk = list(prediction_ids[start : start + 500])
            result = cast(
                CursorResult,
                await self.session.execute(
                    update(PredictionRow)
                    .where(PredictionRow.prediction_id.in_(chunk))
                    .values(resolved=True)
                ),
            )
            total += result.rowcount or 0
        return total

    async def record_outcomes(self, outcomes: Sequence[Any]) -> int:
        """Store resolved outcomes. Also append-only: an outcome is a fact."""
        if not outcomes:
            return 0
        rows = [
            {
                "prediction_id": o.prediction_id,
                "model_id": o.model_id,
                "asset": o.asset.upper(),
                "timeframe": str(o.timeframe),
                "horizon_bars": o.horizon_bars,
                "regime": o.regime,
                "volatility_bucket": o.volatility_bucket,
                "resolved_at": ensure_utc(o.resolved_at),
                "realised_direction": str(o.realised_direction),
                "realised_move_pct": o.realised_move_pct,
                "exit_price": o.exit_price,
                "brier": o.brier,
                "log_loss": o.log_loss,
                "correct": o.correct,
                "probability_of_truth": o.probability_of_truth,
                "scored_at": ensure_utc(o.scored_at),
            }
            for o in outcomes
        ]
        for chunk in _chunks(rows, columns=len(rows[0])):
            stmt = _upsert(self.session, PredictionOutcomeRow).values(list(chunk))
            await self.session.execute(
                stmt.on_conflict_do_nothing(
                    index_elements=[PredictionOutcomeRow.prediction_id]
                )
            )
        await self.mark_resolved([o.prediction_id for o in outcomes])
        return len(rows)

    async def outcomes(
        self,
        asset: str | None = None,
        model_id: str | None = None,
        since: datetime | None = None,
        limit: int = 50000,
    ) -> list[PredictionOutcomeRow]:
        stmt = select(PredictionOutcomeRow).order_by(PredictionOutcomeRow.resolved_at)
        if asset:
            stmt = stmt.where(PredictionOutcomeRow.asset == asset.upper())
        if model_id:
            stmt = stmt.where(PredictionOutcomeRow.model_id == model_id)
        if since:
            stmt = stmt.where(PredictionOutcomeRow.resolved_at >= ensure_utc(since))
        return list((await self.session.execute(stmt.limit(limit))).scalars().all())

    async def upsert_weights(self, updates: Sequence[Any]) -> int:
        """Store the current weight per scope, keeping the previous value visible."""
        if not updates:
            return 0
        now = utcnow()
        rows = [
            {
                "model_id": u.key.model_id,
                "asset": u.key.asset.upper(),
                "timeframe": u.key.timeframe,
                "horizon_bars": u.key.horizon_bars,
                "regime": u.key.regime,
                "raw_skill": u.raw_skill,
                "weight": u.weight,
                "previous_weight": u.previous_weight,
                "samples": u.samples,
                "p_value": u.p_value,
                "significant": u.significant,
                "updated_at": now,
            }
            for u in updates
        ]
        for chunk in _chunks(rows, columns=len(rows[0])):
            stmt = _upsert(self.session, ModelWeightRow).values(list(chunk))
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    ModelWeightRow.model_id,
                    ModelWeightRow.asset,
                    ModelWeightRow.timeframe,
                    ModelWeightRow.horizon_bars,
                    ModelWeightRow.regime,
                ],
                set_={
                    "raw_skill": stmt.excluded.raw_skill,
                    "weight": stmt.excluded.weight,
                    "previous_weight": stmt.excluded.previous_weight,
                    "samples": stmt.excluded.samples,
                    "p_value": stmt.excluded.p_value,
                    "significant": stmt.excluded.significant,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            await self.session.execute(stmt)
        return len(rows)

    async def weights(self) -> list[ModelWeightRow]:
        stmt = select(ModelWeightRow).order_by(ModelWeightRow.weight.desc())
        return list((await self.session.execute(stmt)).scalars().all())

    async def counts(self) -> dict[str, int]:
        """How many predictions are stored, resolved and pending."""
        total = await self.session.scalar(select(func.count()).select_from(PredictionRow))
        resolved = await self.session.scalar(
            select(func.count()).select_from(PredictionRow).where(PredictionRow.resolved.is_(True))
        )
        outcomes = await self.session.scalar(
            select(func.count()).select_from(PredictionOutcomeRow)
        )
        return {
            "predictions": int(total or 0),
            "resolved": int(resolved or 0),
            "pending": int(total or 0) - int(resolved or 0),
            "outcomes": int(outcomes or 0),
        }
