"""Relational schema.

Written against SQLAlchemy 2.0 typed declarative so the ORM layer and the type
checker agree. The design targets TimescaleDB (see ``sql/timescale.sql``) while
remaining valid on SQLite, which is what makes the test suite infrastructure-free.

Two deliberate choices worth stating:

* **Assets are decoupled from exchange symbols.** ``assets`` holds the canonical
  identity (``BTC``); ``instruments`` maps it to ``BTCUSDT`` on Binance spot and
  ``XBTUSD`` on Kraken. Without that indirection, multi-source failover and
  cross-source comparison both become string-munging at query time.
* **Prices are stored as ``Float`` (IEEE double), not ``Numeric``.** This is an
  analytical system, not a ledger: doubles carry ~15 significant digits, far beyond
  any exchange's tick precision, and every downstream consumer is floating-point
  anyway. A settlement system would need the opposite choice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from mie.core.timeframes import UTC, utcnow

__all__ = [
    "OHLCV",
    "Asset",
    "Base",
    "DataQualityEventRow",
    "DataSource",
    "FundingRateRow",
    "GlobalMetricsRow",
    "IngestRunRow",
    "Instrument",
    "OpenInterestRow",
    "SourceQualityScore",
    "UTCDateTime",
]


class UTCDateTime(TypeDecorator):
    """Timestamps that survive a round-trip through SQLite.

    PostgreSQL stores ``timestamptz`` natively, but SQLite has no timezone type: the
    dialect silently drops ``tzinfo`` on write and returns naive datetimes on read.
    Mixing naive and aware datetimes across a storage boundary is a classic source of
    off-by-hours bugs, so this decorator normalises to UTC on the way in and
    re-attaches UTC on the way out. Naive input is rejected rather than assumed.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime {value!r} cannot be persisted; use tz-aware UTC")
        return value.astimezone(UTC)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base. ``JSON`` maps to ``jsonb`` on Postgres via the dialect."""

    type_annotation_map = {dict[str, Any]: JSON}


def _ts_column(**kwargs: Any) -> Mapped[datetime]:
    """Timezone-aware timestamp column. Every time column in the schema uses this."""
    return mapped_column(UTCDateTime, **kwargs)


class Asset(Base):
    """Canonical asset identity, independent of any venue."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    tier: Mapped[int] = mapped_column(Integer, default=2)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts_column(default=utcnow)

    instruments: Mapped[list[Instrument]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Asset {self.symbol}>"


class DataSource(Base):
    """A data provider. ``priority`` drives failover order (lower wins)."""

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="exchange")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts_column(default=utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DataSource {self.name}>"


class Instrument(Base):
    """(asset, source, market_type) → the provider's own symbol."""

    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("asset_id", "source_id", "market_type", name="uq_instrument"),
        Index("ix_instrument_lookup", "source_id", "asset_id", "market_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), index=True
    )
    provider_symbol: Mapped[str] = mapped_column(String(64))
    market_type: Mapped[str] = mapped_column(String(16), default="spot")
    quote: Mapped[str] = mapped_column(String(16), default="USDT")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = _ts_column(default=utcnow)

    asset: Mapped[Asset] = relationship(back_populates="instruments")
    source: Mapped[DataSource] = relationship()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Instrument {self.provider_symbol}@{self.source_id}>"


class OHLCV(Base):
    """The central time-series table.

    Composite primary key ``(instrument_id, timeframe, open_time)`` gives idempotent
    upserts for free and includes the partitioning column, which TimescaleDB requires
    of a hypertable's unique constraints.
    """

    __tablename__ = "ohlcv"
    __table_args__ = (
        Index("ix_ohlcv_asset_tf_time", "instrument_id", "timeframe", "open_time"),
        Index("ix_ohlcv_time", "open_time"),
        Index("ix_ohlcv_final", "instrument_id", "timeframe", "is_final"),
    )

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    timeframe: Mapped[str] = mapped_column(String(8), primary_key=True)
    open_time: Mapped[datetime] = _ts_column(primary_key=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    quote_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    trades: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # False while the bar is still forming. Downstream analytics must filter on this.
    is_final: Mapped[bool] = mapped_column(Boolean, default=True)
    # Bumped whenever a stored bar is rewritten, so revisions are visible rather than
    # silently overwriting history.
    revision: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = _ts_column(default=utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OHLCV {self.instrument_id} {self.timeframe} {self.open_time} c={self.close}>"


class FundingRateRow(Base):
    """Perp funding history — leveraged positioning pressure."""

    __tablename__ = "funding_rates"
    __table_args__ = (Index("ix_funding_instrument_ts", "instrument_id", "ts"),)

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[datetime] = _ts_column(primary_key=True)
    rate: Mapped[float] = mapped_column(Float)
    interval_hours: Mapped[float] = mapped_column(Float, default=8.0)
    mark_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ingested_at: Mapped[datetime] = _ts_column(default=utcnow)


class OpenInterestRow(Base):
    """Open interest history — size of outstanding leverage."""

    __tablename__ = "open_interest"
    __table_args__ = (Index("ix_oi_instrument_ts", "instrument_id", "ts"),)

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[datetime] = _ts_column(primary_key=True)
    open_interest: Mapped[float] = mapped_column(Float)
    open_interest_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    ingested_at: Mapped[datetime] = _ts_column(default=utcnow)


class GlobalMetricsRow(Base):
    """Market-wide context: dominance and aggregate capital."""

    __tablename__ = "global_metrics"
    __table_args__ = (Index("ix_global_ts", "ts"),)

    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[datetime] = _ts_column(primary_key=True)
    btc_dominance: Mapped[float | None] = mapped_column(Float, nullable=True)
    eth_dominance: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_market_cap_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_volume_24h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    stablecoin_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    ingested_at: Mapped[datetime] = _ts_column(default=utcnow)


class DataQualityEventRow(Base):
    """Every defect the validation layer found, kept for audit and scoring.

    Stored as free-text source/asset rather than FKs on purpose: a quality event can
    concern a source or asset the system has not (or can no longer) resolve to a row,
    and losing the event because of a broken reference would defeat its purpose.
    """

    __tablename__ = "data_quality_events"
    __table_args__ = (
        Index("ix_quality_scope_time", "source", "asset", "timeframe", "detected_at"),
        Index("ix_quality_severity_time", "severity", "detected_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(64))
    asset: Mapped[str | None] = mapped_column(String(32), nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(8), nullable=True)
    window_start: Mapped[datetime | None] = _ts_column(nullable=True)
    window_end: Mapped[datetime | None] = _ts_column(nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    detected_at: Mapped[datetime] = _ts_column(default=utcnow, index=True)
    resolved_at: Mapped[datetime | None] = _ts_column(nullable=True)


class SourceQualityScore(Base):
    """Rolling trust score in [0, 1] per (source, asset, timeframe).

    This is the number that later phases multiply into published confidence, which is
    how requirement §20 — degrade rather than pretend — is actually enforced.
    """

    __tablename__ = "source_quality_scores"
    __table_args__ = (UniqueConstraint("source", "asset", "timeframe", name="uq_quality_scope"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    asset: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    score: Mapped[float] = mapped_column(Float, default=1.0)
    events_in_window: Mapped[int] = mapped_column(Integer, default=0)
    last_candle_at: Mapped[datetime | None] = _ts_column(nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = _ts_column(default=utcnow)


class IngestRunRow(Base):
    """Provenance for every ingest job: what, from where, when, and how it went."""

    __tablename__ = "ingest_runs"
    __table_args__ = (Index("ix_ingest_scope_time", "asset", "timeframe", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job: Mapped[str] = mapped_column(String(32), index=True)
    asset: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str | None] = mapped_column(String(8), nullable=True)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    requested_start: Mapped[datetime | None] = _ts_column(nullable=True)
    requested_end: Mapped[datetime | None] = _ts_column(nullable=True)
    covered_start: Mapped[datetime | None] = _ts_column(nullable=True)
    covered_end: Mapped[datetime | None] = _ts_column(nullable=True)
    rows_fetched: Mapped[int] = mapped_column(Integer, default=0)
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    quality_event_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = _ts_column(default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = _ts_column(nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
