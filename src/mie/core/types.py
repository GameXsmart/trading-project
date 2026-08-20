"""Canonical domain types.

These are the contracts every layer agrees on. Providers translate their private
wire formats into these; storage persists them; later phases consume them. Nothing
downstream is allowed to depend on a provider's native shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mie.core.timeframes import Timeframe, ensure_utc, utcnow

__all__ = [
    "Candle",
    "FundingRate",
    "GlobalMetricsPoint",
    "IngestResult",
    "IngestStatus",
    "MarketType",
    "OpenInterestPoint",
    "ProviderHealth",
    "QualityEvent",
    "QualityEventType",
    "QualitySeverity",
]


class MarketType(StrEnum):
    SPOT = "spot"
    PERP = "perp"
    FUTURES = "futures"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Candle(_Frozen):
    """One OHLCV bar, fully attributed to its source.

    ``is_final`` is load-bearing: the candle covering the current wall-clock time is
    still forming, and treating it as complete is look-ahead bias in live operation.
    Only ``is_final`` candles may reach the feature engine.
    """

    asset: str
    quote: str = "USD"
    source: str
    market_type: MarketType = MarketType.SPOT
    timeframe: Timeframe
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    trades: int | None = None
    is_final: bool = True
    ingested_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        # frozen model: mutate through __dict__ during validation only.
        object.__setattr__(self, "asset", self.asset.upper())
        object.__setattr__(self, "quote", self.quote.upper())
        object.__setattr__(self, "open_time", ensure_utc(self.open_time))
        object.__setattr__(self, "ingested_at", ensure_utc(self.ingested_at))
        return self

    @property
    def close_time(self) -> datetime:
        return self.timeframe.close_time(self.open_time)

    @property
    def range_pct(self) -> float:
        return 0.0 if self.low <= 0 else (self.high - self.low) / self.low * 100.0

    @property
    def change_pct(self) -> float:
        return 0.0 if self.open <= 0 else (self.close - self.open) / self.open * 100.0

    @property
    def key(self) -> tuple[str, str, str, str, datetime]:
        """Natural identity of a bar: who, what, which grid, and when."""
        return (self.source, self.asset, str(self.market_type), str(self.timeframe), self.open_time)


class FundingRate(_Frozen):
    """Perpetual-swap funding — a direct read on leveraged positioning."""

    asset: str
    source: str
    ts: datetime
    rate: float
    interval_hours: float = 8.0
    mark_price: float | None = None
    ingested_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        object.__setattr__(self, "asset", self.asset.upper())
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        return self

    @property
    def annualised_pct(self) -> float:
        periods_per_year = (365.0 * 24.0) / self.interval_hours
        return self.rate * periods_per_year * 100.0


class OpenInterestPoint(_Frozen):
    asset: str
    source: str
    ts: datetime
    open_interest: float
    open_interest_value: float | None = None
    ingested_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        object.__setattr__(self, "asset", self.asset.upper())
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        return self


class GlobalMetricsPoint(_Frozen):
    """Market-wide context: dominance and aggregate flows."""

    source: str
    ts: datetime
    btc_dominance: float | None = None
    eth_dominance: float | None = None
    total_market_cap_usd: float | None = None
    total_volume_24h_usd: float | None = None
    stablecoin_share: float | None = None
    ingested_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        return self


class QualitySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def weight(self) -> float:
        """How much one event of this severity depresses the rolling quality score."""
        return {"info": 0.0, "warning": 0.15, "error": 0.5}[self.value]


class QualityEventType(StrEnum):
    SHAPE_INVALID = "shape_invalid"
    GRID_MISALIGNED = "grid_misaligned"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    GAP = "gap"
    OUTLIER = "outlier"
    IMPOSSIBLE_MOVE = "impossible_move"
    STALE_FEED = "stale_feed"
    FLATLINE = "flatline"
    SOURCE_DISCREPANCY = "source_discrepancy"
    PROVIDER_FAILOVER = "provider_failover"
    PROVIDER_ERROR = "provider_error"
    EMPTY_RESPONSE = "empty_response"


class QualityEvent(BaseModel):
    """A recorded defect in incoming data.

    Quality events are never fatal on their own — they accumulate into a score that
    reduces downstream confidence. Losing them silently would defeat the point.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: QualityEventType
    severity: QualitySeverity
    source: str
    asset: str | None = None
    timeframe: Timeframe | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=utcnow)

    def __str__(self) -> str:  # pragma: no cover - logging affordance
        scope = "/".join(str(p) for p in (self.source, self.asset, self.timeframe) if p)
        return f"[{self.severity}] {self.event_type} {scope}: {self.message}"


class ProviderHealth(_Frozen):
    provider: str
    ok: bool
    checked_at: datetime = Field(default_factory=utcnow)
    latency_ms: float | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class IngestStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class IngestResult(BaseModel):
    """Outcome of one ingest job, persisted for provenance and operator debugging."""

    model_config = ConfigDict(extra="forbid")

    job: str
    asset: str
    timeframe: Timeframe | None = None
    source: str | None = None
    status: IngestStatus = IngestStatus.SUCCESS
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    covered_start: datetime | None = None
    covered_end: datetime | None = None
    rows_fetched: int = 0
    rows_written: int = 0
    rows_rejected: int = 0
    quality_events: list[QualityEvent] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    @property
    def duration_s(self) -> float:
        end = self.finished_at or utcnow()
        return (end - self.started_at).total_seconds()

    def summary(self) -> str:
        return (
            f"{self.job} {self.asset} {self.timeframe or ''} via {self.source or 'n/a'}: "
            f"{self.status} - fetched={self.rows_fetched} written={self.rows_written} "
            f"rejected={self.rows_rejected} events={len(self.quality_events)} "
            f"in {self.duration_s:.1f}s"
        )
