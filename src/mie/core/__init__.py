"""Core domain primitives shared by every layer of the engine."""

from mie.core.errors import (
    ConfigError,
    DataQualityError,
    MIEError,
    NotSupported,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    StorageError,
)
from mie.core.events import Event, EventBus, InProcessEventBus, Topics
from mie.core.logging import configure_logging, get_logger
from mie.core.timeframes import Timeframe, ensure_utc, expected_count, floor_to, grid, utcnow
from mie.core.types import (
    Candle,
    FundingRate,
    GlobalMetricsPoint,
    IngestResult,
    IngestStatus,
    MarketType,
    OpenInterestPoint,
    ProviderHealth,
    QualityEvent,
    QualityEventType,
    QualitySeverity,
)

__all__ = [
    "Candle",
    "ConfigError",
    "DataQualityError",
    "Event",
    "EventBus",
    "FundingRate",
    "GlobalMetricsPoint",
    "InProcessEventBus",
    "IngestResult",
    "IngestStatus",
    "MIEError",
    "MarketType",
    "NotSupported",
    "OpenInterestPoint",
    "ProviderError",
    "ProviderHealth",
    "ProviderUnavailable",
    "QualityEvent",
    "QualityEventType",
    "QualitySeverity",
    "RateLimited",
    "StorageError",
    "Timeframe",
    "Topics",
    "configure_logging",
    "ensure_utc",
    "expected_count",
    "floor_to",
    "get_logger",
    "grid",
    "utcnow",
]
