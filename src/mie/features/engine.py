"""Feature engine.

Subscribes to `candle.closed`, maintains one warm indicator set per series, and
persists a feature vector per bar.

Three properties are load-bearing:

* **One indicator set per (instrument, timeframe).** Features are keyed by
  *instrument*, not by asset, for the same reason OHLCV is: feeding Binance and
  Coinbase bars into a single EMA during a failover would silently corrupt it. Two
  venues are two series.
* **Chronological, no replay.** A bar older than or equal to the last one processed
  is refused rather than folded in — recursive indicators cannot un-see a value, so
  an out-of-order bar would permanently poison the state.
* **Only final bars.** The forming bar never reaches an indicator. Combined with the
  repository default this is the second half of the look-ahead defence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mie.config.settings import Settings
from mie.core.events import Event, EventBus, Topics
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import Candle, MarketType
from mie.features.indicators import (
    ADX,
    ATR,
    EMA,
    MACD,
    OBV,
    ROC,
    RSI,
    SMA,
    AnchoredVWAP,
    BollingerBands,
    Indicator,
    RealisedVolatility,
    Stochastic,
)
from mie.features.levels import StructureAnalyzer
from mie.storage.db import Database
from mie.storage.repositories import FeatureRepository, OHLCVRepository

log = get_logger(__name__)

__all__ = ["FEATURE_SET_VERSION", "FeatureEngine", "FeatureSet", "build_indicators"]

#: Bumped whenever the feature definitions change. Stored on every row so a later
#: phase can tell "this model was trained on v1 features" instead of silently mixing
#: two incompatible definitions in one training set.
FEATURE_SET_VERSION = 1

#: Bars per year per timeframe, for annualising realised volatility. Crypto trades
#: continuously, so these are simple calendar divisions with no session adjustment.
_BARS_PER_YEAR: dict[Timeframe, float] = {
    Timeframe.M1: 365 * 24 * 60,
    Timeframe.M5: 365 * 24 * 12,
    Timeframe.M15: 365 * 24 * 4,
    Timeframe.M30: 365 * 24 * 2,
    Timeframe.H1: 365 * 24,
    Timeframe.H4: 365 * 6,
    Timeframe.H12: 365 * 2,
    Timeframe.D1: 365,
    Timeframe.W1: 52,
}


def build_indicators(timeframe: Timeframe) -> list[Indicator]:
    """The standard indicator battery for one series.

    Periods are the conventional ones. That is a deliberate choice rather than a lazy
    one: tuning periods per asset on historical data is a well-known way to
    manufacture backtest performance that does not survive contact with the future,
    and Phase 8 is where any such change would have to prove itself.
    """
    bars_per_year = _BARS_PER_YEAR.get(timeframe, 365 * 24)
    return [
        SMA(20),
        SMA(50),
        SMA(200),
        EMA(9),
        EMA(21),
        EMA(50),
        RSI(14),
        MACD(12, 26, 9),
        ATR(14),
        BollingerBands(20, 2.0),
        ADX(14),
        Stochastic(14, 3),
        OBV(20),
        AnchoredVWAP(),
        RealisedVolatility(20, bars_per_year),
        ROC(10),
    ]


@dataclass(slots=True)
class FeatureSet:
    """Warm indicator state for one (instrument, timeframe) series."""

    asset: str
    timeframe: Timeframe
    source: str
    market_type: MarketType = MarketType.SPOT
    indicators: list[Indicator] = field(default_factory=list)
    structure: StructureAnalyzer | None = None
    last_open_time: datetime | None = None
    bars_seen: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.asset, str(self.timeframe), self.source)

    @property
    def warmup(self) -> int:
        """Bars needed before every indicator in the set is defined."""
        needed = [i.warmup for i in self.indicators]
        if self.structure is not None:
            needed.append(self.structure.warmup)
        return max(needed, default=0)

    @property
    def is_warm(self) -> bool:
        return self.bars_seen >= self.warmup

    def update(self, candle: Candle) -> dict[str, float]:
        """Fold one closed bar into every indicator and return the feature vector.

        Raises on a non-final or out-of-order bar rather than coping quietly: both
        indicate a bug upstream, and both corrupt recursive state irreversibly.
        """
        if not candle.is_final:
            raise ValueError(
                f"refusing provisional candle {candle.asset} {candle.timeframe} "
                f"{candle.open_time.isoformat()}: features must never see a forming bar"
            )
        if self.last_open_time is not None and candle.open_time <= self.last_open_time:
            raise ValueError(
                f"out-of-order candle {candle.open_time.isoformat()} after "
                f"{self.last_open_time.isoformat()}: indicator state cannot be rewound"
            )

        self.last_open_time = candle.open_time
        self.bars_seen += 1

        values: dict[str, float] = {}
        for indicator in self.indicators:
            result = indicator.update(candle)
            if result is None:
                continue
            if isinstance(result, dict):
                # Composite indicators publish one key per component, namespaced by
                # the indicator so `macd.signal` never collides with a bare `signal`.
                for suffix, value in result.items():
                    values[f"{indicator.name}.{suffix}"] = value
            else:
                values[indicator.name] = result

        if self.structure is not None:
            structure = self.structure.update(candle)
            if structure is not None:
                values |= structure.as_features(candle.close)

        # Raw bar context, so a stored vector is self-contained for later analysis.
        values |= {
            "close": candle.close,
            "volume": candle.volume,
            "range_pct": candle.range_pct,
            "change_pct": candle.change_pct,
        }
        return values


class FeatureEngine:
    """Owns feature state for every watched series and persists per-bar vectors."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        bus: EventBus | None = None,
        include_structure: bool = True,
    ) -> None:
        self.db = database
        self.settings = settings
        self.bus = bus
        self.include_structure = include_structure
        self._sets: dict[tuple[str, str, str], FeatureSet] = {}
        self.processed = 0
        self.skipped = 0

    # ------------------------------------------------------------------ wiring

    def subscribe(self) -> None:
        """Attach to the ingestion event stream."""
        if self.bus is None:
            raise ValueError("no event bus configured")
        self.bus.subscribe(Topics.CANDLE_CLOSED, self._on_candle_closed)
        log.info("feature_engine_subscribed", topic=Topics.CANDLE_CLOSED)

    async def _on_candle_closed(self, event: Event) -> None:
        payload = event.payload
        candles = payload if isinstance(payload, list) else [payload]
        for candle in candles:
            if isinstance(candle, Candle):
                await self.handle(candle)

    # ---------------------------------------------------------------- lifecycle

    def _set_for(self, candle: Candle) -> FeatureSet:
        key = (candle.asset, str(candle.timeframe), candle.source)
        existing = self._sets.get(key)
        if existing is not None:
            return existing
        created = FeatureSet(
            asset=candle.asset,
            timeframe=candle.timeframe,
            source=candle.source,
            market_type=candle.market_type,
            indicators=build_indicators(candle.timeframe),
            structure=StructureAnalyzer() if self.include_structure else None,
        )
        self._sets[key] = created
        return created

    async def warmup(
        self, asset: str, timeframe: Timeframe, source: str, extra_bars: int = 50
    ) -> FeatureSet | None:
        """Prime a series from stored history so live bars produce values immediately.

        Without this, a restart would emit nothing until 200+ new bars had arrived —
        on a daily series that is most of a year. Only final bars are loaded, and they
        are replayed in order, so the warm state is identical to having run
        continuously.
        """
        feature_set = FeatureSet(
            asset=asset.upper(),
            timeframe=timeframe,
            source=source,
            indicators=build_indicators(timeframe),
            structure=StructureAnalyzer() if self.include_structure else None,
        )
        needed = feature_set.warmup + extra_bars

        async with self.db.session() as session:
            rows = await OHLCVRepository(session).fetch_recent(
                asset, timeframe, source=source, limit=needed, final_only=True
            )
        if not rows:
            log.debug("feature_warmup_no_history", asset=asset, timeframe=str(timeframe))
            return None

        for row in rows:
            feature_set.update(_row_to_candle(row, asset, timeframe, source))

        self._sets[feature_set.key] = feature_set
        log.info(
            "feature_set_warm",
            asset=asset,
            timeframe=str(timeframe),
            source=source,
            bars=len(rows),
            warm=feature_set.is_warm,
        )
        return feature_set

    # ----------------------------------------------------------------- compute

    async def handle(self, candle: Candle, persist: bool = True) -> dict[str, float] | None:
        """Process one closed bar. Returns the feature vector, or None if skipped."""
        if not candle.is_final:
            self.skipped += 1
            return None

        feature_set = self._set_for(candle)
        try:
            values = feature_set.update(candle)
        except ValueError as exc:
            # Almost always a duplicate delivery from an overlapping poll, which is
            # expected and harmless — the bar is already folded in.
            self.skipped += 1
            log.debug(
                "feature_candle_skipped",
                asset=candle.asset,
                timeframe=str(candle.timeframe),
                reason=str(exc)[:160],
            )
            return None

        if not feature_set.is_warm:
            return values

        self.processed += 1
        if persist:
            await self._persist(candle, values)
        return values

    async def _persist(self, candle: Candle, values: dict[str, float]) -> None:
        async with self.db.session() as session:
            await FeatureRepository(session).upsert(
                asset=candle.asset,
                source=candle.source,
                market_type=candle.market_type,
                timeframe=candle.timeframe,
                open_time=candle.open_time,
                values=values,
                version=FEATURE_SET_VERSION,
            )

    async def backfill(
        self,
        asset: str,
        timeframe: Timeframe,
        source: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Compute and store features across stored history, oldest first.

        Runs the same code path as live operation — one bar at a time, in order —
        rather than a separate vectorised implementation. Two implementations of the
        same feature is two chances to disagree, and the disagreement would only show
        up as a model that behaves differently in backtest than in production.
        """
        async with self.db.session() as session:
            rows = await OHLCVRepository(session).fetch(
                asset, timeframe, source=source, start=start, end=end, final_only=True
            )
        if not rows:
            return 0

        key = (asset.upper(), str(timeframe), source)
        self._sets.pop(key, None)  # rebuild state from the beginning of the range

        written = 0
        batch: list[dict[str, Any]] = []
        for row in rows:
            candle = _row_to_candle(row, asset, timeframe, source)
            feature_set = self._set_for(candle)
            values = feature_set.update(candle)
            if not feature_set.is_warm:
                continue
            batch.append(
                {
                    "open_time": candle.open_time,
                    "values": values,
                }
            )
            written += 1

        if batch:
            async with self.db.session() as session:
                await FeatureRepository(session).upsert_many(
                    asset=asset,
                    source=source,
                    market_type=MarketType.SPOT,
                    timeframe=timeframe,
                    rows=batch,
                    version=FEATURE_SET_VERSION,
                )
        log.info(
            "feature_backfill_complete",
            asset=asset,
            timeframe=str(timeframe),
            source=source,
            bars=len(rows),
            written=written,
        )
        return written

    # -------------------------------------------------------------------- info

    def stats(self) -> dict[str, Any]:
        return {
            "series": len(self._sets),
            "processed": self.processed,
            "skipped": self.skipped,
            "warm": [k for k, v in self._sets.items() if v.is_warm],
            "cold": [k for k, v in self._sets.items() if not v.is_warm],
        }


def _row_to_candle(row: Any, asset: str, timeframe: Timeframe, source: str) -> Candle:
    """Rehydrate a stored OHLCV row into the domain type indicators consume."""
    return Candle(
        asset=asset.upper(),
        source=source,
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
        ingested_at=row.ingested_at or utcnow(),
    )
