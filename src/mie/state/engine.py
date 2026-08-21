"""State engine.

Reads the latest stored feature vector for each timeframe, classifies each one, and
combines them into a :class:`MarketState`.

Two details matter for correctness:

* **`as_of` is honoured, not "latest".** Rebuilding the state as it stood at some past
  moment must use only feature vectors whose bar had closed by then. Without that, any
  historical study of state accuracy is contaminated by hindsight, and Phase 8's
  backtests would be measuring the future's influence on the past.
* **Data quality flows in.** The Phase 1 trust score for each series multiplies into
  that timeframe's confidence, so a degraded feed produces a quieter state rather than
  a confidently wrong one. This is where §20 becomes visible in an actual output.
"""

from __future__ import annotations

from datetime import datetime

from mie.config.settings import Settings
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.state.classifier import TimeframeClassifier
from mie.state.hierarchy import HierarchyAnalyzer
from mie.state.types import MarketState, TimeframeState
from mie.storage.db import Database
from mie.storage.models import FeatureRow, MarketStateRow
from mie.storage.repositories import (
    FeatureRepository,
    MarketStateRepository,
    QualityRepository,
)

log = get_logger(__name__)

__all__ = ["StateEngine"]


class StateEngine:
    """Builds and persists multi-timeframe market state."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        source: str = "binance",
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        self.db = database
        self.settings = settings
        self.source = source
        # Slowest first so the hierarchy reads top-down, the way it is reasoned about.
        self.timeframes = sorted(
            timeframes or settings.ingestion.live_timeframes,
            key=lambda tf: -tf.seconds,
        )
        self.classifier = TimeframeClassifier()
        self.hierarchy = HierarchyAnalyzer()

    async def build(
        self, asset: str, as_of: datetime | None = None, persist: bool = False
    ) -> MarketState:
        """Compute the market state for one asset.

        ``as_of`` reconstructs the state as it stood at that moment, using only bars
        that had closed by then.
        """
        asset = asset.upper()
        states: list[TimeframeState] = []
        qualities: list[float] = []

        async with self.db.session() as session:
            features = FeatureRepository(session)
            quality = QualityRepository(session)

            for timeframe in self.timeframes:
                row = (
                    await features.latest(asset, timeframe, source=self.source)
                    if as_of is None
                    else await _latest_before(features, asset, timeframe, self.source, as_of)
                )
                if row is None:
                    continue

                trust = await quality.get_score(self.source, asset, timeframe)
                qualities.append(trust)
                states.append(
                    self.classifier.classify(
                        asset=asset,
                        timeframe=timeframe,
                        features=row.payload,
                        as_of=row.open_time,
                        data_quality=trust,
                    )
                )

        overall_quality = sum(qualities) / len(qualities) if qualities else 1.0
        state = self.hierarchy.analyse(asset, states, data_quality=overall_quality)

        if persist and states:
            await self.persist(state)
        return state

    async def build_all(
        self, assets: list[str] | None = None, persist: bool = False
    ) -> list[MarketState]:
        """Compute state for every configured asset."""
        symbols = assets or self.settings.universe.symbols()
        return [await self.build(symbol, persist=persist) for symbol in symbols]

    async def persist(self, state: MarketState) -> None:
        """Store the state, including every per-timeframe level.

        The levels are stored, not just the aggregate: the "Why?" panel needs them,
        and Phase 9 slices model performance by regime, which requires knowing what
        the regime actually was at prediction time rather than inferring it later.
        """
        async with self.db.session() as session:
            await MarketStateRepository(session).upsert(state)

    async def latest_stored(self, asset: str) -> MarketStateRow | None:
        """Most recently persisted state row for an asset.

        Returns the stored row rather than a rehydrated :class:`MarketState`: the row
        is what callers reading history actually want, and pretending to return the
        domain object would misrepresent what comes back.
        """
        async with self.db.session() as session:
            return await MarketStateRepository(session).latest(asset)


async def _latest_before(
    features: FeatureRepository,
    asset: str,
    timeframe: Timeframe,
    source: str,
    as_of: datetime,
) -> FeatureRow | None:
    """Newest feature vector whose bar had closed at or before ``as_of``.

    The bound is on the bar's *close*, not its open: a bar opening before ``as_of``
    but closing after it was still forming at that moment, and its features could not
    have been known.
    """
    rows = await features.fetch(asset, timeframe, source=source, end=as_of)
    usable = [r for r in rows if timeframe.close_time(r.open_time) <= as_of]
    return usable[-1] if usable else None
