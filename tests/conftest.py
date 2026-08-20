"""Shared fixtures.

Every fixture here is infrastructure-free: an in-file SQLite database in a tmp
directory and the deterministic synthetic provider. The suite must pass on a fresh
clone with no network and no services running, because a test that needs an exchange
to be up is a test that will eventually fail for reasons unrelated to the code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mie.config.settings import (
    AssetConfig,
    AssetUniverse,
    DatabaseConfig,
    IngestionConfig,
    ProviderConfig,
    QualityConfig,
    Settings,
)
from mie.core.timeframes import UTC, Timeframe
from mie.core.types import Candle, MarketType
from mie.providers.fake import FakeProvider, fake_config
from mie.providers.manager import ProviderManager
from mie.storage.db import Database
from mie.storage.repositories import ReferenceRepository

#: A fixed "now" so tests that reason about finality and staleness are reproducible.
#: On the grid for every supported timeframe including 1w (a Monday, 00:00 UTC).
FIXED_NOW = datetime(2025, 6, 2, 0, 0, tzinfo=UTC)


@pytest.fixture
def universe() -> AssetUniverse:
    return AssetUniverse(
        default_quote="USDT",
        assets=[
            AssetConfig(symbol="BTC", name="Bitcoin", tier=1),
            AssetConfig(symbol="ETH", name="Ethereum", tier=1),
            AssetConfig(symbol="SOL", name="Solana", tier=2),
        ],
    )


@pytest.fixture
def settings(tmp_path: Path, universe: AssetUniverse) -> Settings:
    """Settings pointed at a throwaway database, with throttling disabled."""
    return Settings(
        database=DatabaseConfig(url=f"sqlite+aiosqlite:///{tmp_path.as_posix()}/test.db"),
        ingestion=IngestionConfig(
            timeframes=[Timeframe.M1, Timeframe.H1, Timeframe.D1],
            live_timeframes=[Timeframe.M1, Timeframe.H1],
            poll_interval_s=0.01,
            batch_limit=200,
            max_concurrency=4,
            collect_derivatives=False,
            collect_global_metrics=False,
        ),
        quality=QualityConfig(),
        providers=[ProviderConfig(name="fake", priority=1)],
        universe=universe,
    )


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Database]:
    db = Database(settings)
    await db.create_schema()
    # The instrument-id cache is class-level; leaking it between tests would let one
    # test's ids resolve against another test's database.
    ReferenceRepository.clear_cache()
    try:
        yield db
    finally:
        await db.dispose()
        ReferenceRepository.clear_cache()


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider(fake_config("fake", priority=1))


@pytest.fixture
def manager(provider: FakeProvider) -> ProviderManager:
    return ProviderManager([provider])


def make_candle(
    open_time: datetime,
    close: float = 100.0,
    *,
    asset: str = "BTC",
    timeframe: Timeframe = Timeframe.H1,
    source: str = "fake",
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 10.0,
    is_final: bool = True,
) -> Candle:
    """Build a well-formed candle; override individual fields to break it on purpose."""
    open_price = close if open_ is None else open_
    return Candle(
        asset=asset,
        source=source,
        market_type=MarketType.SPOT,
        timeframe=timeframe,
        open_time=open_time,
        open=open_price,
        high=max(open_price, close) if high is None else high,
        low=min(open_price, close) if low is None else low,
        close=close,
        volume=volume,
        is_final=is_final,
    )


def series(
    count: int,
    *,
    start: datetime | None = None,
    timeframe: Timeframe = Timeframe.H1,
    base: float = 100.0,
    step: float = 0.5,
    asset: str = "BTC",
) -> list[Candle]:
    """A clean, gently trending series — the baseline every defect test perturbs."""
    start = start or (FIXED_NOW - timeframe.delta * count)
    return [
        make_candle(
            start + timeframe.delta * i,
            close=base + step * i,
            open_=base + step * (i - 1) if i else base,
            asset=asset,
            timeframe=timeframe,
        )
        for i in range(count)
    ]


@pytest.fixture
def clean_series() -> list[Candle]:
    return series(60)


@pytest.fixture
def hour() -> timedelta:
    return timedelta(hours=1)
