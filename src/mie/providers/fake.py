"""Deterministic synthetic provider.

Exists so the test suite can exercise ingestion, validation and failover without a
network — network-dependent tests are flaky tests, and a validator that is only ever
run against well-formed live data is a validator nobody has checked.

Two properties make it useful:

* **Deterministic.** The bar for a given (asset, timeframe, open_time) is always
  identical, derived from a hash rather than from call order, so a test can fetch
  overlapping windows and assert they agree.
* **Fault-injecting.** Gaps, duplicates, shape violations, price spikes, staleness
  and outright failures can all be requested, which is how the quality layer's
  detectors get tested against the defects they exist to find.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from datetime import datetime, timedelta
from typing import Any

from mie.config.settings import ProviderConfig, RateLimitConfig
from mie.core.errors import NotSupported, ProviderUnavailable
from mie.core.timeframes import Timeframe, ensure_utc, grid, utcnow
from mie.core.types import (
    Candle,
    FundingRate,
    GlobalMetricsPoint,
    MarketType,
    OpenInterestPoint,
    ProviderHealth,
)
from mie.providers.base import MarketDataProvider, ProviderCapabilities

__all__ = ["FakeProvider", "fake_config"]

_BASE_PRICES = {
    "BTC": 60_000.0,
    "ETH": 3_000.0,
    "SOL": 150.0,
    "BNB": 550.0,
    "XRP": 0.6,
    "ADA": 0.45,
    "DOGE": 0.15,
    "AVAX": 30.0,
    "LINK": 15.0,
    "DOT": 6.0,
}


def fake_config(name: str = "fake", priority: int = 1, **options: Any) -> ProviderConfig:
    """A ProviderConfig with the throttling turned off, for tests."""
    return ProviderConfig(
        name=name,
        priority=priority,
        timeout_s=1.0,
        max_retries=1,
        rate_limit=RateLimitConfig(rate=10_000, burst=10_000),
        options=options,
    )


class FakeProvider(MarketDataProvider):
    """Synthetic market data with optional, precisely-targeted defects."""

    kind = "synthetic"

    def __init__(self, config: ProviderConfig, symbol_overrides: dict[str, str] | None = None) -> None:
        super().__init__(config, symbol_overrides)
        self.name = config.name
        opts = config.options
        self.fail_count: int = int(opts.get("fail_count", 0))
        self.always_fail: bool = bool(opts.get("always_fail", False))
        self.latency_s: float = float(opts.get("latency_s", 0.0))
        self.drop_every: int = int(opts.get("drop_every", 0))
        self.duplicate_every: int = int(opts.get("duplicate_every", 0))
        self.spike_every: int = int(opts.get("spike_every", 0))
        self.break_shape_every: int = int(opts.get("break_shape_every", 0))
        self.stale_after: datetime | None = opts.get("stale_after")
        self.flatline: bool = bool(opts.get("flatline", False))
        self.price_offset_pct: float = float(opts.get("price_offset_pct", 0.0))
        self.unsupported: frozenset[str] = frozenset(
            a.upper() for a in opts.get("unsupported_assets", ())
        )
        self.calls: int = 0
        self.call_log: list[dict[str, Any]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            timeframes=frozenset(Timeframe),
            market_types=frozenset({MarketType.SPOT, MarketType.PERP}),
            max_candles_per_request=int(self.config.options.get("max_candles", 1000)),
            supports_funding=True,
            supports_open_interest=True,
            supports_global_metrics=True,
            unsupported_assets=self.unsupported,
        )

    async def health(self) -> ProviderHealth:
        healthy = not self.always_fail
        return ProviderHealth(
            provider=self.name,
            ok=healthy,
            latency_ms=self.latency_s * 1000,
            error=None if healthy else "synthetic failure",
            detail=self.breaker.snapshot(),
        )

    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
    ) -> list[Candle]:
        self.calls += 1
        self.call_log.append(
            {"asset": asset, "timeframe": str(timeframe), "start": start, "end": end, "limit": limit}
        )
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.always_fail or self.calls <= self.fail_count:
            raise ProviderUnavailable(self.name, "synthetic failure")
        if asset.upper() in self.unsupported:
            raise NotSupported(self.name, f"no {asset.upper()} market")

        now = utcnow()
        max_candles = self.capabilities.max_candles_per_request
        requested = min(limit or max_candles, max_candles)

        if end is None:
            end = now
        if start is None:
            start = end - timedelta(seconds=timeframe.seconds * requested)
        start, end = ensure_utc(start), ensure_utc(end)
        if self.stale_after is not None:
            end = min(end, ensure_utc(self.stale_after))

        candles: list[Candle] = []
        for index, open_time in enumerate(grid(start, end, timeframe)):
            if len(candles) >= requested:
                break
            if self.drop_every and index % self.drop_every == 0 and index > 0:
                continue  # synthetic gap
            candle = self._make(asset, quote, timeframe, open_time, index, now)
            candles.append(candle)
            if self.duplicate_every and index % self.duplicate_every == 0 and index > 0:
                candles.append(candle)  # synthetic duplicate
        return candles

    def _make(
        self,
        asset: str,
        quote: str,
        timeframe: Timeframe,
        open_time: datetime,
        index: int,
        now: datetime,
    ) -> Candle:
        base = _BASE_PRICES.get(asset.upper(), 100.0) * (1 + self.price_offset_pct / 100.0)
        bucket = int(open_time.timestamp()) // timeframe.seconds

        if self.flatline:
            price = base
            noise = 0.0
        else:
            # A slow cycle plus a hash-derived jitter: deterministic per bucket, but
            # not so smooth that indicator tests become degenerate.
            price = base * (1.0 + 0.08 * math.sin(bucket / 97.0))
            noise = (_unit_hash(asset, timeframe, bucket) - 0.5) * 0.004
            price *= 1.0 + noise

        spread = abs(price) * (0.0 if self.flatline else 0.0015)
        open_price = price
        close_price = price * (1.0 + noise)
        high = max(open_price, close_price) + spread
        low = min(open_price, close_price) - spread
        volume = 0.0 if self.flatline else 100.0 + _unit_hash(asset, timeframe, bucket + 1) * 900.0

        if self.spike_every and index % self.spike_every == 0 and index > 0:
            close_price *= 3.0  # implausible single-bar move
            high = max(high, close_price)
        if self.break_shape_every and index % self.break_shape_every == 0 and index > 0:
            high, low = low, high  # high < low: must be rejected outright

        return Candle(
            asset=asset.upper(),
            quote=quote,
            source=self.name,
            timeframe=timeframe,
            open_time=open_time,
            open=round(open_price, 6),
            high=round(high, 6),
            low=round(low, 6),
            close=round(close_price, 6),
            volume=round(volume, 4),
            quote_volume=round(volume * price, 2),
            trades=int(volume),
            is_final=timeframe.close_time(open_time) <= now,
        )

    async def fetch_funding(self, asset: str, limit: int = 100) -> list[FundingRate]:
        now = utcnow()
        return [
            FundingRate(
                asset=asset.upper(),
                source=self.name,
                ts=now - timedelta(hours=8 * i),
                rate=(_unit_hash(asset, Timeframe.H1, i) - 0.5) * 0.0006,
            )
            for i in range(min(limit, 10))
        ]

    async def fetch_open_interest(self, asset: str, limit: int = 100) -> list[OpenInterestPoint]:
        now = utcnow()
        return [
            OpenInterestPoint(
                asset=asset.upper(),
                source=self.name,
                ts=now - timedelta(hours=i),
                open_interest=10_000 + _unit_hash(asset, Timeframe.H1, i) * 1_000,
            )
            for i in range(min(limit, 10))
        ]

    async def fetch_global_metrics(self) -> GlobalMetricsPoint:
        return GlobalMetricsPoint(
            source=self.name,
            ts=utcnow(),
            btc_dominance=54.0,
            eth_dominance=17.0,
            total_market_cap_usd=2.1e12,
            total_volume_24h_usd=8.5e10,
            stablecoin_share=6.5,
        )


def _unit_hash(asset: str, timeframe: Timeframe, bucket: int) -> float:
    """Stable pseudo-random float in [0, 1) — same inputs, same value, forever."""
    seed = f"{asset.upper()}:{timeframe}:{bucket}".encode()
    digest = hashlib.blake2b(seed, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64
