"""Live provider contract tests.

Marked ``network`` and excluded from the default run — a suite that needs an exchange
to be up is a suite that fails for reasons unrelated to the code. They exist because
the rest of the tests use a synthetic provider, and something has to catch the day an
exchange changes its response shape.

    pytest -m network           # run only these
    pytest -m "not network"     # the default; everything else
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from mie.config.settings import load_settings
from mie.core.errors import ProviderError
from mie.core.timeframes import Timeframe, utcnow
from mie.providers.binance import BinanceProvider
from mie.providers.coingecko import CoinGeckoProvider
from mie.providers.manager import ProviderManager, build_providers

pytestmark = pytest.mark.network


def _config(name: str):
    """The shipped ProviderConfig for one provider, so live tests use real timeouts."""
    config = load_settings().provider(name)
    assert config is not None, f"{name} missing from config/default.yaml"
    return config


@pytest.fixture
async def providers():
    """The exchange providers, built from the *shipped* configuration.

    Deliberately not the test-only `fake_config`: that sets a one-second timeout and a
    single attempt, which is right for offline tests and far too tight for a real
    round trip across the internet. Using the real config means this suite also
    exercises the timeouts, retry policy, rate limits, and symbol overrides that
    production actually runs with.
    """
    settings = load_settings()
    built = [p for p in build_providers(settings) if p.capabilities.timeframes]
    try:
        yield built
    finally:
        for provider in built:
            await provider.close()


class TestLiveContract:
    async def test_every_provider_is_reachable(self, providers) -> None:
        for provider in providers:
            health = await provider.health()
            assert health.ok, f"{provider.name} unhealthy: {health.error}"

    @pytest.mark.parametrize("timeframe", [Timeframe.M5, Timeframe.H1, Timeframe.D1])
    async def test_candles_are_well_formed_and_aligned(self, providers, timeframe) -> None:
        """The contract the whole system relies on: aligned, ordered, shape-valid bars."""
        for provider in providers:
            candles = await provider.fetch_ohlcv("BTC", timeframe, limit=10)
            assert candles, f"{provider.name} returned nothing"
            times = [c.open_time for c in candles]
            assert times == sorted(times), f"{provider.name} returned unordered candles"
            assert len(times) == len(set(times)), f"{provider.name} returned duplicates"
            for candle in candles:
                assert timeframe.is_aligned(candle.open_time), f"{provider.name} misaligned"
                assert candle.high >= max(candle.open, candle.close)
                assert candle.low <= min(candle.open, candle.close)
                assert candle.volume >= 0
                assert candle.open_time.tzinfo is not None

    async def test_the_forming_bar_is_marked_provisional(self, providers) -> None:
        """If this regresses, live operation silently gains look-ahead bias."""
        for provider in providers:
            candles = await provider.fetch_ohlcv("BTC", Timeframe.H1, limit=5)
            newest = max(candles, key=lambda c: c.open_time)
            assert newest.is_final is (newest.close_time <= utcnow())
            assert all(c.is_final for c in candles if c.close_time <= utcnow())

    async def test_requested_windows_are_respected(self, providers) -> None:
        end = Timeframe.H1.floor(utcnow())
        start = end - timedelta(hours=6)
        for provider in providers:
            candles = await provider.fetch_ohlcv("BTC", Timeframe.H1, start=start, end=end)
            assert candles
            assert all(start <= c.open_time < end for c in candles), provider.name

    async def test_unknown_symbols_raise_rather_than_return_junk(self, providers) -> None:
        for provider in providers:
            with pytest.raises(ProviderError):
                await provider.fetch_ohlcv("NOTAREALCOIN", Timeframe.H1, limit=5)


class TestLiveDerivativesAndMetrics:
    async def test_binance_funding_and_open_interest(self) -> None:
        provider = BinanceProvider(_config("binance"))
        try:
            funding = await provider.fetch_funding("BTC", limit=5)
            assert funding
            assert all(abs(f.rate) < 0.05 for f in funding), "funding rates are small numbers"
            assert all(f.ts.tzinfo is not None for f in funding)

            oi = await provider.fetch_open_interest("BTC", limit=5)
            assert oi
            assert all(point.open_interest > 0 for point in oi)
        finally:
            await provider.close()

    async def test_coingecko_global_metrics(self) -> None:
        provider = CoinGeckoProvider(_config("coingecko"))
        try:
            metrics = await provider.fetch_global_metrics()
        finally:
            await provider.close()
        assert 0 < (metrics.btc_dominance or 0) < 100
        assert (metrics.total_market_cap_usd or 0) > 1e10


class TestLiveCrossSource:
    async def test_independent_venues_agree_on_price(self, providers) -> None:
        """Real venues genuinely differ, but a spread beyond ~2% on BTC means
        something is broken rather than merely different."""
        manager = ProviderManager(providers)
        events = await manager.compare_sources("BTC", Timeframe.H1, limit=3, tolerance_pct=2.0)
        assert events == [], f"unexpected inter-venue disagreement: {[str(e) for e in events]}"

    async def test_failover_reaches_a_live_fallback(self, providers) -> None:
        manager = ProviderManager(providers)
        primary = manager.get("binance")
        assert primary is not None
        for _ in range(primary.breaker.failure_threshold):
            primary.breaker.record_failure("simulated outage")

        outcome = await manager.fetch_ohlcv("BTC", Timeframe.H1, limit=3)
        assert outcome.ok
        assert outcome.provider != "binance"
        assert outcome.candles
