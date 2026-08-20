"""Provider layer: throttling, circuit breaking, failover, and capability routing."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from mie.core.errors import NotSupported
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import MarketType, QualityEventType
from mie.providers.base import CircuitBreaker, CircuitState, TokenBucket
from mie.providers.binance import BinanceProvider
from mie.providers.coinbase import CoinbaseProvider
from mie.providers.fake import FakeProvider, fake_config
from mie.providers.kraken import KrakenProvider
from mie.providers.manager import ProviderManager


class TestTokenBucket:
    async def test_burst_is_served_immediately(self) -> None:
        bucket = TokenBucket(rate=10, burst=5)
        started = asyncio.get_running_loop().time()
        for _ in range(5):
            await bucket.acquire()
        assert asyncio.get_running_loop().time() - started < 0.05

    async def test_exhausted_bucket_throttles(self) -> None:
        """The point of the bucket: sustained rate is enforced once burst is spent."""
        bucket = TokenBucket(rate=20, burst=2)
        for _ in range(2):
            await bucket.acquire()
        waited = await bucket.acquire()
        assert waited > 0

    async def test_tokens_refill_over_time(self) -> None:
        bucket = TokenBucket(rate=100, burst=1)
        await bucket.acquire()
        await asyncio.sleep(0.05)
        assert bucket.available > 0

    def test_rate_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate=0, burst=1)


class TestCircuitBreaker:
    def test_opens_after_the_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3, cooldown_s=60)
        for _ in range(2):
            breaker.record_failure("boom")
        assert breaker.allows() is True
        breaker.record_failure("boom")
        assert breaker.state == CircuitState.OPEN
        assert breaker.allows() is False

    def test_success_resets_the_counter(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure("boom")
        breaker.record_failure("boom")
        breaker.record_success()
        assert breaker.consecutive_failures == 0
        breaker.record_failure("boom")
        assert breaker.allows() is True

    def test_half_open_probe_after_cooldown(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, cooldown_s=0.0)
        breaker.record_failure("boom")
        assert breaker.allows() is True
        assert breaker.state == CircuitState.HALF_OPEN

    def test_failed_probe_reopens_immediately(self) -> None:
        """No point burning the whole threshold again on a provider just confirmed down."""
        breaker = CircuitBreaker(failure_threshold=4, cooldown_s=0.0)
        for _ in range(4):
            breaker.record_failure("boom")
        breaker.allows()  # -> HALF_OPEN
        breaker.record_failure("still down")
        assert breaker.state == CircuitState.OPEN


class TestFakeProvider:
    async def test_output_is_deterministic(self) -> None:
        """Overlapping fetches must agree, or every ingestion test is untrustworthy."""
        provider = FakeProvider(fake_config())
        end = Timeframe.H1.floor(utcnow())
        start = end - timedelta(hours=10)
        first = await provider.fetch_ohlcv("BTC", Timeframe.H1, start=start, end=end)
        second = await provider.fetch_ohlcv("BTC", Timeframe.H1, start=start, end=end)
        assert [c.close for c in first] == [c.close for c in second]

    async def test_candles_are_grid_aligned_and_well_formed(self) -> None:
        provider = FakeProvider(fake_config())
        candles = await provider.fetch_ohlcv("ETH", Timeframe.M15, limit=20)
        assert candles
        for candle in candles:
            assert Timeframe.M15.is_aligned(candle.open_time)
            assert candle.low <= min(candle.open, candle.close)
            assert candle.high >= max(candle.open, candle.close)

    async def test_fault_injection_produces_the_requested_defects(self) -> None:
        provider = FakeProvider(fake_config(drop_every=5, duplicate_every=7))
        candles = await provider.fetch_ohlcv("BTC", Timeframe.H1, limit=40)
        times = [c.open_time for c in candles]
        assert len(times) != len(set(times)), "duplicates were requested"

    async def test_failures_can_be_scheduled(self) -> None:
        from mie.core.errors import ProviderUnavailable

        provider = FakeProvider(fake_config(fail_count=2))
        for _ in range(2):
            with pytest.raises(ProviderUnavailable):
                await provider.fetch_ohlcv("BTC", Timeframe.H1, limit=5)
        assert await provider.fetch_ohlcv("BTC", Timeframe.H1, limit=5)


class TestSymbolMapping:
    """Canonical symbols must survive translation to each venue's dialect."""

    def test_binance_quotes_in_usdt(self) -> None:
        provider = BinanceProvider(fake_config("binance"))
        assert provider.symbol_for("BTC", "USD") == "BTCUSDT"
        assert provider.symbol_for("ETH", "USDT") == "ETHUSDT"

    def test_coinbase_uses_dashed_usd_products(self) -> None:
        provider = CoinbaseProvider(fake_config("coinbase"))
        assert provider.symbol_for("BTC", "USDT") == "BTC-USD"

    def test_overrides_take_precedence(self) -> None:
        """Kraken calls Bitcoin XBT; convention cannot derive that."""
        provider = KrakenProvider(fake_config("kraken"), symbol_overrides={"BTC": "XBTUSD"})
        assert provider.symbol_for("BTC", "USD") == "XBTUSD"
        assert provider.symbol_for("ETH", "USD") == "ETHUSD"


class TestCapabilityRouting:
    def test_unsupported_timeframes_are_excluded(self) -> None:
        binance = BinanceProvider(fake_config("binance", priority=1))
        coinbase = CoinbaseProvider(fake_config("coinbase", priority=2))
        manager = ProviderManager([binance, coinbase])

        assert [p.name for p in manager.candidates("BTC", Timeframe.H1)] == ["binance", "coinbase"]
        assert [p.name for p in manager.candidates("BTC", Timeframe.H12)] == ["binance"]

    def test_unsupported_assets_are_excluded(self) -> None:
        """Coinbase has no BNB market; skipping it costs nothing, a 404 costs a round trip."""
        coinbase = CoinbaseProvider(fake_config("coinbase"))
        assert not coinbase.supports("BNB", Timeframe.H1)

    def test_preferred_provider_is_honoured_only_when_capable(self) -> None:
        binance = BinanceProvider(fake_config("binance", priority=1))
        coinbase = CoinbaseProvider(fake_config("coinbase", priority=2))
        manager = ProviderManager([binance, coinbase])

        assert [p.name for p in manager.candidates("BTC", Timeframe.H1, preferred="coinbase")] == [
            "coinbase"
        ]
        assert manager.candidates("BTC", Timeframe.H12, preferred="coinbase") == []

    def test_perp_market_routing(self) -> None:
        binance = BinanceProvider(fake_config("binance"))
        assert binance.supports("BTC", Timeframe.H1, MarketType.PERP)
        coinbase = CoinbaseProvider(fake_config("coinbase"))
        assert not coinbase.supports("BTC", Timeframe.H1, MarketType.PERP)


class TestFailover:
    async def test_falls_through_to_the_next_provider(self) -> None:
        primary = FakeProvider(fake_config("primary", priority=1, always_fail=True))
        secondary = FakeProvider(fake_config("secondary", priority=2))
        manager = ProviderManager([secondary, primary])

        outcome = await manager.fetch_ohlcv("BTC", Timeframe.H1, limit=5)
        assert outcome.provider == "secondary"
        assert outcome.candles
        assert [name for name, _ in outcome.attempts] == ["primary", "secondary"]

    async def test_failover_is_recorded_not_silent(self) -> None:
        """Data arriving from the third-choice venue is information, not a non-event."""
        primary = FakeProvider(fake_config("primary", priority=1, always_fail=True))
        secondary = FakeProvider(fake_config("secondary", priority=2))
        manager = ProviderManager([secondary, primary])

        outcome = await manager.fetch_ohlcv("BTC", Timeframe.H1, limit=5)
        kinds = {e.event_type for e in outcome.events}
        assert QualityEventType.PROVIDER_FAILOVER in kinds
        assert QualityEventType.PROVIDER_ERROR in kinds

    async def test_open_circuit_is_skipped_without_a_request(self) -> None:
        primary = FakeProvider(fake_config("primary", priority=1, always_fail=True))
        secondary = FakeProvider(fake_config("secondary", priority=2))
        manager = ProviderManager([secondary, primary])

        for _ in range(5):
            await manager.fetch_ohlcv("BTC", Timeframe.H1, limit=5)
        calls_before = primary.calls
        await manager.fetch_ohlcv("BTC", Timeframe.H1, limit=5)

        assert primary.breaker.state == CircuitState.OPEN
        assert primary.calls == calls_before, "an open circuit must not issue requests"

    async def test_capability_gaps_do_not_penalise_the_breaker(self) -> None:
        """Failing over for an unsupported asset is routing, not a fault."""
        primary = FakeProvider(fake_config("primary", priority=1, unsupported_assets=["BNB"]))
        secondary = FakeProvider(fake_config("secondary", priority=2))
        manager = ProviderManager([secondary, primary])

        outcome = await manager.fetch_ohlcv("BNB", Timeframe.H1, limit=5)
        assert outcome.provider == "secondary"
        assert primary.breaker.consecutive_failures == 0

    async def test_total_failure_reports_honestly(self) -> None:
        primary = FakeProvider(fake_config("primary", priority=1, always_fail=True))
        secondary = FakeProvider(fake_config("secondary", priority=2, always_fail=True))
        manager = ProviderManager([primary, secondary])

        outcome = await manager.fetch_ohlcv("BTC", Timeframe.H1, limit=5)
        assert outcome.ok is False
        assert outcome.candles == []
        assert outcome.error and "all providers failed" in outcome.error

    async def test_no_capable_provider_is_an_explicit_error(self) -> None:
        provider = FakeProvider(fake_config("only", unsupported_assets=["DOGE"]))
        manager = ProviderManager([provider])
        outcome = await manager.fetch_ohlcv("DOGE", Timeframe.H1, limit=5)
        assert outcome.ok is False
        assert "no provider serves" in (outcome.error or "")

    async def test_health_probes_every_provider(self) -> None:
        healthy = FakeProvider(fake_config("healthy", priority=1))
        broken = FakeProvider(fake_config("broken", priority=2, always_fail=True))
        manager = ProviderManager([healthy, broken])

        health = await manager.health()
        assert health["healthy"].ok is True
        assert health["broken"].ok is False


class TestCrossSourceAudit:
    async def test_price_disagreement_is_detected(self) -> None:
        """A venue quoting 5% away from its peers is broken, not merely different."""
        reference = FakeProvider(fake_config("reference", priority=1))
        divergent = FakeProvider(fake_config("divergent", priority=2, price_offset_pct=5.0))
        manager = ProviderManager([reference, divergent])

        events = await manager.compare_sources("BTC", Timeframe.H1, limit=5, tolerance_pct=0.5)
        assert events
        assert all(e.event_type is QualityEventType.SOURCE_DISCREPANCY for e in events)
        assert events[0].details["deviation_pct"] > 0.5

    async def test_agreeing_sources_produce_no_events(self) -> None:
        left = FakeProvider(fake_config("left", priority=1))
        right = FakeProvider(fake_config("right", priority=2))
        manager = ProviderManager([left, right])
        assert await manager.compare_sources("BTC", Timeframe.H1, limit=5) == []

    async def test_a_single_source_cannot_be_audited(self) -> None:
        manager = ProviderManager([FakeProvider(fake_config("solo"))])
        assert await manager.compare_sources("BTC", Timeframe.H1) == []


class TestAggregatorBoundary:
    async def test_coingecko_refuses_the_ohlcv_role(self) -> None:
        """Mixing aggregated cross-venue pricing into exchange candles would produce a
        silently inconsistent series."""
        from mie.providers.coingecko import CoinGeckoProvider

        provider = CoinGeckoProvider(fake_config("coingecko"))
        with pytest.raises(NotSupported):
            await provider.fetch_ohlcv("BTC", Timeframe.H1)
        assert provider.capabilities.timeframes == frozenset()
