"""Pluggable market-data providers and the failover manager."""

from mie.providers.base import (
    CircuitBreaker,
    CircuitState,
    HttpProvider,
    MarketDataProvider,
    ProviderCapabilities,
    TokenBucket,
)
from mie.providers.binance import BinanceProvider
from mie.providers.coinbase import CoinbaseProvider
from mie.providers.coingecko import CoinGeckoProvider
from mie.providers.fake import FakeProvider, fake_config
from mie.providers.kraken import KrakenProvider
from mie.providers.manager import (
    PROVIDER_REGISTRY,
    FetchOutcome,
    ProviderManager,
    build_providers,
)

__all__ = [
    "PROVIDER_REGISTRY",
    "BinanceProvider",
    "CircuitBreaker",
    "CircuitState",
    "CoinGeckoProvider",
    "CoinbaseProvider",
    "FakeProvider",
    "FetchOutcome",
    "HttpProvider",
    "KrakenProvider",
    "MarketDataProvider",
    "ProviderCapabilities",
    "ProviderManager",
    "TokenBucket",
    "build_providers",
    "fake_config",
]
