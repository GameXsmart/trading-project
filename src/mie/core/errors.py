"""Exception hierarchy.

Errors are split by *what the caller should do about them*, not by where they were
raised. The provider manager needs to know whether to fail over, back off, or give
up, and that decision is encoded in the type.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "DataQualityError",
    "MIEError",
    "NotSupported",
    "ProviderError",
    "ProviderUnavailable",
    "RateLimited",
    "StorageError",
]


class MIEError(Exception):
    """Base for every error the engine raises deliberately."""


class ConfigError(MIEError):
    """Configuration is missing, malformed, or internally inconsistent."""


class StorageError(MIEError):
    """Persistence failed in a way the caller cannot paper over."""


class ProviderError(MIEError):
    """A data provider failed. Retryable unless a subclass says otherwise."""

    retryable = True

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ProviderUnavailable(ProviderError):
    """Provider is down, unreachable, or its circuit breaker is open."""


class RateLimited(ProviderError):
    """Provider refused the request for rate reasons; honour retry_after."""

    def __init__(self, provider: str, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(provider, message)


class NotSupported(ProviderError):
    """Provider cannot serve this asset/timeframe/market combination.

    Not retryable: failing over is correct, retrying is pointless.
    """

    retryable = False


class DataQualityError(MIEError):
    """Incoming data is too broken to persist."""
