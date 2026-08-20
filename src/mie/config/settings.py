"""Typed configuration.

Three layers, lowest priority first:

1. defaults declared on the models below,
2. ``config/default.yaml`` (checked in, no secrets),
3. environment variables / ``.env`` (``MIE_`` prefix, ``__`` nests).

So ``MIE_DATABASE__URL=postgresql+asyncpg://...`` overrides ``database.url`` from
YAML. Secrets live only in layer 3 — nothing sensitive is ever committed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from mie.core.errors import ConfigError
from mie.core.timeframes import Timeframe

__all__ = [
    "AppConfig",
    "AssetConfig",
    "AssetUniverse",
    "DatabaseConfig",
    "IngestionConfig",
    "ProviderConfig",
    "QualityConfig",
    "Settings",
    "load_assets",
    "load_settings",
    "project_root",
]


def project_root() -> Path:
    """Repository root, derived from this file's location (src/mie/config/…)."""
    return Path(__file__).resolve().parents[3]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppConfig(_Strict):
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    data_dir: str = "./data"


class DatabaseConfig(_Strict):
    """Storage target.

    SQLite is the zero-infrastructure default so the repo is runnable and testable
    out of the box; TimescaleDB is the production target and is selected purely by
    changing this URL.
    """

    url: str = "sqlite+aiosqlite:///./data/mie.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    statement_timeout_s: int = 30
    apply_timescale: bool = True

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith(("postgresql", "postgres"))

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


class RateLimitConfig(_Strict):
    """Token bucket: sustained ``rate`` requests/second, bursting to ``burst``."""

    rate: float = 8.0
    burst: int = 16


class CircuitBreakerConfig(_Strict):
    failure_threshold: int = 4
    cooldown_s: float = 60.0
    half_open_probes: int = 1


class ProviderConfig(_Strict):
    name: str
    enabled: bool = True
    priority: int = 100  # lower wins
    timeout_s: float = 15.0
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    base_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class IngestionConfig(_Strict):
    timeframes: list[Timeframe] = Field(
        default_factory=lambda: [
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.H12,
            Timeframe.D1,
            Timeframe.W1,
        ]
    )
    # History depth per timeframe. Deep 1m history is enormous and rarely improves a
    # daily-horizon model, so depth is tuned per resolution rather than uniform.
    backfill_days: dict[Timeframe, int] = Field(
        default_factory=lambda: {
            Timeframe.M1: 7,
            Timeframe.M5: 30,
            Timeframe.M15: 90,
            Timeframe.M30: 180,
            Timeframe.H1: 365,
            Timeframe.H4: 730,
            Timeframe.H12: 1095,
            Timeframe.D1: 1825,
            Timeframe.W1: 2555,
        }
    )
    live_timeframes: list[Timeframe] = Field(
        default_factory=lambda: [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]
    )
    poll_interval_s: float = 20.0
    max_concurrency: int = 6
    batch_limit: int = 1000
    lookback_candles_on_poll: int = 3
    collect_derivatives: bool = True
    collect_global_metrics: bool = True
    derivatives_interval_s: float = 300.0
    global_metrics_interval_s: float = 600.0

    @field_validator("timeframes", "live_timeframes", mode="before")
    @classmethod
    def _parse_tfs(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [Timeframe.parse(v) for v in value]
        return value

    @field_validator("backfill_days", mode="before")
    @classmethod
    def _parse_backfill(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {Timeframe.parse(k): v for k, v in value.items()}
        return value


class QualityConfig(_Strict):
    """Thresholds for the validation layer.

    Defaults are intentionally permissive enough not to flood the event log on
    normal crypto volatility, and strict enough to catch genuinely broken feeds.
    """

    # Robust z-score (MAD-scaled) beyond which a bar is flagged as an outlier.
    # Calibrated against real BTC history rather than guessed: on 1000 bars of
    # 5m/1h/1d Binance data, z>10 fires on 0.2-0.4% of bars — ordinary fat-tailed
    # crypto volatility, not corruption — while nothing at all exceeded z>30. At 25
    # the check stays silent on real market action and still catches the kind of
    # corruption the hard cap misses.
    outlier_mad_threshold: float = 25.0
    outlier_min_samples: int = 30
    # Hard ceilings on a single bar's absolute return. Beyond these, the data is far
    # more likely to be corrupt than real — even for crypto.
    max_move_pct: dict[Timeframe, float] = Field(
        default_factory=lambda: {
            Timeframe.M1: 12.0,
            Timeframe.M5: 20.0,
            Timeframe.M15: 25.0,
            Timeframe.M30: 30.0,
            Timeframe.H1: 35.0,
            Timeframe.H4: 50.0,
            Timeframe.H12: 60.0,
            Timeframe.D1: 75.0,
            Timeframe.W1: 150.0,
        }
    )
    staleness_multiplier: float = 3.0
    flatline_run_length: int = 8
    # Weighted events per 1000 candles at which the score loses ~63% of its value.
    # Quality is judged as a *rate*, not a count: 40 warnings across a year of hourly
    # history is a healthy feed, while 40 warnings across an hour is a broken one.
    event_rate_tolerance: float = 8.0
    # Floor on the assessed-candle denominator, so a handful of events on a tiny
    # sample cannot be diluted into insignificance.
    min_exposure_candles: int = 50
    gap_warning_ratio: float = 0.01
    gap_error_ratio: float = 0.10
    source_discrepancy_pct: float = 0.5
    score_window_hours: int = 24
    min_score: float = 0.1

    @field_validator("max_move_pct", mode="before")
    @classmethod
    def _parse_moves(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {Timeframe.parse(k): float(v) for k, v in value.items()}
        return value


class AssetConfig(_Strict):
    symbol: str
    name: str = ""
    tier: int = 2
    enabled: bool = True
    quote: str = "USDT"
    # Per-provider symbol overrides for the cases convention cannot cover
    # (Kraken calls Bitcoin XBT, for example).
    overrides: dict[str, str] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()


class AssetUniverse(_Strict):
    """The set of assets under observation, loaded from ``config/assets.yaml``."""

    default_quote: str = "USDT"
    assets: list[AssetConfig] = Field(default_factory=list)

    def enabled(self) -> list[AssetConfig]:
        return [a for a in self.assets if a.enabled]

    def symbols(self) -> list[str]:
        return [a.symbol for a in self.enabled()]

    def get(self, symbol: str) -> AssetConfig | None:
        target = symbol.strip().upper()
        return next((a for a in self.assets if a.symbol == target), None)


class _YamlSource(PydanticBaseSettingsSource):
    """Feeds ``config/default.yaml`` into the settings resolution chain."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False  # not used; __call__ supplies the whole mapping

    def __call__(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {self._path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{self._path} must contain a mapping at the top level")
        return raw


class Settings(BaseSettings):
    """Root configuration object. Construct via :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="MIE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)

    # Populated by load_settings from config/assets.yaml; kept off the env chain
    # because it is a separately edited file with a different lifecycle.
    universe: AssetUniverse = Field(default_factory=AssetUniverse)

    _config_path: Path | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get("MIE_CONFIG_FILE", project_root() / "config" / "default.yaml"))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, yaml_path),
            file_secret_settings,
        )

    def enabled_providers(self) -> list[ProviderConfig]:
        """Enabled providers in resolution order (lowest priority number first)."""
        return sorted((p for p in self.providers if p.enabled), key=lambda p: (p.priority, p.name))

    def provider(self, name: str) -> ProviderConfig | None:
        return next((p for p in self.providers if p.name == name), None)

    def resolve_data_dir(self) -> Path:
        path = Path(self.app.data_dir)
        if not path.is_absolute():
            path = project_root() / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_database_url(self) -> str:
        """Absolutise relative SQLite paths so the CWD cannot move the database."""
        url = self.database.url
        prefix = "sqlite+aiosqlite:///"
        if url.startswith(prefix) and not url.startswith(prefix + "/"):
            relative = url[len(prefix) :]
            if relative.startswith("./"):
                relative = relative[2:]
            target = (project_root() / relative).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            return prefix + target.as_posix()
        return url


def load_assets(path: Path | None = None) -> AssetUniverse:
    """Load the asset universe. Missing file yields an empty universe, not a crash."""
    path = path or project_root() / "config" / "assets.yaml"
    if not path.exists():
        return AssetUniverse()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    default_quote = raw.get("default_quote", "USDT")
    entries = raw.get("assets") or []
    assets: list[AssetConfig] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"symbol": entry}
        entry.setdefault("quote", default_quote)
        assets.append(AssetConfig(**entry))
    return AssetUniverse(default_quote=default_quote, assets=assets)


def load_settings(
    config_file: Path | None = None,
    assets_file: Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build the fully resolved settings object.

    ``overrides`` take highest precedence and exist so tests can pin values without
    mutating the process environment.
    """
    if config_file is not None:
        os.environ["MIE_CONFIG_FILE"] = str(config_file)
    settings = Settings(**overrides)
    if "universe" not in overrides:
        settings.universe = load_assets(assets_file)
    settings._config_path = config_file
    if not settings.providers:
        raise ConfigError(
            "no providers configured — check config/default.yaml or MIE_PROVIDERS env"
        )
    return settings
