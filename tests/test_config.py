"""Configuration loading and layering."""

from __future__ import annotations

from pathlib import Path

import pytest

from mie.config.settings import Settings, load_assets, load_settings, project_root
from mie.core.errors import ConfigError
from mie.core.timeframes import Timeframe


class TestShippedConfig:
    """The checked-in configuration must be valid — a broken default.yaml is a
    broken clone."""

    def test_default_config_loads(self) -> None:
        settings = load_settings()
        assert settings.providers
        assert settings.universe.symbols()
        assert Timeframe.H1 in settings.ingestion.timeframes

    def test_shipped_universe_covers_the_requested_assets(self) -> None:
        symbols = set(load_settings().universe.symbols())
        required = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT"}
        assert required <= symbols

    def test_providers_are_ordered_by_priority(self) -> None:
        priorities = [p.priority for p in load_settings().enabled_providers()]
        assert priorities == sorted(priorities)

    def test_no_secrets_are_committed(self) -> None:
        """A key in a checked-in file is a leaked key."""
        for name in ("default.yaml", "assets.yaml"):
            text = (project_root() / "config" / name).read_text(encoding="utf-8").lower()
            for marker in ("api_key:", "secret:", "password:", "token:"):
                assert marker not in text


class TestLayering:
    def test_yaml_values_reach_the_model(self, tmp_path: Path) -> None:
        config = tmp_path / "custom.yaml"
        config.write_text(
            "app:\n  log_level: DEBUG\n"
            "ingestion:\n  poll_interval_s: 99\n"
            "providers:\n  - name: fake\n    priority: 5\n",
            encoding="utf-8",
        )
        settings = load_settings(config_file=config)
        assert settings.app.log_level == "DEBUG"
        assert settings.ingestion.poll_interval_s == 99
        assert settings.providers[0].name == "fake"

    def test_environment_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "custom.yaml"
        config.write_text(
            "app:\n  log_level: INFO\nproviders:\n  - name: fake\n", encoding="utf-8"
        )
        monkeypatch.setenv("MIE_APP__LOG_LEVEL", "ERROR")
        settings = load_settings(config_file=config)
        assert settings.app.log_level == "ERROR"

    def test_missing_providers_is_a_configuration_error(self, tmp_path: Path) -> None:
        config = tmp_path / "empty.yaml"
        config.write_text("app:\n  log_level: INFO\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="no providers"):
            load_settings(config_file=config)

    def test_malformed_yaml_is_reported_clearly(self, tmp_path: Path) -> None:
        config = tmp_path / "bad.yaml"
        config.write_text("app: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_settings(config_file=config)

    def test_unknown_keys_are_rejected(self, tmp_path: Path) -> None:
        """A typo in a config key should fail loudly, not be silently ignored."""
        config = tmp_path / "typo.yaml"
        config.write_text(
            "ingestion:\n  poll_intervel_s: 30\nproviders:\n  - name: fake\n",
            encoding="utf-8",
        )
        with pytest.raises(Exception, match=r"poll_intervel_s|extra"):
            load_settings(config_file=config)


class TestTimeframeCoercion:
    def test_timeframe_strings_are_parsed(self, tmp_path: Path) -> None:
        config = tmp_path / "tf.yaml"
        config.write_text(
            "ingestion:\n"
            "  timeframes: ['15m', '4h']\n"
            "  backfill_days:\n    '15m': 5\n"
            "providers:\n  - name: fake\n",
            encoding="utf-8",
        )
        settings = load_settings(config_file=config)
        assert settings.ingestion.timeframes == [Timeframe.M15, Timeframe.H4]
        assert settings.ingestion.backfill_days[Timeframe.M15] == 5


class TestAssets:
    def test_asset_shorthand_and_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "assets.yaml"
        path.write_text(
            "default_quote: USDT\n"
            "assets:\n"
            "  - SOL\n"
            "  - symbol: btc\n"
            "    name: Bitcoin\n"
            "    tier: 1\n"
            "    overrides: { kraken: XBTUSD }\n",
            encoding="utf-8",
        )
        universe = load_assets(path)
        assert universe.symbols() == ["SOL", "BTC"]
        assert universe.get("BTC").overrides["kraken"] == "XBTUSD"
        assert universe.get("sol").quote == "USDT"

    def test_disabled_assets_are_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "assets.yaml"
        path.write_text(
            "assets:\n  - symbol: BTC\n  - symbol: ETH\n    enabled: false\n", encoding="utf-8"
        )
        assert load_assets(path).symbols() == ["BTC"]

    def test_missing_file_yields_an_empty_universe(self, tmp_path: Path) -> None:
        assert load_assets(tmp_path / "nope.yaml").assets == []


class TestPaths:
    def test_relative_sqlite_paths_are_absolutised(self, tmp_path: Path) -> None:
        """Otherwise the database moves when the working directory does."""
        settings = Settings(
            database={"url": "sqlite+aiosqlite:///./data/test.db"},
            providers=[{"name": "fake"}],
        )
        resolved = settings.resolved_database_url()
        assert "./" not in resolved
        assert resolved.endswith("data/test.db")

    def test_postgres_urls_are_left_alone(self) -> None:
        settings = Settings(
            database={"url": "postgresql+asyncpg://u:p@host:5432/mie"},
            providers=[{"name": "fake"}],
        )
        assert settings.resolved_database_url().startswith("postgresql+asyncpg://")
        assert settings.database.is_postgres
