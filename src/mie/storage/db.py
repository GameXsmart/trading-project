"""Async engine, session management, and schema bootstrap.

One ``Database`` instance owns the engine for a process. Dialect differences are
confined to this module: everything above it writes plain SQLAlchemy and works on
both SQLite (dev/test) and TimescaleDB (production).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mie.config.settings import Settings
from mie.core.errors import StorageError
from mie.core.logging import get_logger
from mie.storage.models import Base

log = get_logger(__name__)

__all__ = ["Database"]


class Database:
    """Owns the async engine and hands out sessions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = settings.resolved_database_url()
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def url(self) -> str:
        return self._url

    @property
    def dialect(self) -> str:
        return "postgresql" if self._settings.database.is_postgres else "sqlite"

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = self._create_engine()
            self._sessionmaker = async_sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )
        return self._engine

    def _create_engine(self) -> AsyncEngine:
        cfg = self._settings.database
        kwargs: dict[str, object] = {"echo": cfg.echo, "future": True}
        if cfg.is_postgres:
            kwargs |= {
                "pool_size": cfg.pool_size,
                "max_overflow": cfg.max_overflow,
                "pool_pre_ping": True,
            }
        engine = create_async_engine(self._url, **kwargs)

        if cfg.is_sqlite:
            # SQLite defaults are wrong for a write-heavy ingest loop: the rollback
            # journal serialises readers against the writer and blocks instantly on
            # contention. WAL plus a busy timeout makes concurrent polling workable.
            @event.listens_for(engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.close()

        log.debug("engine_created", dialect=self.dialect, url=_redact(self._url))
        return engine

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None

    # ------------------------------------------------------------------- sessions

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commit on success, roll back on any exception."""
        _ = self.engine  # force lazy initialisation of engine + sessionmaker
        assert self._sessionmaker is not None
        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # --------------------------------------------------------------------- schema

    async def create_schema(self) -> None:
        """Create tables, then apply TimescaleDB extras when on Postgres.

        ``create_all`` is adequate while the schema is young; the moment a
        destructive migration is needed this becomes Alembic. That threshold is
        documented in docs/data-model.md rather than pre-solved here.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("schema_created", dialect=self.dialect, tables=len(Base.metadata.tables))

        if self._settings.database.is_postgres and self._settings.database.apply_timescale:
            await self._apply_timescale()

    async def _apply_timescale(self) -> None:
        """Convert time-series tables into hypertables with compression policies.

        Failure is logged, not raised: a plain Postgres without the extension is a
        perfectly usable (if slower) target, and refusing to start would be worse.
        """
        script = Path(__file__).resolve().parents[3] / "sql" / "timescale.sql"
        if not script.exists():
            log.warning("timescale_script_missing", path=str(script))
            return
        statements = [s.strip() for s in script.read_text(encoding="utf-8").split(";") if s.strip()]
        applied = 0
        async with self.engine.begin() as conn:
            for statement in statements:
                if statement.lstrip().startswith("--"):
                    continue
                try:
                    await conn.execute(text(statement))
                    applied += 1
                except SQLAlchemyError as exc:
                    log.warning(
                        "timescale_statement_skipped",
                        error=str(exc).splitlines()[0][:200],
                        statement=statement.splitlines()[0][:120],
                    )
        log.info("timescale_applied", statements=applied)

    async def drop_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        log.warning("schema_dropped", dialect=self.dialect)

    # --------------------------------------------------------------------- health

    async def healthcheck(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            log.error("database_unreachable", error=str(exc)[:300])
            return False

    async def ensure_ready(self) -> None:
        if not await self.healthcheck():
            raise StorageError(f"database not reachable at {_redact(self._url)}")


def _redact(url: str) -> str:
    """Strip credentials before a URL reaches a log line."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"
