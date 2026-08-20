"""Structured logging.

Everything the engine emits is structured: an ingest run, a quality event, and a
provider failure all need to be queryable after the fact, and grepping prose does
not scale. Console rendering stays human-readable in development; production emits
JSON for shipping.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]

_configured = False


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Install the structlog pipeline. Safe to call repeatedly."""
    global _configured

    numeric = getattr(logging, level.upper(), logging.INFO)

    # Windows consoles still default to a legacy code page, which raises
    # UnicodeEncodeError on any non-ASCII character in a log line and turns a routine
    # message into a logging failure. Force UTF-8 and degrade rather than raise.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A non-reconfigurable stream (a pipe, a captured buffer) is fine as-is.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric, force=True)

    # Third-party libraries are chatty at INFO and drown the signal.
    for noisy in ("httpx", "httpcore", "asyncio", "aiosqlite"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    if json_output:
        processors += [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> Any:
    """Return a bound logger, configuring defaults on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
