"""Command-line interface.

The operational surface for Phase 1. Every command is a thin wrapper over the same
objects the service uses, so anything the CLI can do is reachable programmatically
and nothing here contains logic of its own.

    mie db init                       create the schema and register the universe
    mie providers                     probe provider health
    mie backfill BTC 1h --days 30     backfill one series
    mie backfill-all                  backfill the whole configured matrix
    mie poll --once                   run one live-poll tick
    mie run                           run the full ingestion service
    mie status                        coverage and freshness per series
    mie quality                       recent quality events and trust scores
    mie audit BTC 1h                  compare providers against each other
    mie features compute BTC 1h       compute features over stored history
    mie features show BTC 1h          show the latest feature vector
    mie state BTC                     multi-timeframe market state
    mie patterns measure              measure every detector against history
    mie patterns show                 which patterns earned predictive use
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import timedelta

import typer
from rich.console import Console
from rich.table import Table

from mie.config.settings import Settings, load_settings
from mie.core.logging import configure_logging, get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import IngestStatus
from mie.features.engine import _row_to_candle
from mie.ingestion.service import IngestionService
from mie.patterns.evaluation import PatternEvaluator
from mie.patterns.registry import PatternRegistry
from mie.state.engine import StateEngine
from mie.storage.repositories import (
    FeatureRepository,
    IngestRunRepository,
    OHLCVRepository,
    PatternStatsRepository,
    QualityRepository,
)

app = typer.Typer(
    name="mie",
    help="Crypto Market Intelligence Engine — analytical only, never executes trades.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database schema management.")
app.add_typer(db_app, name="db")
features_app = typer.Typer(help="Technical feature computation (Phase 2).")
app.add_typer(features_app, name="features")
patterns_app = typer.Typer(help="Pattern detection and statistical validation (Phase 4).")
app.add_typer(patterns_app, name="patterns")

console = Console()
log = get_logger(__name__)


def _settings(verbose: bool = False) -> Settings:
    settings = load_settings()
    configure_logging(
        "DEBUG" if verbose else settings.app.log_level, json_output=settings.app.log_json
    )
    return settings


def _tf(value: str) -> Timeframe:
    try:
        return Timeframe.parse(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


# --------------------------------------------------------------------------- db


@db_app.command("init")
def db_init(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Create tables (and TimescaleDB extras) and register assets and sources."""

    async def _run() -> None:
        settings = _settings(verbose)
        async with IngestionService(settings) as service:
            await service.bootstrap()
            console.print(f"[green]schema ready[/green] at {service.db.url}")
            console.print(
                f"registered {len(settings.universe.enabled())} assets, "
                f"{len(service.manager.providers)} providers"
            )

    asyncio.run(_run())


@db_app.command("reset")
def db_reset(
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Drop every table and recreate the schema. Destroys all stored data."""
    if not yes:
        typer.confirm("This deletes all stored market data. Continue?", abort=True)

    async def _run() -> None:
        settings = _settings(verbose)
        async with IngestionService(settings) as service:
            await service.db.drop_schema()
            await service.bootstrap()
            console.print("[yellow]database reset[/yellow]")

    asyncio.run(_run())


@db_app.command("info")
def db_info() -> None:
    """Show the resolved database target and connectivity."""

    async def _run() -> None:
        settings = _settings()
        async with IngestionService(settings) as service:
            reachable = await service.db.healthcheck()
            console.print(f"url      : {service.db.url}")
            console.print(f"dialect  : {service.db.dialect}")
            console.print(
                f"reachable: {'[green]yes[/green]' if reachable else '[red]no[/red]'}"
            )

    asyncio.run(_run())


# -------------------------------------------------------------------- providers


@app.command("providers")
def providers_health() -> None:
    """Probe every configured provider and report latency and capability."""

    async def _run() -> None:
        settings = _settings()
        async with IngestionService(settings) as service:
            health = await service.manager.health()
            table = Table(title="Provider health", header_style="bold")
            table.add_column("provider")
            table.add_column("priority", justify="right")
            table.add_column("status")
            table.add_column("latency", justify="right")
            table.add_column("timeframes")
            table.add_column("extras")
            for provider in service.manager.providers:
                state = health.get(provider.name)
                caps = provider.capabilities
                extras = ", ".join(
                    name
                    for name, on in (
                        ("funding", caps.supports_funding),
                        ("OI", caps.supports_open_interest),
                        ("global", caps.supports_global_metrics),
                    )
                    if on
                )
                table.add_row(
                    provider.name,
                    str(provider.config.priority),
                    "[green]ok[/green]" if state and state.ok else f"[red]{state.error if state else 'unknown'}[/red]",
                    f"{state.latency_ms:.0f}ms" if state and state.latency_ms else "-",
                    ", ".join(sorted(t.value for t in caps.timeframes)) or "none",
                    extras or "-",
                )
            console.print(table)

    asyncio.run(_run())


@app.command("assets")
def list_assets() -> None:
    """List the configured observation universe."""
    settings = _settings()
    table = Table(title="Asset universe", header_style="bold")
    table.add_column("symbol")
    table.add_column("name")
    table.add_column("tier", justify="right")
    table.add_column("quote")
    table.add_column("overrides")
    for asset in settings.universe.assets:
        table.add_row(
            asset.symbol,
            asset.name,
            str(asset.tier),
            asset.quote,
            ", ".join(f"{k}={v}" for k, v in asset.overrides.items()) or "-",
        )
    console.print(table)


# --------------------------------------------------------------------- ingestion


@app.command("backfill")
def backfill(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Argument(..., help="One of 1m 5m 15m 30m 1h 4h 12h 1d 1w."),
    days: int | None = typer.Option(None, "--days", help="History depth; defaults to config."),
    source: str | None = typer.Option(None, "--source", help="Force one provider."),
    force: bool = typer.Option(False, "--force", help="Re-fetch instead of resuming."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Backfill one series."""

    async def _run() -> None:
        settings = _settings(verbose)
        frame = _tf(timeframe)
        async with IngestionService(settings) as service:
            await service.bootstrap()
            start = utcnow() - timedelta(days=days) if days else None
            result = await service.backfill_engine.backfill(
                asset, frame, start=start, source=source, force=force
            )
            _print_result(result)

    asyncio.run(_run())


@app.command("backfill-all")
def backfill_all(
    assets: str | None = typer.Option(None, "--assets", help="Comma-separated; default all."),
    timeframes: str | None = typer.Option(None, "--timeframes", help="Comma-separated."),
    source: str | None = typer.Option(None, "--source"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Backfill the whole configured asset × timeframe matrix."""

    async def _run() -> None:
        settings = _settings(verbose)
        symbols = [a.strip().upper() for a in assets.split(",")] if assets else None
        frames = [_tf(t.strip()) for t in timeframes.split(",")] if timeframes else None
        async with IngestionService(settings) as service:
            await service.bootstrap()
            results = await service.backfill_all(symbols, frames, force=force, source=source)

            table = Table(title="Backfill summary", header_style="bold")
            for column in ("asset", "tf", "source", "status", "written", "rejected", "events"):
                table.add_column(column, justify="right" if column in ("written", "rejected", "events") else "left")
            for result in results:
                table.add_row(
                    result.asset,
                    str(result.timeframe or "-"),
                    result.source or "-",
                    _status_markup(result.status),
                    str(result.rows_written),
                    str(result.rows_rejected),
                    str(len(result.quality_events)),
                )
            console.print(table)
            console.print(
                f"[bold]{sum(r.rows_written for r in results)}[/bold] candles written across "
                f"{len(results)} series"
            )

    asyncio.run(_run())


@app.command("poll")
def poll(
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the live poller (one tick with --once, otherwise until interrupted)."""

    async def _run() -> None:
        settings = _settings(verbose)
        async with IngestionService(settings) as service:
            await service.bootstrap()
            service.poller.watch_universe()
            if once:
                results = await service.poller.tick()
                table = Table(title="Poll tick", header_style="bold")
                for column in ("asset", "tf", "source", "status", "written"):
                    table.add_column(column)
                for result in results:
                    table.add_row(
                        result.asset,
                        str(result.timeframe or "-"),
                        result.source or "-",
                        _status_markup(result.status),
                        str(result.rows_written),
                    )
                console.print(table)
                return
            await _run_until_interrupted(service.poller.run(service._stop), service)

    asyncio.run(_run())


@app.command("run")
def run_service(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run the full ingestion service: live polling, derivatives, metrics, scoring."""

    async def _run() -> None:
        settings = _settings(verbose)
        async with IngestionService(settings) as service:
            await service.bootstrap()
            console.print("[green]ingestion service started[/green] — Ctrl-C to stop")
            await _run_until_interrupted(service.run(), service)
            console.print(f"stats: {service.stats}")

    asyncio.run(_run())


# ---------------------------------------------------------------- observability


@app.command("status")
def status(
    assets: str | None = typer.Option(None, "--assets"),
    timeframes: str | None = typer.Option(None, "--timeframes"),
) -> None:
    """Show stored coverage, completeness and freshness per series."""

    async def _run() -> None:
        settings = _settings()
        symbols = (
            [a.strip().upper() for a in assets.split(",")] if assets else settings.universe.symbols()
        )
        frames = (
            [_tf(t.strip()) for t in timeframes.split(",")]
            if timeframes
            else settings.ingestion.live_timeframes
        )
        async with IngestionService(settings) as service:
            table = Table(title="Stored coverage", header_style="bold")
            for column in ("asset", "tf", "rows", "first", "last", "complete", "age"):
                table.add_column(column, justify="right" if column in ("rows", "complete", "age") else "left")
            async with service.db.session() as session:
                repo = OHLCVRepository(session)
                quality = QualityRepository(session)
                scores = {
                    (s.asset, s.timeframe): s.score for s in await quality.all_scores()
                }
                for asset in symbols:
                    for frame in frames:
                        info = await repo.coverage(asset, frame)
                        if not info["rows"]:
                            table.add_row(asset, str(frame), "0", "-", "-", "-", "-")
                            continue
                        completeness = info["completeness"] or 0
                        table.add_row(
                            asset,
                            str(frame),
                            str(info["rows"]),
                            info["first"].strftime("%Y-%m-%d"),
                            info["last"].strftime("%Y-%m-%d %H:%M"),
                            _pct_markup(completeness),
                            _age_markup(info["age_s"], frame),
                        )
                runs = await IngestRunRepository(session).recent(limit=5)
            console.print(table)
            if scores:
                worst = sorted(scores.items(), key=lambda kv: kv[1])[:5]
                console.print(
                    "quality scores (worst): "
                    + ", ".join(f"{a}/{t}={v:.2f}" for (a, t), v in worst)
                )
            if runs:
                console.print("\n[bold]recent ingest runs[/bold]")
                for run in runs:
                    console.print(
                        f"  {run.started_at:%Y-%m-%d %H:%M} {run.job:9} {run.asset:5} "
                        f"{run.timeframe or '-':4} {run.status:8} written={run.rows_written}"
                    )

    asyncio.run(_run())


@app.command("quality")
def quality_report(
    hours: int = typer.Option(24, "--hours"),
    severity: str | None = typer.Option(None, "--severity", help="info | warning | error"),
    limit: int = typer.Option(25, "--limit"),
) -> None:
    """Show recent data-quality events and the trust scores derived from them."""

    async def _run() -> None:
        settings = _settings()
        async with IngestionService(settings) as service:
            async with service.db.session() as session:
                repo = QualityRepository(session)
                counts = await repo.event_counts(hours=hours)
                events = await repo.recent_events(hours=hours, severity=severity, limit=limit)
                scores = await repo.all_scores()

            if counts:
                console.print(f"[bold]event counts (last {hours}h)[/bold]")
                for event_type, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                    console.print(f"  {count:5}  {event_type}")
            else:
                console.print(f"no quality events in the last {hours}h")

            if events:
                table = Table(title=f"Recent events (last {hours}h)", header_style="bold")
                for column in ("when", "severity", "type", "scope", "message"):
                    table.add_column(column)
                for event in events:
                    table.add_row(
                        event.detected_at.strftime("%m-%d %H:%M"),
                        _severity_markup(event.severity),
                        event.event_type,
                        f"{event.source}/{event.asset or '-'}/{event.timeframe or '-'}",
                        (event.message or "")[:80],
                    )
                console.print(table)

            if scores:
                table = Table(title="Source trust scores", header_style="bold")
                for column in ("source", "asset", "tf", "score", "events", "why"):
                    table.add_column(column)
                for score in scores[:20]:
                    reasons = (score.details or {}).get("reasons") or []
                    table.add_row(
                        score.source,
                        score.asset,
                        score.timeframe,
                        _score_markup(score.score),
                        str(score.events_in_window),
                        ", ".join(reasons)[:60] or "clean",
                    )
                console.print(table)

    asyncio.run(_run())


@app.command("audit")
def audit(
    asset: str = typer.Argument(...),
    timeframe: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit"),
    tolerance: float = typer.Option(0.5, "--tolerance", help="Allowed inter-venue spread, %."),
) -> None:
    """Cross-check providers against each other for the same recent window."""

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        async with IngestionService(settings) as service:
            events = await service.manager.compare_sources(
                asset, frame, limit=limit, tolerance_pct=tolerance
            )
            if not events:
                console.print(
                    f"[green]providers agree[/green] on {asset.upper()} {frame} "
                    f"within {tolerance}%"
                )
                return
            async with service.db.session() as session:
                await QualityRepository(session).record_events(events)
            for event in events:
                console.print(f"[yellow]{event}[/yellow]")
            console.print(f"\n{len(events)} discrepancy event(s) recorded")

    asyncio.run(_run())


# ------------------------------------------------------------------- formatting


def _print_result(result) -> None:
    console.print(f"[bold]{result.summary()}[/bold]")
    if result.error:
        console.print(f"[red]error:[/red] {result.error}")
    by_severity: dict[str, int] = {}
    for event in result.quality_events:
        by_severity[str(event.severity)] = by_severity.get(str(event.severity), 0) + 1
    if by_severity:
        console.print("quality events: " + ", ".join(f"{k}={v}" for k, v in by_severity.items()))
    for event in result.quality_events[:10]:
        console.print(f"  {_severity_markup(str(event.severity))} {event.event_type}: {event.message}")


def _status_markup(status: IngestStatus | str) -> str:
    colours = {
        "success": "green",
        "partial": "yellow",
        "failed": "red",
        "skipped": "dim",
    }
    text = str(status)
    return f"[{colours.get(text, 'white')}]{text}[/]"


def _severity_markup(severity: str) -> str:
    colours = {"info": "dim", "warning": "yellow", "error": "red"}
    return f"[{colours.get(severity, 'white')}]{severity}[/]"


def _score_markup(score: float) -> str:
    colour = "green" if score >= 0.75 else "yellow" if score >= 0.35 else "red"
    return f"[{colour}]{score:.2f}[/]"


def _pct_markup(ratio: float) -> str:
    colour = "green" if ratio >= 0.995 else "yellow" if ratio >= 0.95 else "red"
    return f"[{colour}]{ratio:.1%}[/]"


def _age_markup(age_s: float, timeframe: Timeframe) -> str:
    """Freshness relative to the timeframe — 30 minutes is fine on 1d, stale on 1m."""
    ratio = age_s / timeframe.seconds
    colour = "green" if ratio <= 2 else "yellow" if ratio <= 5 else "red"
    if age_s < 3600:
        text = f"{age_s / 60:.0f}m"
    elif age_s < 86400:
        text = f"{age_s / 3600:.1f}h"
    else:
        text = f"{age_s / 86400:.1f}d"
    return f"[{colour}]{text}[/]"


async def _run_until_interrupted(coro, service: IngestionService) -> None:
    """Run a long-lived coroutine, stopping cleanly on SIGINT/SIGTERM.

    Signal handlers are installed where the platform supports them and fall back to
    KeyboardInterrupt on Windows, where asyncio does not implement add_signal_handler.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(coro)
    # Held so the shutdown task cannot be garbage-collected before it completes.
    pending: set[asyncio.Task] = set()

    def _request_stop() -> None:
        console.print("\n[yellow]stopping...[/yellow]")
        stopper = asyncio.ensure_future(service.stop())
        pending.add(stopper)
        stopper.add_done_callback(pending.discard)

    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows asyncio has no add_signal_handler; KeyboardInterrupt covers it there.
        with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
            loop.add_signal_handler(sig, _request_stop)

    try:
        await task
    except KeyboardInterrupt:
        await service.stop()
        task.cancel()


@features_app.command("compute")
def features_compute(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Argument(..., help="One of 1m 5m 15m 30m 1h 4h 12h 1d 1w."),
    source: str = typer.Option("binance", "--source", help="Which venue's series."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compute and store feature vectors across stored history for one series."""

    async def _run() -> None:
        settings = _settings(verbose)
        frame = _tf(timeframe)
        async with IngestionService(settings) as service:
            written = await service.features.backfill(asset, frame, source)
            if not written:
                console.print(
                    f"[yellow]no features written[/yellow] — is there stored "
                    f"{asset.upper()} {frame} history from {source}?"
                )
                return
            console.print(
                f"[green]{written}[/green] feature vectors written for "
                f"{asset.upper()} {frame} ({source})"
            )

    asyncio.run(_run())


@features_app.command("compute-all")
def features_compute_all(
    timeframes: str | None = typer.Option(None, "--timeframes", help="Comma-separated."),
    source: str = typer.Option("binance", "--source"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compute features for the whole configured universe."""

    async def _run() -> None:
        settings = _settings(verbose)
        frames = (
            [_tf(t.strip()) for t in timeframes.split(",")]
            if timeframes
            else settings.ingestion.live_timeframes
        )
        async with IngestionService(settings) as service:
            table = Table(title="Feature computation", header_style="bold")
            for column in ("asset", "tf", "vectors"):
                table.add_column(column, justify="right" if column == "vectors" else "left")
            total = 0
            for asset in settings.universe.enabled():
                for frame in frames:
                    written = await service.features.backfill(asset.symbol, frame, source)
                    total += written
                    table.add_row(asset.symbol, str(frame), str(written))
            console.print(table)
            console.print(f"[bold]{total}[/bold] feature vectors written")

    asyncio.run(_run())


@features_app.command("show")
def features_show(
    asset: str = typer.Argument(...),
    timeframe: str = typer.Argument(...),
    source: str | None = typer.Option(None, "--source"),
) -> None:
    """Show the most recent stored feature vector for a series."""

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        async with IngestionService(settings) as service:
            async with service.db.session() as session:
                repo = FeatureRepository(session)
                latest = await repo.latest(asset, frame, source=source)
                stored = await repo.count(asset, frame, source=source)
            if latest is None:
                console.print(f"[yellow]no features stored[/yellow] for {asset.upper()} {frame}")
                return

            console.print(
                f"[bold]{asset.upper()} {frame}[/bold]  as of "
                f"{latest.open_time:%Y-%m-%d %H:%M} UTC  "
                f"(v{latest.version}, {stored} vectors stored)"
            )
            table = Table(header_style="bold")
            table.add_column("feature")
            table.add_column("value", justify="right")
            for key in sorted(latest.payload):
                value = latest.payload[key]
                table.add_row(key, f"{value:,.6g}" if isinstance(value, float) else str(value))
            console.print(table)

    asyncio.run(_run())


@app.command("state")
def market_state(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframes: str | None = typer.Option(None, "--timeframes", help="Comma-separated."),
    source: str = typer.Option("binance", "--source"),
    save: bool = typer.Option(False, "--save", help="Persist the computed state."),
) -> None:
    """Show the hierarchical multi-timeframe market state for an asset."""

    async def _run() -> None:
        settings = _settings()
        frames = [_tf(t.strip()) for t in timeframes.split(",")] if timeframes else None
        async with IngestionService(settings) as service:
            engine = StateEngine(service.db, settings, source=source, timeframes=frames)
            state = await engine.build(asset, persist=save)

            console.print(f"\n[bold]{state.asset}[/bold]  {state.as_of:%Y-%m-%d %H:%M} UTC")
            console.print(
                f"bias [bold]{state.bias}[/bold] | {state.alignment} | regime "
                f"[bold]{state.regime}[/bold]"
            )
            console.print(
                f"agreement {_pct_markup(state.agreement)} | confidence "
                f"{_score_markup(state.confidence)} | data quality "
                f"{_score_markup(state.data_quality)}\n"
            )

            table = Table(title="Timeframe hierarchy", header_style="bold")
            for column in ("tf", "direction", "strength", "confidence", "as of"):
                table.add_column(
                    column, justify="right" if column in ("strength", "confidence") else "left"
                )
            for level in state.timeframes:
                table.add_row(
                    str(level.timeframe),
                    _direction_markup(level.direction),
                    f"{level.strength:.2f}",
                    _score_markup(level.confidence),
                    f"{level.as_of:%m-%d %H:%M}",
                )
            console.print(table)

            console.print(f"\n[bold]Interpretation[/bold]\n{state.interpretation}\n")

            leading = next((s for s in state.timeframes if s.is_usable), None)
            if leading and leading.evidence:
                console.print(f"[bold]Why ({leading.timeframe})[/bold]")
                for item in leading.evidence[:6]:
                    console.print(f"  [green]+[/green] {item}")
                for item in leading.counter_evidence[:4]:
                    console.print(f"  [red]-[/red] {item}")
            if state.conflicts:
                console.print("\n[bold]Conflicts[/bold]")
                for conflict in state.conflicts:
                    console.print(f"  [yellow]! {conflict}[/yellow]")

            console.print(
                "\n[dim]Analytical assessment only - probabilities and scenarios, "
                "never guarantees. Not investment advice.[/dim]"
            )

    asyncio.run(_run())


def _direction_markup(direction: object) -> str:
    """Colour a direction label: green for up, red for down, dim for neutral."""
    text = str(direction)
    if "up" in text:
        return f"[green]{text}[/green]"
    if "down" in text:
        return f"[red]{text}[/red]"
    return f"[dim]{text}[/dim]"


@patterns_app.command("measure")
def patterns_measure(
    assets: str | None = typer.Option(None, "--assets", help="Comma-separated."),
    timeframes: str | None = typer.Option(None, "--timeframes", help="Comma-separated."),
    source: str = typer.Option("binance", "--source"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Measure every detector against history and record which ones earn predictive use.

    Each pattern is scored against the *unconditional* outcome rate over the same
    sample, not against a coin flip, and the whole sweep is corrected for multiple
    comparisons. Patterns that fail are withheld from the predictive path entirely.
    """

    async def _run() -> None:
        settings = _settings(verbose)
        symbols = (
            [a.strip().upper() for a in assets.split(",")]
            if assets
            else settings.universe.symbols()
        )
        frames = (
            [_tf(t.strip()) for t in timeframes.split(",")]
            if timeframes
            else [Timeframe.H1, Timeframe.H4]
        )
        evaluator = PatternEvaluator()
        registry = PatternRegistry()

        async with IngestionService(settings) as service:
            for asset in symbols:
                for frame in frames:
                    async with service.db.session() as session:
                        rows = await OHLCVRepository(session).fetch(
                            asset, frame, source=source
                        )
                    if len(rows) < 500:
                        continue
                    candles = [_row_to_candle(r, asset, frame, source) for r in rows]
                    result = evaluator.evaluate(candles, asset, frame)
                    registry.extend(result.stats)
                    console.print(
                        f"  {asset:5} {frame!s:4} {len(candles):>6} bars, "
                        f"{len(result.detections):>5} detections, "
                        f"{len(result.informative)}/{len(result.stats)} informative"
                    )

            if not len(registry):
                console.print("[yellow]no series had enough history to measure[/yellow]")
                return

            async with service.db.session() as session:
                await PatternStatsRepository(session).upsert_many(
                    registry.admitted() + registry.rejected()
                )

        admitted = registry.admitted()
        console.print(
            f"\n[bold]{len(admitted)}[/bold] of [bold]{len(registry)}[/bold] measured "
            f"pattern/asset/timeframe/horizon combinations are informative"
        )
        if admitted:
            table = Table(title="Admitted to the predictive path", header_style="bold")
            for column in ("pattern", "asset", "tf", "h", "n", "rate", "baseline", "edge", "p"):
                table.add_column(column)
            for stat in admitted:
                e = stat.estimate
                table.add_row(
                    str(stat.kind), stat.asset, str(stat.timeframe),
                    str(stat.horizon_bars), str(e.trials),
                    f"{e.rate:.1%}", f"{e.baseline:.1%}",
                    f"[green]{e.edge:+.1%}[/green]", f"{e.p_value:.4f}",
                )
            console.print(table)
        console.print(
            f"[dim]{len(registry.rejected())} withheld: not distinguishable from the "
            f"market's own behaviour after correcting for multiple comparisons.[/dim]"
        )

    asyncio.run(_run())


@patterns_app.command("show")
def patterns_show(
    asset: str | None = typer.Option(None, "--asset"),
    all_results: bool = typer.Option(False, "--all", help="Include withheld patterns."),
) -> None:
    """Show which patterns have earned the right to influence predictions."""

    async def _run() -> None:
        settings = _settings()
        async with IngestionService(settings) as service, service.db.session() as session:
            repo = PatternStatsRepository(session)
            rows = (
                await repo.for_asset(asset)
                if asset
                else await repo.all_stats(informative_only=not all_results)
            )
        if not rows:
            console.print(
                "[yellow]no measurements stored[/yellow] - run `mie patterns measure` first"
            )
            return

        table = Table(title="Pattern evidence", header_style="bold")
        for column in ("pattern", "asset", "tf", "h", "n", "rate", "base", "edge", "p", "verdict"):
            table.add_column(column)
        for row in rows[:40]:
            colour = "green" if row.informative else "dim"
            table.add_row(
                row.kind, row.asset, row.timeframe, str(row.horizon_bars),
                str(row.occurrences), f"{row.rate:.1%}", f"{row.baseline:.1%}",
                f"[{colour}]{row.edge:+.1%}[/{colour}]", f"{row.p_value:.4f}",
                f"[{colour}]{row.verdict}[/{colour}]",
            )
        console.print(table)
        console.print(
            "[dim]Only patterns marked informative may influence a prediction. "
            "Everything else is descriptive only.[/dim]"
        )

    asyncio.run(_run())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
