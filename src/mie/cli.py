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
    mie similar BTC                   historical analogues of the current state
    mie news                          deduplicated, classified news feed
    mie news-impact BTC               measured impact of news on volatility
    mie evaluate BTC                  walk-forward model skill vs baseline
    mie calibrate BTC                 fit and judge per-model calibration
    mie ensemble BTC                  the ensemble, its confidence and the gate
    mie backtest BTC                  walk-forward folds with a leakage probe
    mie predict BTC                   record predictions for later scoring
    mie learn                         resolve, measure, reweight - and say what changed
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import timedelta
from statistics import median

import typer
from rich.console import Console
from rich.table import Table

from mie.backtest.harness import WalkForwardHarness
from mie.backtest.leakage import LeakageProbe, Verdict
from mie.backtest.windows import FoldScheme
from mie.config.settings import Settings, load_settings
from mie.core.logging import configure_logging, get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import IngestStatus
from mie.ensemble.calibration import CalibrationLibrary, reliability_diagram
from mie.ensemble.gate import SuperPredictionGate
from mie.ensemble.meta import EnsembleModel, SkillWeights
from mie.features.engine import _row_to_candle
from mie.ingestion.service import IngestionService
from mie.learning.loop import LearningLoop
from mie.learning.records import PredictionRecord, ResolvedOutcome, volatility_bucket
from mie.learning.weights import WeightKey
from mie.models.baselines import ClimatologyBaseline, PersistenceBaseline
from mie.models.evaluation import WalkForwardEvaluator, summarise_thresholds
from mie.models.predictors import ALL_MODELS
from mie.models.runner import ContextSource, build_contexts
from mie.models.types import Distribution, Horizon, Outcome
from mie.news.engine import NewsEngine
from mie.news.impact import ImpactValidator
from mie.patterns.evaluation import PatternEvaluator
from mie.patterns.registry import PatternRegistry
from mie.patterns.similarity import SimilarityEngine
from mie.state.engine import StateEngine
from mie.storage.repositories import (
    FeatureRepository,
    IngestRunRepository,
    NewsEventRepository,
    OHLCVRepository,
    PatternStatsRepository,
    PredictionRepository,
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


@app.command("similar")
def similar(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    horizons: str = typer.Option("12,48", "--horizons", help="Comma-separated, in bars."),
    source: str = typer.Option("binance", "--source"),
) -> None:
    """Find historical analogues of the current market state and report what followed.

    Answers "it has not looked like this before" when that is the truth, rather than
    returning the nearest available strangers.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        wanted = [int(h.strip()) for h in horizons.split(",")]

        async with IngestionService(settings) as service, service.db.session() as session:
            rows = await FeatureRepository(session).fetch(asset, frame, source=source)
        if len(rows) < 300:
            console.print(
                f"[yellow]not enough feature history[/yellow] for {asset.upper()} {frame} "
                f"- run `mie features compute {asset.upper()} {frame}` first"
            )
            return

        history = [(r.open_time, r.payload) for r in rows]
        closes = [r.payload.get("close", 0.0) for r in rows]
        engine = SimilarityEngine()

        console.print(
            f"\n[bold]{asset.upper()} {frame}[/bold]  as of "
            f"{rows[-1].open_time:%Y-%m-%d %H:%M} UTC  ({len(history)} historical states)\n"
        )
        for horizon in wanted:
            result = engine.search(
                history, closes, len(history) - 1, horizon, asset, frame
            )
            if not result.has_evidence:
                console.print(
                    f"  [yellow]+{horizon} bars: insufficient evidence[/yellow] - only "
                    f"{len(result.analogues)} comparable situations in "
                    f"{result.searched} searched (ceiling {result.distance_ceiling:.2f})"
                )
                continue
            estimate = result.estimate
            assert estimate is not None
            verdict = (
                "[green]differs from baseline[/green]"
                if result.is_informative
                else "[dim]matches baseline[/dim]"
            )
            console.print(
                f"  [bold]+{horizon} bars[/bold]: {len(result.analogues)} analogues rose "
                f"{estimate.rate:.0%} [{estimate.low:.0%}-{estimate.high:.0%}] "
                f"vs baseline {estimate.baseline:.0%} - {verdict}"
            )
            console.print(
                f"      median {result.median_return_pct:+.2f}%  "
                f"mean {result.mean_return_pct:+.2f}%  "
                f"worst analogue drawdown {result.worst_case_pct:+.1f}%"
            )

        console.print(
            "\n[dim]Analogues are historical situations that resembled this one on "
            "scale-free features. Resemblance is not causation, and a distribution "
            "of past outcomes is not a forecast. Not investment advice.[/dim]"
        )

    asyncio.run(_run())


@app.command("news")
def news(
    asset: str | None = typer.Option(None, "--asset", help="Filter to one asset."),
    limit: int = typer.Option(15, "--limit"),
    min_importance: float = typer.Option(0.0, "--min-importance"),
) -> None:
    """Fetch, deduplicate and classify current crypto news.

    One story republished by several outlets is reported once, with its coverage
    counted - which is the most honest importance signal available before any price
    data is consulted.
    """

    async def _run() -> None:
        _settings()
        async with NewsEngine() as engine:
            events = await engine.fetch_events()
        if not events:
            console.print("[yellow]no recent news retrieved[/yellow]")
            return

        selected = NewsEngine.for_asset(events, asset) if asset else events
        selected = [e for e in selected if e.importance >= min_importance]
        selected = sorted(selected, key=lambda e: -e.importance)[:limit]

        overall, counted = NewsEngine.market_sentiment(events, asset)
        label = asset.upper() if asset else "market-wide"
        tone = "green" if overall > 0.1 else "red" if overall < -0.1 else "dim"
        console.print(
            f"\n[bold]{label} news sentiment[/bold] "
            f"[{tone}]{overall:+.3f}[/{tone}] across {counted} distinct stories "
            f"({len(events)} total, {sum(1 for e in events if e.coverage > 1)} "
            f"merged across outlets)\n"
        )

        table = Table(header_style="bold")
        for column in ("imp", "cat", "sentiment", "outlets", "assets", "story"):
            table.add_column(column)
        for event in selected:
            colour = (
                "green" if event.sentiment_score > 0.1
                else "red" if event.sentiment_score < -0.1
                else "dim"
            )
            table.add_row(
                f"{event.importance:.2f}",
                str(event.category)[:14],
                f"[{colour}]{str(event.sentiment)[:13]}[/{colour}]",
                str(event.coverage),
                ",".join(event.assets[:2]) or "-",
                (("[recycled] " if event.is_recycled else "") + event.title)[:62],
            )
        console.print(table)
        console.print(
            "\n[dim]Sentiment is a reading of the text, not a forecast of prices. "
            "Coverage counts distinct outlets. Not investment advice.[/dim]"
        )

    asyncio.run(_run())


@app.command("news-impact")
def news_impact(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    source: str = typer.Option("binance", "--source"),
    save: bool = typer.Option(True, "--save/--no-save", help="Store fetched stories."),
) -> None:
    """Measure what actually followed news events, by category.

    Impact is measured, never asserted: each category is compared against the
    unconditional rate of elevated volatility over the same price history, and a
    category without enough events reports insufficient evidence rather than a number.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)

        async with NewsEngine() as engine:
            events = await engine.fetch_events()

        async with IngestionService(settings) as service:
            if save and events:
                async with service.db.session() as session:
                    await NewsEventRepository(session).upsert_many(events)
            async with service.db.session() as session:
                stored_total = await NewsEventRepository(session).count()
                rows = await OHLCVRepository(session).fetch(asset, frame, source=source)

        if not rows:
            console.print(f"[yellow]no price history[/yellow] for {asset.upper()} {frame}")
            return

        candles = [_row_to_candle(r, asset, frame, source) for r in rows]
        results = ImpactValidator().validate(events, candles, asset, frame)
        relevant = [
            e for e in events if e.relevance_for(asset) >= 0.5 and not e.is_recycled
        ]

        console.print(
            f"\n[bold]{asset.upper()} news impact[/bold] - {len(relevant)} relevant "
            f"stories in this fetch, {stored_total} stored in total\n"
        )
        if not results:
            console.print(
                "[yellow]no category had enough paired events to measure[/yellow]"
            )
        else:
            table = Table(header_style="bold")
            for column in ("category", "h", "n", "vol ratio", "elevated", "baseline", "verdict"):
                table.add_column(column)
            for m in sorted(results, key=lambda m: -m.events):
                elevated = f"{m.elevated.rate:.0%}" if m.elevated else "-"
                baseline = f"{m.elevated.baseline:.0%}" if m.elevated else "-"
                colour = "green" if m.moves_volatility or m.moves_direction else "dim"
                table.add_row(
                    str(m.category), str(m.horizon_hours), str(m.events),
                    f"{m.median_volatility_ratio:.2f}x", elevated, baseline,
                    f"[{colour}]{m.verdict}[/{colour}]",
                )
            console.print(table)

        console.print(
            "\n[dim]RSS feeds carry only a few days of history, so this measurement "
            "grows more meaningful as stored events accumulate. Categories below the "
            "evidence threshold report insufficient evidence rather than a guess.[/dim]"
        )

    asyncio.run(_run())


async def _evaluation_contexts(
    settings: Settings, asset: str, frame: Timeframe, horizon: int, source: str
) -> list | None:
    """Load stored history and build non-overlapping evaluation points.

    Shared by ``evaluate``, ``calibrate`` and ``ensemble`` so that all three see
    exactly the same data. Three commands each assembling their own view of history is
    three chances for one of them to be subtly wrong about what was knowable when.
    """
    async with IngestionService(settings) as service, service.db.session() as session:
        rows = await OHLCVRepository(session).fetch(asset, frame, source=source)
        features = await FeatureRepository(session).fetch(asset, frame, source=source)
        peers = {}
        for other in settings.universe.symbols():
            if other == asset.upper():
                continue
            peer_rows = await OHLCVRepository(session).fetch(
                other, frame, source=source, limit=5000
            )
            if peer_rows:
                peers[other] = [_row_to_candle(r, other, frame, source) for r in peer_rows]

    if len(rows) < 600:
        console.print(
            f"[yellow]not enough history[/yellow] for {asset.upper()} {frame} "
            f"({len(rows)} bars; need 600+)"
        )
        return None

    candles = [_row_to_candle(r, asset, frame, source) for r in rows]
    history = [(f.open_time, f.payload) for f in features]
    contexts = build_contexts(
        ContextSource(asset, frame, candles, history, peers=peers),
        Horizon(bars=horizon, timeframe=frame),
        warmup=450,
    )
    if not contexts:
        console.print("[yellow]no evaluation points could be built[/yellow]")
        return None
    return contexts


@app.command("evaluate")
def evaluate(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    horizon: int = typer.Option(12, "--horizon", help="Forecast horizon, in bars."),
    baseline: str = typer.Option(
        "climatology", "--baseline", help="climatology | persistence"
    ),
    source: str = typer.Option("binance", "--source"),
    show_slices: bool = typer.Option(False, "--slices", help="Show every regime slice."),
) -> None:
    """Walk history forward and score every model against a baseline.

    A model passes only by beating the baseline with statistical significance on the
    paired Brier differences, after correcting across every slice tested. Positive
    skill alone is not a pass: with dozens of slices, some model posts a positive
    number by luck.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        chosen = (
            PersistenceBaseline() if baseline.startswith("pers") else ClimatologyBaseline()
        )

        contexts = await _evaluation_contexts(settings, asset, frame, horizon, source)
        if contexts is None:
            return

        report = WalkForwardEvaluator(chosen).evaluate([m() for m in ALL_MODELS], contexts)
        balance = summarise_thresholds(contexts)

        console.print(
            f"\n[bold]{asset.upper()} {frame} +{horizon} bars[/bold] vs "
            f"[bold]{chosen.model_id}[/bold] - {len(contexts)} non-overlapping points"
        )
        console.print(
            f"class balance: up {balance['up']:.0%} / flat {balance['flat']:.0%} / "
            f"down {balance['down']:.0%}, threshold "
            f"{balance['median_threshold_pct']:.2f}%\n"
        )

        table = Table(header_style="bold")
        for column in ("model", "n", "brier", "baseline", "skill", "p", "conf", "verdict"):
            table.add_column(column)
        shown = (
            report.scores if show_slices else [s for s in report.scores if s.regime == "all"]
        )
        for score in sorted(shown, key=lambda s: -s.skill):
            colour = "green" if score.beats_baseline else "dim"
            label = (
                score.model_id if score.regime == "all" else f"{score.model_id}/{score.regime}"
            )
            table.add_row(
                label, str(score.predictions), f"{score.brier:.4f}",
                f"{score.baseline_brier:.4f}",
                f"[{colour}]{score.skill:+.4f}[/{colour}]",
                f"{score.p_value:.4f}", f"{score.mean_confidence:.2f}",
                f"[{colour}]{score.verdict[:38]}[/{colour}]",
            )
        console.print(table)

        passing = sorted(report.passing_models())
        failing = sorted(report.failing_models())
        console.print(
            f"\n[bold]PASS[/bold] ({len(passing)}): "
            f"[green]{', '.join(passing) or 'none'}[/green]"
        )
        console.print(
            f"[bold]FAIL[/bold] ({len(failing)}): [dim]{', '.join(failing) or 'none'}[/dim]"
        )
        console.print(
            "\n[dim]A model ships only by beating the baseline with significance across "
            "the whole family of slices tested. Climatology is the honest bar; beating "
            "persistence proves little, since even abstaining beats it.[/dim]"
        )

    asyncio.run(_run())


@app.command("calibrate")
def calibrate(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    horizon: int = typer.Option(12, "--horizon", help="Forecast horizon, in bars."),
    source: str = typer.Option("binance", "--source"),
) -> None:
    """Fit per-model calibration and report whether it actually helped.

    Each curve is fitted on an earlier window and judged on a later one. Curves that
    do not improve held-out calibration are discarded and the model's own numbers kept,
    so this command routinely reports that calibration was *not* adopted.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        contexts = await _evaluation_contexts(settings, asset, frame, horizon, source)
        if contexts is None:
            return

        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [m() for m in ALL_MODELS], contexts
        )
        library = CalibrationLibrary()
        entries = [e for group in report.scored.values() for e in group]
        library.fit(entries)

        console.print(
            f"\n[bold]{asset.upper()} {frame} +{horizon} bars[/bold] - calibration from "
            f"{len(contexts)} non-overlapping points"
        )

        fitted = [r for r in library.records if r.curves]
        skipped = len(library.records) - len(fitted)
        table = Table(header_style="bold", box=None, pad_edge=False)
        for column in ("model", "regime", "n", "ECE in", "ECE out", "delta", "verdict"):
            table.add_column(column)
        for record in sorted(fitted, key=lambda r: (r.model_id, r.regime)):
            colour = "green" if record.improved else "dim"
            table.add_row(
                record.model_id, record.regime, str(record.samples),
                f"{record.ece_before:.4f}", f"{record.ece_after:.4f}",
                f"[{colour}]{record.improvement:+.4f}[/{colour}]",
                f"[{colour}]{'kept' if record.improved else 'discarded'}[/{colour}]",
            )
        console.print(table)
        if skipped:
            console.print(
                f"[dim]{skipped} further (model, regime) pairs had too little data to "
                f"fit at all.[/dim]"
            )

        # Reliability of what the panel says as a whole, before any calibration.
        pairs = [
            (e.prediction.distribution.probability(outcome), 1.0 if e.actual is outcome else 0.0)
            for e in entries
            if e.prediction.confidence > 0
            for outcome in Outcome
        ]
        diagram = reliability_diagram(pairs)
        console.print(f"\n[bold]raw panel reliability[/bold] - ECE {diagram.ece:.4f}")
        bins = Table(header_style="bold")
        for column in ("stated", "n", "observed", "95% interval", "consistent"):
            bins.add_column(column)
        for entry in diagram.bins:
            if entry.count < 20:
                continue
            mark = "[green]yes[/green]" if entry.contains_nominal else "[red]no[/red]"
            bins.add_row(
                f"{entry.lower:.1f}-{entry.upper:.1f}", str(entry.count),
                f"{entry.observed:.3f}",
                f"[{entry.observed_low:.3f}, {entry.observed_high:.3f}]", mark,
            )
        console.print(bins)
        # 100 rather than the default 20: a headline claim about discrimination
        # should not rest on a bin holding thirty observations.
        populated = diagram.populated(100)
        if len(populated) >= 2:
            spread = populated[-1].observed - populated[0].observed
            stated = populated[-1].mean_predicted - populated[0].mean_predicted
            console.print(
                f"discrimination: across a stated range of {stated:.3f} the observed "
                f"frequency moves {spread:+.3f}"
            )
        console.print(
            f"\nusable records: [bold]{len(library.usable())}[/bold] of "
            f"{len(library.records)}"
        )
        console.print(
            "[dim]A curve is kept only if it improved calibration on data it was not "
            "fitted on. 'Discarded' means the fitted curve did worse out-of-sample than "
            "leaving the model's own numbers alone - isotonic regression has enough "
            "freedom to fit noise, and at these sample sizes it usually does.[/dim]"
        )

    asyncio.run(_run())


@app.command("ensemble")
def ensemble(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    horizon: int = typer.Option(12, "--horizon", help="Forecast horizon, in bars."),
    source: str = typer.Option("binance", "--source"),
    points: int = typer.Option(1, "--points", help="How many recent points to show."),
) -> None:
    """Run the full Phase 7 stack and show what, if anything, it will publish.

    Weights come only from models that beat climatology out-of-sample with significance
    across the whole family of slices tested. Where none do, the ensemble publishes
    nothing and says which condition failed.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        contexts = await _evaluation_contexts(settings, asset, frame, horizon, source)
        if contexts is None:
            return

        models = [m() for m in ALL_MODELS]
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(models, contexts)
        weights = SkillWeights.from_report(report)
        library = CalibrationLibrary()
        library.fit([e for group in report.scored.values() for e in group])

        engine = EnsembleModel(models, weights, library)
        gate = SuperPredictionGate()

        console.print(
            f"\n[bold]{asset.upper()} {frame} +{horizon} bars[/bold] - ensemble over "
            f"{len(contexts)} non-overlapping points"
        )
        console.print(f"[dim]{weights.summary()}[/dim]")
        console.print(
            f"[dim]calibration: {len(library.usable())} usable of "
            f"{len(library.records)} records[/dim]\n"
        )

        for context_, realised in contexts[-max(1, points) :]:
            result = engine.predict_detailed(context_)
            decision = gate.evaluate(result, library, [m.model_id for m in models])

            console.print(
                f"[bold]{context_.as_of:%Y-%m-%d %H:%M}[/bold] regime "
                f"[cyan]{context_.regime}[/cyan]"
            )
            if result.published:
                console.print(
                    f"  [green]{result.prediction.distribution}[/green] "
                    f"confidence {result.prediction.confidence:.0%}"
                )
            else:
                console.print("  [yellow]insufficient evidence[/yellow]")
                for reason in result.suppressed_because:
                    console.print(f"    [dim]- {reason}[/dim]")
            console.print(f"  [dim]{result.agreement.summary()}[/dim]")
            console.print(f"  [dim]{result.factors.explain()}[/dim]")

            for check in decision.checks:
                mark = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
                console.print(f"    {mark}  {check.name}: [dim]{check.detail}[/dim]")
            console.print(
                f"  -> [bold]{'SUPER PREDICTION' if decision.passed else 'no super prediction'}"
                f"[/bold]  [dim](outcome was {realised:+.2f}%)[/dim]\n"
            )

        console.print(
            "[dim]Analytical output only. Nothing here is an instruction to trade, and "
            "no prediction is a guaranteed outcome.[/dim]"
        )

    asyncio.run(_run())


@app.command("backtest")
def backtest(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    horizon: int = typer.Option(12, "--horizon", help="Forecast horizon, in bars."),
    folds: int = typer.Option(5, "--folds"),
    scheme: str = typer.Option("expanding", "--scheme", help="expanding | rolling"),
    source: str = typer.Option("binance", "--source"),
    probe: bool = typer.Option(True, "--probe/--no-probe", help="Run the leakage probe."),
) -> None:
    """Walk history forward in folds, fitting on each and testing on the next.

    Calibration and skill weights are fitted on the training window only, then applied
    to a test window that starts after a purge of one horizon plus an embargo. Every
    model is probed for look-ahead first; anything caught reading the future is
    excluded from the results rather than annotated in them.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        chosen_scheme = (
            FoldScheme.ROLLING if scheme.startswith("roll") else FoldScheme.EXPANDING
        )

        async with IngestionService(settings) as service, service.db.session() as session:
            rows = await OHLCVRepository(session).fetch(asset, frame, source=source)
            features = await FeatureRepository(session).fetch(asset, frame, source=source)

        if len(rows) < 800:
            console.print(
                f"[yellow]not enough history[/yellow] for a fold backtest "
                f"({len(rows)} bars; need 800+)"
            )
            return

        candles = [_row_to_candle(r, asset, frame, source) for r in rows]
        context_source = ContextSource(
            asset, frame, candles, [(f.open_time, f.payload) for f in features]
        )
        harness = WalkForwardHarness(
            folds=folds, scheme=chosen_scheme, run_probe=probe, probe=LeakageProbe(max_points=12)
        )
        report = harness.run(
            [m() for m in ALL_MODELS], context_source, Horizon(bars=horizon, timeframe=frame)
        )

        console.print(
            f"\n[bold]{asset.upper()} {frame} +{horizon} bars[/bold] - "
            f"{len(report.folds)} {chosen_scheme} folds over {len(candles)} bars"
        )

        if probe:
            console.print("\n[bold]leakage probe[/bold]")
            for model_id in sorted(report.leakage):
                verdict = report.leakage[model_id].verdict
                colour = {
                    Verdict.LEAKING: "red",
                    Verdict.INCONCLUSIVE: "yellow",
                    Verdict.CLEAN: "green",
                    Verdict.SUSPICIOUS: "red",
                }[verdict]
                console.print(f"  [{colour}]{verdict.value:13}[/{colour}] {model_id}")
            if report.untestable:
                console.print(
                    f"  [dim]{len(report.untestable)} models never responded to their "
                    f"own inputs, so 'clean' would be unearned for them[/dim]"
                )

        if not report.folds:
            console.print("\n[yellow]no usable folds could be built[/yellow]")
            return

        console.print("\n[bold]folds[/bold]")
        table = Table(header_style="bold", box=None, pad_edge=False)
        for column in ("#", "train", "gap", "test", "trn pts", "tst pts", "weights", "calib", "published"):
            table.add_column(column)
        for result in report.folds:
            table.add_row(
                str(result.fold.index),
                result.train_window.label(),
                str(result.fold.gap_bars),
                result.test_window.label(),
                str(result.train_points),
                str(result.test_points),
                str(len(result.trained_weights.skilled_models())),
                f"{result.usable_calibrations}/{result.fitted_calibrations}",
                str(result.ensemble_published),
            )
        console.print(table)

        console.print("\n[bold]per-fold skill (overall slice)[/bold]")
        stability = report.stability()
        series = report.skill_by_model()
        skills = Table(header_style="bold", box=None, pad_edge=False)
        skills.add_column("model")
        for index in range(len(report.folds)):
            skills.add_column(f"f{index}")
        skills.add_column("mean")
        skills.add_column("spread")
        for model_id in sorted(series):
            average, spread = stability.get(model_id, (0.0, 0.0))
            skills.add_row(
                model_id,
                *[f"{value:+.3f}" for value in series[model_id]],
                f"{average:+.4f}",
                f"[{'red' if spread > 0.05 else 'dim'}]{spread:.4f}[/]",
            )
        console.print(skills)

        for group in report.identical_series():
            console.print(
                f"[yellow]identical in every fold[/yellow]: {', '.join(group)} "
                f"- not {len(group)} independent results"
            )

        every = sorted(report.passing_in_every_fold())
        any_fold = sorted(report.passing_any_fold())
        console.print(
            f"\npass in [bold]every[/bold] fold ({len(every)}): "
            f"[green]{', '.join(every) or 'none'}[/green]"
        )
        console.print(
            f"pass in [bold]any[/bold] fold ({len(any_fold)}): "
            f"[dim]{', '.join(any_fold) or 'none'}[/dim]"
        )
        if report.excluded:
            console.print(
                f"[red]EXCLUDED for leakage[/red]: {', '.join(sorted(report.excluded))}"
            )
        console.print(
            "\n[dim]Passing one fold out of five is what a model with no skill does "
            "roughly one time in five. The spread across folds is the number worth "
            "reading: a model that swings has found an era, not an edge.[/dim]"
        )

    asyncio.run(_run())


@app.command("predict")
def predict(
    asset: str = typer.Argument(..., help="Canonical symbol, e.g. BTC."),
    timeframe: str = typer.Option("1h", "--timeframe"),
    horizon: int = typer.Option(12, "--horizon", help="Forecast horizon, in bars."),
    source: str = typer.Option("binance", "--source"),
    points: int = typer.Option(1, "--points", help="How many recent points to record."),
) -> None:
    """Record predictions for later scoring. Written before the outcome exists.

    Storage is append-only and hash-stamped: re-running the same point collides on its
    derived id and is dropped, so a re-run can neither duplicate the sample nor revise
    what was said.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)
        contexts = await _evaluation_contexts(settings, asset, frame, horizon, source)
        if contexts is None:
            return

        models = [*[m() for m in ALL_MODELS], ClimatologyBaseline()]
        records = []
        for context, _ in contexts[-max(1, points) :]:
            returns = [abs(r) for r in context.returns(100)]
            bucket = volatility_bucket(median(returns) if returns else 0.0)
            for model in models:
                try:
                    records.append(
                        PredictionRecord.of(model.predict(context), volatility=bucket)
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("predict_failed", model=model.model_id, error=str(exc)[:200])

        async with IngestionService(settings) as service, service.db.session() as session:
            repo = PredictionRepository(session)
            offered = await repo.append(records)
            await session.commit()
            counts = await repo.counts()

        console.print(
            f"\n[bold]{asset.upper()} {frame} +{horizon}[/bold]: offered {offered} "
            f"predictions from {len(models)} models"
        )
        console.print(
            f"stored: [bold]{counts['predictions']}[/bold] predictions, "
            f"{counts['resolved']} resolved, {counts['pending']} pending"
        )
        console.print(
            "[dim]Append-only: duplicates of an existing prediction point are dropped "
            "rather than merged, so a re-run cannot inflate the sample.[/dim]"
        )

    asyncio.run(_run())


@app.command("learn")
def learn(
    timeframe: str = typer.Option("1h", "--timeframe"),
    source: str = typer.Option("binance", "--source"),
    show_slices: bool = typer.Option(False, "--slices", help="Show sliced metrics."),
) -> None:
    """Resolve due predictions, measure, reweight — and report whether anything changed.

    Storing predictions is not learning. This command reports which of three states it
    is in: nothing to learn from yet, learned nothing, or learned something — and in
    the last case, exactly which weight or calibration curve moved and on what sample.
    """

    async def _run() -> None:
        settings = _settings()
        frame = _tf(timeframe)

        async with IngestionService(settings) as service, service.db.session() as session:
            repo = PredictionRepository(session)
            # Every record, not only the unresolved ones: the resolver wants the
            # pending rows, but recalibration needs the resolved ones to pair what a
            # model said with what happened.
            rows = await repo.records()
            stored_outcomes = await repo.outcomes()
            weight_rows = await repo.weights()
            assets = {r.asset for r in rows}
            candles_by_asset = {}
            for asset in assets:
                bars = await OHLCVRepository(session).fetch(asset, frame, source=source)
                candles_by_asset[asset] = [
                    _row_to_candle(b, asset, frame, source) for b in bars
                ]

        records = [_row_to_record(r) for r in rows]
        outcomes = [_row_to_outcome(r) for r in stored_outcomes]
        previous = {
            WeightKey(w.model_id, w.asset, w.timeframe, w.horizon_bars, w.regime): w.weight
            for w in weight_rows
        }

        loop = LearningLoop()
        report = loop.run(records, candles_by_asset, existing_outcomes=outcomes,
                          previous_weights=previous)

        if report.resolved or report.weights.updates:
            async with IngestionService(settings) as service, service.db.session() as session:
                repo = PredictionRepository(session)
                newly, _ = loop.resolver.resolve(records, candles_by_asset)
                await repo.record_outcomes(newly)
                await repo.upsert_weights(report.weights.updates)
                await session.commit()

        console.print(
            f"\n[bold]learning loop[/bold]  resolved {report.resolved}, "
            f"pending {report.pending}, outcomes in evidence {report.total_outcomes}"
        )
        if report.corrupted:
            console.print(
                f"[red]{report.corrupted} records refused for hash mismatch[/red]"
            )

        changes = report.weight_changes
        if changes:
            table = Table(header_style="bold", box=None, pad_edge=False)
            for column in ("scope", "was", "now", "delta", "skill", "n", "p"):
                table.add_column(column)
            for update in sorted(changes, key=lambda u: -abs(u.delta))[:25]:
                colour = "green" if update.delta > 0 else "red"
                table.add_row(
                    update.key.label(), f"{update.previous_weight:.4f}",
                    f"{update.weight:.4f}",
                    f"[{colour}]{update.delta:+.4f}[/{colour}]",
                    f"{update.raw_skill:+.4f}", str(update.samples),
                    f"{update.p_value:.3f}",
                )
            console.print(table)

        for record in report.adopted_calibrations:
            console.print(f"  [green]calibration adopted[/green]: {record.summary()}")

        if show_slices and report.metrics.slices:
            slices = Table(header_style="bold", box=None, pad_edge=False)
            for column in ("model", "dimension", "value", "n", "brier", "accuracy"):
                slices.add_column(column)
            for entry in sorted(
                report.metrics.with_evidence(), key=lambda s: (s.model_id, s.dimension)
            ):
                slices.add_row(
                    entry.model_id, entry.dimension, entry.value, str(entry.count),
                    f"{entry.brier:.4f}", f"{entry.accuracy:.2%}",
                )
            console.print(slices)
            thin = len(report.metrics.slices) - len(report.metrics.with_evidence())
            if thin:
                console.print(f"[dim]{thin} slices had insufficient evidence[/dim]")

        colour = "green" if report.learned else "yellow"
        console.print(f"\n[bold][{colour}]{report.verdict}[/{colour}][/bold]")
        console.print(
            "[dim]Storing predictions is not learning. This line reports whether the "
            "loop changed what the system will do next.[/dim]"
        )

    asyncio.run(_run())


def _row_to_record(row) -> PredictionRecord:
    """Rebuild a stored prediction, preserving its hash for verification."""
    frame = _tf(row.timeframe)
    return PredictionRecord(
        prediction_id=row.prediction_id,
        content_hash=row.content_hash,
        model_id=row.model_id,
        model_version=row.model_version,
        asset=row.asset,
        timeframe=frame,
        horizon_bars=row.horizon_bars,
        as_of=row.as_of,
        resolves_at=row.resolves_at,
        distribution=Distribution(up=row.prob_up, flat=row.prob_flat, down=row.prob_down),
        confidence=row.confidence,
        move_threshold_pct=row.move_threshold_pct,
        reference_price=row.reference_price,
        regime=row.regime,
        volatility_bucket=row.volatility_bucket,
        data_quality=row.data_quality,
        is_actionable=row.is_actionable,
        evidence=row.evidence or {},
        resolved=row.resolved,
        created_at=row.created_at,
    )


def _row_to_outcome(row) -> ResolvedOutcome:
    frame = _tf(row.timeframe)
    return ResolvedOutcome(
        prediction_id=row.prediction_id,
        model_id=row.model_id,
        asset=row.asset,
        timeframe=frame,
        horizon_bars=row.horizon_bars,
        regime=row.regime,
        volatility_bucket=row.volatility_bucket,
        as_of=row.resolved_at - frame.delta * row.horizon_bars,
        resolved_at=row.resolved_at,
        realised_direction=Outcome(row.realised_direction),
        realised_move_pct=row.realised_move_pct,
        exit_price=row.exit_price,
        brier=row.brier,
        log_loss=row.log_loss,
        correct=row.correct,
        probability_of_truth=row.probability_of_truth,
        scored_at=row.scored_at,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
