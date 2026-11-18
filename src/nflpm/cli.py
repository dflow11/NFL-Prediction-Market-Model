"""nflpm command line. Later phases add ingest, transform, features, validate, train, ..."""
from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def archive(slugs: Optional[list[str]] = typer.Argument(None, help="Only these game slugs.")) -> None:
    """Archive Polymarket price history for every started 2026 game window."""
    from nflpm.ingest.polymarket import archive as run
    raise typer.Exit(run(slugs or []))


@app.command()
def ingest(source: str = typer.Argument("all", help="nflverse | kalshi | all")) -> None:
    """Ingest raw data: nflverse schedules; Kalshi markets, match, candles."""
    if source in ("nflverse", "all"):
        from nflpm.ingest.nflverse import ingest_schedules
        for p in ingest_schedules():
            print(p)
    if source in ("kalshi", "all"):
        from nflpm.ingest.kalshi import ingest_kalshi
        raise typer.Exit(ingest_kalshi())


@app.command()
def transform() -> None:
    """Build clean tables from raw (games.parquet)."""
    from nflpm.transform.games import write_games
    print(write_games())


@app.command()
def collect() -> None:
    """One Kalshi + Polymarket snapshot pass (run every 5 min by nflpm-collect.timer)."""
    from nflpm.collect.snapshots import collect as run
    raise typer.Exit(run())
