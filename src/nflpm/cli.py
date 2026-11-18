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
        from nflpm.ingest.nflverse import ingest_nflverse
        for p in ingest_nflverse():
            print(p)
    if source in ("kalshi", "all"):
        from nflpm.ingest.kalshi import ingest_kalshi
        raise typer.Exit(ingest_kalshi())


@app.command()
def transform() -> None:
    """Build clean tables from raw: games, team_games, snapshots (+ reports/coverage.csv), prices, settlements."""
    import polars as pl
    from nflpm.paths import data_root
    from nflpm.transform.games import write_games, write_team_games
    from nflpm.transform.prices import write_prices
    from nflpm.transform.snapshots import write_snapshots
    games = write_games()
    print(games)
    print(write_team_games())
    print(write_snapshots(pl.read_parquet(games)))
    for p in write_prices(data_root()):
        print(p)


@app.command()
def features() -> None:
    """Build features_{mode} (market, team form, context) + eval targets; print lead-lag slopes."""
    import polars as pl
    from nflpm.features.build import write_features
    from nflpm.models.evaluate import lead_lag
    from nflpm.paths import data_root
    root = data_root()
    paths = write_features()
    for p in paths:
        print(p)
    t = pl.read_parquet(root / "features/targets.parquet")
    with pl.Config(tbl_rows=20, tbl_cols=10):
        for p in paths[:-1]:
            print(lead_lag(pl.read_parquet(p), t))


@app.command()
def train() -> None:
    """Fit F (frozen: 2000-2023, calibrated on 2024) per mode; walk-forward B; write predictions."""
    from nflpm.models.train import train_all
    for p in train_all():
        print(p)


@app.command()
def validate() -> None:
    """Validate clean and feature tables; exit 1 on any failure."""
    from nflpm.paths import data_root
    from nflpm.validate.schemas import validate_all
    errs = validate_all(data_root())
    for e in errs:
        print("FAIL", e)
    print(f"{len(errs)} validation failures")
    raise typer.Exit(1 if errs else 0)


@app.command()
def collect() -> None:
    """One Kalshi + Polymarket snapshot pass (run every 5 min by nflpm-collect.timer)."""
    from nflpm.collect.snapshots import collect as run
    raise typer.Exit(run())
