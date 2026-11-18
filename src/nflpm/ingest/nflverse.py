"""nflverse ingest via nflreadpy (`nflpm ingest nflverse`): schedules, pbp, injuries.

Completed seasons: raw/nflverse/{name}_{season}.parquet, written once (immutable).
Current season: raw/nflverse/{name}_{season}/{fetched_at}.parquet, a new file per run; readers use the latest.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from nflpm.paths import UTC, config, data_root, stamp, write_raw

DATASETS = {  # name -> (loader, first season)
    "schedules": (nfl.load_schedules, None),  # None = seasons.first
    "pbp": (nfl.load_pbp, None),
    "injuries": (nfl.load_injuries, 2009),
}


def raw_dir() -> Path:
    return data_root() / "raw/nflverse"


def _use_data_cache() -> None:
    update_config(cache_mode="filesystem", cache_dir=data_root() / "cache/nflreadpy", verbose=False)


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def first_season(name: str) -> int:
    return DATASETS[name][1] or config()["seasons"]["first"]


def ingest(name: str) -> list[Path]:
    """One season at a time (a pbp season is ~100 MB in memory)."""
    _use_data_cache()
    load, current = DATASETS[name][0], config()["seasons"]["current"]
    written = []
    for s in range(first_season(name), current + 1):
        path = (raw_dir() / f"{name}_{s}.parquet" if s < current
                else raw_dir() / f"{name}_{s}/{stamp(datetime.now(UTC))}.parquet")
        if s < current and path.exists():
            continue
        write_raw(path, _parquet(load(s)))
        written.append(path)
    return written


def ingest_nflverse() -> list[Path]:
    return [p for name in DATASETS for p in ingest(name)]


def season_file(name: str, season: int) -> Path | None:
    if season < config()["seasons"]["current"]:
        p = raw_dir() / f"{name}_{season}.parquet"
        return p if p.exists() else None
    latest = sorted((raw_dir() / f"{name}_{season}").glob("*.parquet"))
    return latest[-1] if latest else None


def read(name: str, seasons: range | None = None) -> pl.DataFrame:
    """Seasons of one dataset (default: all on disk); the current season from its latest fetch."""
    seasons = seasons or range(first_season(name), config()["seasons"]["current"] + 1)
    files = [f for s in seasons if (f := season_file(name, s))]
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def read_schedules() -> pl.DataFrame:
    return read("schedules")
