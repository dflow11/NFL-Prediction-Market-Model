"""nflverse ingest via nflreadpy (`nflpm ingest`)."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from nflpm.paths import UTC, config, data_root, stamp, write_raw


def raw_dir() -> Path:
    return data_root() / "raw/nflverse"


def _use_data_cache() -> None:
    update_config(cache_mode="filesystem", cache_dir=data_root() / "cache/nflreadpy", verbose=False)


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def ingest_schedules() -> list[Path]:
    """Completed seasons: one immutable file each, written once. Current season: a new file per run."""
    _use_data_cache()
    first, current = config()["seasons"]["first"], config()["seasons"]["current"]
    todo = [s for s in range(first, current) if not (raw_dir() / f"schedules_{s}.parquet").exists()]
    df = nfl.load_schedules(todo + [current])  # one nflverse file holds every season
    written = []
    for s in todo:
        written.append(raw_dir() / f"schedules_{s}.parquet")
        write_raw(written[-1], _parquet(df.filter(pl.col("season") == s)))
    written.append(raw_dir() / f"schedules_{current}/{stamp(datetime.now(UTC))}.parquet")
    write_raw(written[-1], _parquet(df.filter(pl.col("season") == current)))
    return written


def read_schedules() -> pl.DataFrame:
    """All seasons; the current season from its latest fetch."""
    current = config()["seasons"]["current"]
    files = sorted(raw_dir().glob("schedules_*.parquet"))
    latest = sorted((raw_dir() / f"schedules_{current}").glob("*.parquet"))[-1:]
    return pl.concat([pl.read_parquet(f) for f in files + latest], how="diagonal_relaxed")
