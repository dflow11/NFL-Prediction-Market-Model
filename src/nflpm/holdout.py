"""Holdout rows (kickoff ≥ holdout.start_utc) never enter fitting, tuning or the default eval."""
from __future__ import annotations

from datetime import datetime

import polars as pl

from nflpm.paths import config, parse_ts


def start() -> datetime:
    return parse_ts(config()["holdout"]["start_utc"])


def is_holdout(kickoff: pl.Expr | str = "kickoff_utc") -> pl.Expr:
    k = pl.col(kickoff) if isinstance(kickoff, str) else kickoff
    return k >= start()


def drop_holdout(df: pl.DataFrame, kickoff: str = "kickoff_utc") -> pl.DataFrame:
    return df.filter(~is_holdout(kickoff))
