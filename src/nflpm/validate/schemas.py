"""Validation (`nflpm validate`): schemas for clean/feature tables plus leakage and matching checks.
Every check returns a list of failure messages; `validate_all` raises if any table fails."""
from __future__ import annotations

from pathlib import Path

import pandera.polars as pa
import polars as pl

from nflpm import holdout
from nflpm.transform.prices import HOME_KEY, LONG_KEY

WEATHER = ("temp", "wind", "roof_realized")  # pregame-only features
MARKET_PREFIXES = ("kalshi_", "poly_")
FEATURE_KEY = ["game_id", "mode"]


def _prob(nullable: bool = True) -> pa.Column:
    return pa.Column(pl.Float64, pa.Check.in_range(0, 1), nullable=nullable)


def _le(lo: str, hi: str) -> pa.Check:
    return pa.Check(lambda d: d.lazyframe.select(
        (pl.col(lo) <= pl.col(hi)) | pl.col(lo).is_null() | pl.col(hi).is_null()), name=f"{lo}<={hi}")


SCHEMAS = {
    "games": pa.DataFrameSchema({"game_id": pa.Column(pl.String), "season": pa.Column(pl.Int32),
                                 "home": pa.Column(pl.String), "away": pa.Column(pl.String),
                                 "line_home_prob": _prob()}, unique=["game_id"]),
    "prices_long": pa.DataFrameSchema({"bid": _prob(), "ask": _prob(), "mid": _prob(), "last": _prob()},
                                      unique=LONG_KEY, checks=[_le("bid", "ask")]),
    "prices_home": pa.DataFrameSchema({"p_home_mid": _prob(False), "p_home_bid": _prob(), "p_home_ask": _prob()},
                                      unique=HOME_KEY, checks=[_le("p_home_bid", "p_home_ask")]),
    "snapshots": pa.DataFrameSchema({"bid": _prob(), "ask": _prob()},
                                    unique=["venue", "market_or_token", "ts_utc"], checks=[_le("bid", "ask")]),
    "features": pa.DataFrameSchema({"game_id": pa.Column(pl.String), "mode": pa.Column(pl.String),
                                    "predict_time": pa.Column(pl.Datetime("us", "UTC"))}, unique=FEATURE_KEY),
}


def schema_errors(name: str, df: pl.DataFrame) -> list[str]:
    try:
        SCHEMAS[name].validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as e:
        return [f"{name}: {f['check']} ({f.get('column')}): {f['error'][:300]}"
                for kind in e.message.values() for fails in kind.values() for f in fails]


def feature_errors(df: pl.DataFrame, mode: str) -> list[str]:
    """Point-in-time and feature-hygiene checks on one mode's feature table."""
    errs = schema_errors("features", df)
    for c in (c for c in df.columns if c.endswith("_src_max_ts")):
        n = df.filter(pl.col(c) >= pl.col("predict_time")).height
        if n:
            errs.append(f"features[{mode}]: {n} rows with {c} >= predict_time (leakage)")
    errs += [f"features[{mode}]: {c} is a line_* column" for c in df.columns if c.startswith("line_")]
    errs += [f"features[{mode}]: {c} is an eval target" for c in df.columns if c.endswith("_close")]
    if mode == "tuesday":
        errs += [f"features[tuesday]: {c} has values (weather/roof are pregame-only)"
                 for c in WEATHER if c in df.columns and df[c].null_count() < df.height]
    for p in MARKET_PREFIXES:
        feats = [c for c in df.columns if c.startswith(p) and not c.endswith(("_staleness_sec", "_src_max_ts"))]
        stale = f"{p}staleness_sec"
        if feats and stale not in df.columns:
            errs.append(f"features[{mode}]: {feats} without {stale}")
        elif feats:
            n = df.filter(pl.any_horizontal(pl.col(feats).is_not_null()) & pl.col(stale).is_null()).height
            if n:
                errs.append(f"features[{mode}]: {n} rows with {p}* values but null {stale}")
    return errs


def training_matrix_errors(X: pl.DataFrame, kickoff: pl.Series) -> list[str]:
    """Call on every matrix before fitting/tuning."""
    errs = [f"training matrix: {c} is a line_* column" for c in X.columns if c.startswith("line_")]
    errs += [f"training matrix: {c} is an outcome/label column" for c in X.columns
             if c in ("home_win", "home_score", "away_score", "result")]
    n = int((kickoff >= holdout.start()).sum())
    if n:
        errs.append(f"training matrix: {n} holdout rows")
    return errs


def match_errors(kalshi: pl.DataFrame, kalshi_markets: pl.DataFrame, poly: pl.DataFrame) -> list[str]:
    errs = []
    for name, df, venue_id in (("match_kalshi", kalshi, "event_ticker"), ("match_polymarket", poly, "market_id")):
        for col in ("game_id", venue_id):
            dup = df.group_by(col).len().filter(pl.col("len") > 1)
            if dup.height:
                errs.append(f"{name}: {dup.height} {col} values matched more than once, e.g. {dup[col][0]}")
    per_event = kalshi_markets.group_by("event_ticker").len()
    bad = kalshi.join(per_event, on="event_ticker", how="left").filter(pl.col("len") != 2)
    if bad.height:
        errs.append(f"match_kalshi: {bad.height} matched events without exactly 2 team markets")
    return errs


def validate_all(root: Path) -> list[str]:
    """Validate every table that exists under data_root. Returns failures (empty = pass)."""
    clean, errs = root / "clean", []
    for name in ("games", "prices_long", "prices_home", "snapshots"):
        if (clean / f"{name}.parquet").exists():
            errs += schema_errors(name, pl.read_parquet(clean / f"{name}.parquet"))
    if all((clean / f).exists() for f in ("match_kalshi.parquet", "kalshi_markets.parquet", "match_polymarket.parquet")):
        errs += match_errors(*(pl.read_parquet(clean / f) for f in
                               ("match_kalshi.parquet", "kalshi_markets.parquet", "match_polymarket.parquet")))
    for f in sorted((root / "features").glob("features_*.parquet")):
        errs += feature_errors(pl.read_parquet(f), f.stem.removeprefix("features_"))
    return errs
