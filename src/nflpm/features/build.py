"""features/features_{mode}.parquet: one row per (game_id, mode), `{group}_src_max_ts` per group
(`nflpm features`). Eval targets go to features/targets.parquet, never into feature tables."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from nflpm.features.context import qb_change, schedule_context
from nflpm.features.market import grid, market_features, targets
from nflpm.features.team_form import team_form
from nflpm.ingest.nflverse import read, read_schedules
from nflpm.paths import UTC, config, data_root
from nflpm.transform.games import MODES
from nflpm.validate.schemas import WEATHER

HOME_SCHEMA = {"venue": pl.String, "game_id": pl.String, "ts_kind": pl.String, "resolution_sec": pl.Int64,
               "ts_utc": pl.Datetime("us", "UTC"), "p_home_mid": pl.Float64, "p_home_bid": pl.Float64,
               "p_home_ask": pl.Float64, "quote_source": pl.String, "one_side": pl.Boolean}


def build_features(games: pl.DataFrame, sched: pl.DataFrame, tg: pl.DataFrame, injuries: pl.DataFrame,
                   home: pl.DataFrame, now: datetime) -> pl.DataFrame:
    first = config()["seasons"]["first"]  # the first season only supplies priors
    ids = set(games.filter((pl.col("season") > first) & pl.col("kickoff_utc").is_not_null())["game_id"])
    gr = grid(games, ids, now)
    mk = market_features(gr, home)
    tf = team_form(gr, games, tg)
    qb = qb_change(tf, games, injuries)
    f = (mk.join(tf.drop("predict_time", "kickoff_utc"), on=["game_id", "mode"])
         .join(schedule_context(sched, games), on="game_id", how="left")
         .join(qb, on=["game_id", "mode"], how="left"))
    return f.with_columns(
        # weather and the realized retractable-roof status are pregame-only
        **{c: pl.when(pl.col("mode") == "pregame").then(pl.col(c)) for c in WEATHER},
        context_src_max_ts=pl.max_horizontal("home_inj_ts", "away_inj_ts"),
    ).drop("home_inj_ts", "away_inj_ts", "home_prev_qb_id", "away_prev_qb_id").sort("kickoff_utc", "game_id")


def write_features(now: datetime | None = None) -> list[Path]:
    root, now = data_root(), now or datetime.now(UTC)
    games = pl.read_parquet(root / "clean/games.parquet")
    hp = root / "clean/prices_home.parquet"
    home = pl.read_parquet(hp) if hp.exists() else pl.DataFrame(schema=HOME_SCHEMA)
    f = build_features(games, read_schedules(), pl.read_parquet(root / "clean/team_games.parquet"),
                       read("injuries"), home, now)
    (root / "features").mkdir(exist_ok=True)
    out = []
    for m in MODES:
        out.append(root / f"features/features_{m}.parquet")
        f.filter(pl.col("mode") == m).write_parquet(out[-1])
    out.append(root / "features/targets.parquet")
    targets(games.filter(pl.col("game_id").is_in(home["game_id"].unique().to_list())), home).write_parquet(out[-1])
    return out
