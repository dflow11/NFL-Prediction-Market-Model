"""Market features per (game_id, mode) and eval targets (`nflpm features`).

ASOF joins take the latest price with ts_utc strictly before predict_time (Kalshi ts = candle end),
so every source timestamp is < predict_time as the point-in-time rule requires.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from nflpm.paths import append_jsonl, config, data_root
from nflpm.transform.games import MODES, game_ends, predict_time

CLIP = (0.01, 0.99)
# At equal timestamps prefer the finest/most direct quote.
PRIORITY = {("candle_end", 60): 0, ("snapshot", 0): 1, ("candle_end", 3600): 2}


def logit(p: pl.Expr) -> pl.Expr:
    p = p.clip(*CLIP)
    return (p / (1 - p)).log()


def inv_logit(x: pl.Expr) -> pl.Expr:
    return 1 / (1 + (-x).exp())


def _one_per_ts(df: pl.DataFrame) -> pl.DataFrame:
    """One quote per (game_id, ts_utc): snapshots/1-min first, then finer history grains."""
    rank = pl.struct("ts_kind", "resolution_sec").map_elements(
        lambda s: PRIORITY.get((s["ts_kind"], s["resolution_sec"]), 10 + s["resolution_sec"]), return_dtype=pl.Int64)
    return (df.with_columns(_rank=rank).sort("game_id", "ts_utc", "_rank")
            .unique(["game_id", "ts_utc"], keep="first", maintain_order=True).drop("_rank").sort("ts_utc"))


def _asof(grid: pl.DataFrame, prices: pl.DataFrame, at: str, strict: bool) -> pl.DataFrame:
    return grid.sort(at).join_asof(prices, left_on=at, right_on="ts_utc", by="game_id", strategy="backward",
                                   allow_exact_matches=not strict, check_sortedness=False)


def prev_game_end(games: pl.DataFrame) -> pl.DataFrame:
    """game_id -> latest game_end_ts among both teams' previous games."""
    g = games.filter(pl.col("kickoff_utc").is_not_null())
    g = g.join(game_ends(g), on="game_id")
    team_games = pl.concat([g.select("game_id", "kickoff_utc", "game_end_ts", team="home"),
                            g.select("game_id", "kickoff_utc", "game_end_ts", team="away")]).sort("team", "kickoff_utc")
    prev = team_games.with_columns(prev_end=pl.col("game_end_ts").shift(1).over("team"))
    return prev.group_by("game_id").agg(pl.col("prev_end").max())


def grid(games: pl.DataFrame, game_ids: set[str], now: datetime) -> pl.DataFrame:
    """One row per (game_id, mode) whose predict_time has passed. Tuesday rows whose predict_time is
    not after both teams' previous game end are dropped and logged."""
    g = games.filter(pl.col("game_id").is_in(list(game_ids)) & pl.col("kickoff_utc").is_not_null())
    rows = [{"game_id": r["game_id"], "mode": m, "kickoff_utc": r["kickoff_utc"],
             "predict_time": predict_time(r["kickoff_utc"], m)} for r in g.iter_rows(named=True) for m in MODES]
    out = pl.DataFrame(rows).join(prev_game_end(games), on="game_id", how="left")
    bad = out.filter((pl.col("mode") == "tuesday") & (pl.col("predict_time") <= pl.col("prev_end")))
    for r in bad.iter_rows(named=True):
        append_jsonl(data_root() / "logs/features/dropped.jsonl",
                     {"ts": now.isoformat(), "game_id": r["game_id"], "mode": "tuesday",
                      "reason": "predict_time not after a team's previous game end",
                      "predict_time": r["predict_time"].isoformat(), "prev_end": r["prev_end"].isoformat()})
    return (out.join(bad.select("game_id", "mode"), on=["game_id", "mode"], how="anti")
            .filter(pl.col("predict_time") < now).drop("prev_end"))


def market_features(grid: pl.DataFrame, home: pl.DataFrame) -> pl.DataFrame:
    k = _one_per_ts(home.filter(pl.col("venue") == "kalshi"))
    snaps = _one_per_ts(home.filter((pl.col("venue") == "polymarket") & (pl.col("ts_kind") == "snapshot")))
    hist = _one_per_ts(home.filter((pl.col("venue") == "polymarket") & (pl.col("ts_kind") == "history_point")))
    max_age = config()["collector"]["max_snapshot_age_sec"]
    stale = lambda c: (pl.col("predict_time") - pl.col(c)).dt.total_seconds()  # noqa: E731

    out = _asof(grid, k.select("game_id", "ts_utc", kalshi_mid="p_home_mid", kalshi_bid="p_home_bid",
                               kalshi_ask="p_home_ask"), "predict_time", True).rename({"ts_utc": "k_ts"})
    out = _asof(out, snaps.select("game_id", "ts_utc", s_mid="p_home_mid"), "predict_time", True).rename({"ts_utc": "s_ts"})
    out = _asof(out, hist.select("game_id", "ts_utc", h_mid="p_home_mid", h_res="resolution_sec"),
                "predict_time", True).rename({"ts_utc": "h_ts"})
    use_snap = pl.col("s_ts").is_not_null() & (stale("s_ts") <= max_age)
    return out.with_columns(
        kalshi_staleness_sec=stale("k_ts"),
        poly_mid=pl.when(use_snap).then("s_mid").otherwise("h_mid"),
        poly_ts=pl.when(use_snap).then("s_ts").otherwise("h_ts"),
        poly_resolution_sec=pl.when(use_snap).then(0).otherwise("h_res"),
        poly_source=pl.when(use_snap).then(pl.lit("snapshot")).when(pl.col("h_ts").is_not_null()).then(pl.lit("history")),
    ).with_columns(
        poly_staleness_sec=stale("poly_ts"),
        gap_logit=logit(pl.col("kalshi_mid")) - logit(pl.col("poly_mid")),
        market_src_max_ts=pl.max_horizontal("k_ts", "poly_ts"),
    ).select("game_id", "mode", "kickoff_utc", "predict_time", "kalshi_mid", "kalshi_bid", "kalshi_ask",
             "kalshi_staleness_sec", "poly_mid", "poly_staleness_sec", "poly_resolution_sec", "poly_source",
             "gap_logit", "market_src_max_ts").sort("game_id", "mode")


def targets(games: pl.DataFrame, home: pl.DataFrame) -> pl.DataFrame:
    """Eval targets only, never features: last home probability per venue. Kalshi: ts ≤ kickoff (a candle
    ending at kickoff is the last pre-game minute). Polymarket: ts < kickoff — its point stamped at kickoff
    spikes as the book clears at game start (deviation from the spec's ≤, approved 2026-11-18)."""
    g = games.select("game_id", "kickoff_utc").filter(pl.col("kickoff_utc").is_not_null())
    out = g
    for venue, col, strict in (("kalshi", "kalshi_close", False), ("polymarket", "poly_close", True)):
        p = _one_per_ts(home.filter(pl.col("venue") == venue)).select("game_id", "ts_utc", **{col: "p_home_mid"})
        out = _asof(out, p, "kickoff_utc", strict=strict).rename({"ts_utc": f"{col}_ts"})
    return out.sort("game_id")
