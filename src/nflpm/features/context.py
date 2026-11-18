"""Context features: rest, bye, travel, neutral site, roof, weather (pregame only), QB change."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import polars as pl

from nflpm.transform.teams import aliases, home_tz

ET = ZoneInfo("America/New_York")
WEST = {"America/Los_Angeles"}
EAST = {"America/New_York", "America/Detroit", "America/Indiana/Indianapolis"}


def schedule_context(sched: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Per game, from the published schedule (known in advance): rest, short week, bye, West→East
    early kickoff, neutral site, roof class; raw weather and realized roof (gated by mode in build)."""
    g = games.filter(pl.col("kickoff_utc").is_not_null()).select("game_id", "season", "kickoff_utc", "home", "away",
                                                                 "neutral_site")
    tg = pl.concat([g.select("game_id", "season", "kickoff_utc", team="home"),
                    g.select("game_id", "season", "kickoff_utc", team="away")]).sort("team", "kickoff_utc")
    rest = tg.with_columns(rest=(pl.col("kickoff_utc") - pl.col("kickoff_utc").shift(1)).over("team", "season")
                           .dt.total_seconds() / 86400)
    out = g
    for side in ("home", "away"):
        r = rest.select("game_id", team="team", **{f"{side}_rest_days": "rest"})
        out = out.join(r, left_on=["game_id", side], right_on=["game_id", "team"], how="left")
        out = out.with_columns(**{f"{side}_short_week": pl.col(f"{side}_rest_days") < 6,
                                  f"{side}_bye": pl.col(f"{side}_rest_days") >= 13})
    tz = lambda c: pl.struct(c, "season").map_elements(lambda s: home_tz(s[c], s["season"]), return_dtype=pl.String)  # noqa: E731
    et_hour = pl.col("kickoff_utc").dt.convert_time_zone("America/New_York").dt.hour()
    roof = sched.select("game_id", "stadium_id", "roof", "temp", "wind")
    retractable = (roof.group_by("stadium_id").agg(pl.col("roof").is_in(["open", "closed"]).any().alias("retractable")))
    out = (out.join(roof, on="game_id", how="left").join(retractable, on="stadium_id", how="left")
           .with_columns(away_west_to_east_early=~pl.col("neutral_site") & tz("away").is_in(list(WEST))
                         & tz("home").is_in(list(EAST)) & (et_hour < 14))
           .with_columns(
               roof_class=pl.when(pl.col("retractable")).then(pl.lit("retractable"))
               .when(pl.col("roof").is_in(["dome", "closed"])).then(pl.lit("dome")).when(pl.col("roof").is_not_null())
               .then(pl.lit("outdoors")),
               roof_realized=pl.when(pl.col("retractable")).then("roof"),
               # temp/wind measured at outdoor/open stadiums only
               temp=pl.col("temp").cast(pl.Float64), wind=pl.col("wind").cast(pl.Float64)))
    # Travel / home-field context is 0 at neutral sites.
    return out.with_columns(
        neutral_site=pl.col("neutral_site").cast(pl.Int8),
        away_west_to_east_early=pl.col("away_west_to_east_early").cast(pl.Int8)
    ).drop("stadium_id", "roof", "retractable", "kickoff_utc", "season", "home", "away")


def qb_change(grid: pl.DataFrame, games: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    """Per side: previous game's starter listed Out/Doubtful in this game week's injury rows published
    before predict_time. Null when no qualifying rows (always before 2009). Published = date_modified;
    rows without one (2025+) count as published 16:00 ET the day before kickoff, the latest an official
    game-status report comes out: pregame sees them, tuesday never does (as with real timestamps).
    Needs grid columns kickoff_utc, home_prev_qb_id / away_prev_qb_id."""
    inj = injuries
    if "date_modified" not in inj.columns:
        inj = inj.with_columns(date_modified=pl.lit(None, pl.Datetime("us", "UTC")))
    inj = inj.select(season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32),
                     team=pl.col("team").replace_strict(aliases("nflverse"), default=None, return_dtype=pl.String),
                     gsis_id="gsis_id", out=pl.col("report_status").is_in(["Out", "Doubtful"]),
                     date_modified="date_modified")
    g = grid.join(games.select("game_id", "season", "week", "home", "away"), on="game_id")
    out = grid.select("game_id", "mode")
    for side in ("home", "away"):
        rows = (g.select("game_id", "mode", "predict_time", "season", "week", team=side, qb=f"{side}_prev_qb_id",
                         day_before=pl.col("kickoff_utc").dt.convert_time_zone("America/New_York").dt.truncate("1d")
                         .dt.offset_by("-8h").dt.convert_time_zone("UTC"))
                .join(inj, on=["season", "week", "team"])
                # ponytail: report-deadline proxy for missing date_modified; real timestamps win when present
                .with_columns(date_modified=pl.coalesce("date_modified", "day_before"))
                .filter(pl.col("date_modified") < pl.col("predict_time")))
        flag = rows.group_by("game_id", "mode").agg(
            **{f"{side}_qb_change": (pl.col("out") & (pl.col("gsis_id") == pl.col("qb"))).any().cast(pl.Int8),
               f"{side}_inj_ts": pl.col("date_modified").max()})
        out = out.join(flag, on=["game_id", "mode"], how="left")
    return out
