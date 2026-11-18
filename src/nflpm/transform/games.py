"""clean/games.parquet from nflverse schedules (`nflpm transform`)."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from nflpm.ingest.nflverse import read_schedules
from nflpm.paths import UTC, config, data_root
from nflpm.transform.teams import franchise

ET = ZoneInfo("America/New_York")
WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MODES = ("pregame", "tuesday")


def predict_time(kickoff: datetime, mode: str) -> datetime:
    """pregame = kickoff − offset; tuesday = latest Tuesday 12:00 ET strictly before kickoff."""
    m = config()["modes"]
    if mode == "pregame":
        return kickoff - timedelta(minutes=m["pregame"]["offset_minutes"])
    k = kickoff.astimezone(ET)
    hh, mm = map(int, m["tuesday"]["time_et"].split(":"))
    d = k.date() - timedelta(days=(k.weekday() - WEEKDAYS.index(m["tuesday"]["weekday"])) % 7)
    t = datetime.combine(d, time(hh, mm), ET)
    return (t if t < k else t - timedelta(days=7)).astimezone(UTC)


def _implied(ml: pl.Expr) -> pl.Expr:
    """American moneyline -> implied probability (with vig)."""
    return pl.when(ml > 0).then(100 / (ml + 100)).otherwise(-ml / (-ml + 100))


def build_games(sched: pl.DataFrame) -> pl.DataFrame:
    home_p, away_p = _implied(pl.col("home_moneyline")), _implied(pl.col("away_moneyline"))
    to_franchise = lambda c: pl.col(c).map_elements(lambda x: franchise(x, "nflverse"),  # noqa: E731
                                                    return_dtype=pl.String)
    return sched.select(
        "game_id", "season", "week", "game_type",
        # gameday + gametime are America/New_York; gametime is null for all of 1999 -> null kickoff
        kickoff_utc=pl.concat_str("gameday", "gametime", separator=" ")
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M")
        .dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC"),
        home=to_franchise("home_team"),
        away=to_franchise("away_team"),
        neutral_site=pl.col("location") == "Neutral",
        home_score="home_score",
        away_score="away_score",
        home_win=pl.when(pl.col("home_score") > pl.col("away_score")).then(True)
        .when(pl.col("home_score") < pl.col("away_score")).then(False),  # tie / unplayed -> null
        # close_* renamed line_*: nflverse names no book and no capture time
        line_home_prob=home_p / (home_p + away_p),  # multiplicative no-vig
        line_spread="spread_line",
        line_total="total_line",
    ).sort("season", "kickoff_utc", "game_id")


def write_games() -> Path:
    out = data_root() / "clean/games.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_games(read_schedules()).write_parquet(out)
    return out


def _team(c: str) -> pl.Expr:
    from nflpm.transform.teams import aliases
    return pl.col(c).replace_strict(aliases("nflverse"), default=None, return_dtype=pl.String)


def team_games_season(pbp: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game for one season's pbp. game_end_ts = wall-clock UTC time of the last play
    (pbp `time_of_day`), else kickoff + game_end_fallback_minutes."""
    plays = pbp.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null())
    off = plays.group_by("game_id", team=_team("posteam")).agg(
        off_epa=pl.col("epa").mean(), off_success=pl.col("success").mean(), off_plays=pl.len())
    dfn = plays.group_by("game_id", team=_team("defteam")).agg(
        def_epa=pl.col("epa").mean(), def_success=pl.col("success").mean(), def_plays=pl.len())
    ints = pbp.filter(pl.col("interception") == 1).group_by("game_id", team=_team("posteam")).agg(ints_thrown=pl.len())
    fum_lost = pbp.filter(pl.col("fumble_lost") == 1).group_by("game_id", team=_team("fumbled_1_team")).agg(fumbles_lost=pl.len())
    fum = pbp.filter(pl.col("fumble") == 1)
    fum_total = fum.group_by("game_id").agg(fumbles_game=pl.len())
    fum_rec = fum.group_by("game_id", team=_team("fumble_recovery_1_team")).agg(fumbles_recovered=pl.len())
    qb = (pbp.filter(pl.col("qb_dropback") == 1)
          .group_by("game_id", team=_team("posteam"), qb_id=pl.coalesce("passer_player_id", "passer_id"))
          .agg(qb_dropbacks=pl.len()).drop_nulls("qb_id")
          .sort("qb_dropbacks", "qb_id", descending=[True, False]).unique(["game_id", "team"], keep="first"))
    tod = (pl.col("time_of_day").str.replace(r"\.\d+Z$", "Z")  # some seasons carry milliseconds
           .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False))
    end = pbp.group_by("game_id").agg(last_play=tod.max())
    k = pl.col("kickoff_utc")
    # Implausible last-play times (2003-04 and some 2016 rows are off by hours) fall back. Real games
    # take ≥2h40; the 2h30 floor keeps a 1-hour time-zone error from ending a game too early (leakage).
    plausible = pl.col("last_play").is_between(k + pl.duration(minutes=150), k + pl.duration(hours=8))
    g = games.filter(pl.col("home_score").is_not_null()).join(end, on="game_id", how="left").with_columns(
        game_end_ts=pl.when(plausible).then("last_play")
        .otherwise(k + pl.duration(minutes=config()["game_end_fallback_minutes"])),
        game_end_source=pl.when(plausible).then(pl.lit("last_play")).otherwise(pl.lit("fallback")))
    base = ["game_id", "season", "week", "game_type", "kickoff_utc", "game_end_ts", "game_end_source", "neutral_site"]
    sides = pl.concat([
        g.select(*base, team="home", opp="away", is_home=pl.lit(True), points_for="home_score", points_against="away_score"),
        g.select(*base, team="away", opp="home", is_home=pl.lit(False), points_for="away_score", points_against="home_score")])
    tg = sides
    for t in (off, dfn, ints, fum_lost, fum_rec, qb):
        tg = tg.join(t, on=["game_id", "team"], how="left")
    tg = tg.join(fum_total, on="game_id", how="left").with_columns(
        pl.col("ints_thrown", "fumbles_lost", "fumbles_recovered", "fumbles_game").fill_null(0))
    give = tg.select("game_id", opp="team", takeaways=pl.col("ints_thrown") + pl.col("fumbles_lost"))
    return tg.join(give, on=["game_id", "opp"], how="left").with_columns(
        giveaways=pl.col("ints_thrown") + pl.col("fumbles_lost"))


def write_team_games() -> Path:
    """clean/team_games.parquet, aggregated once from pbp (nothing downstream reloads full pbp)."""
    from nflpm.ingest.nflverse import first_season, season_file
    games = build_games(read_schedules())
    seasons = range(first_season("pbp"), config()["seasons"]["current"] + 1)
    parts = [team_games_season(pl.read_parquet(f), games.filter(pl.col("season") == s))
             for s in seasons if (f := season_file("pbp", s))]
    out = data_root() / "clean/team_games.parquet"
    pl.concat(parts, how="diagonal_relaxed").sort("kickoff_utc", "game_id", "team").write_parquet(out)
    return out


def game_ends(games: pl.DataFrame) -> pl.DataFrame:
    """game_id -> game_end_ts: pbp-derived (clean/team_games) where available, else kickoff + fallback."""
    out = games.select("game_id", fb=pl.col("kickoff_utc") + pl.duration(minutes=config()["game_end_fallback_minutes"]))
    p = data_root() / "clean/team_games.parquet"
    if p.exists():
        out = out.join(pl.read_parquet(p, columns=["game_id", "game_end_ts"]).unique("game_id"), on="game_id", how="left")
    else:
        out = out.with_columns(game_end_ts=pl.lit(None, pl.Datetime("us", "UTC")))
    return out.select("game_id", game_end_ts=pl.coalesce("game_end_ts", "fb"))
