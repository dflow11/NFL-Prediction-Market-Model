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
