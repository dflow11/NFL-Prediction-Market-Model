"""Team-form features. Point-in-time: only team-games with game_end_ts < predict_time, league-wide
stats included. Each team's running state after every finished game is joined with a strict ASOF on
game_end_ts < predict_time, so a game still in progress (or ending exactly at predict time) never counts."""
from __future__ import annotations

import polars as pl

STATS = ["off_epa", "def_epa", "off_success", "def_success"]
SHRINK_GAMES = 4  # prior-season blend counts as this many games
PRIOR_TO_MEAN = 0.5  # prior season regressed halfway to the league mean


def team_states(tg: pl.DataFrame) -> pl.DataFrame:
    """State of each team after each of its games: rolling last 4/8, season-to-date and luck stats."""
    t = tg.filter(pl.col("game_end_ts").is_not_null()).sort("team", "game_end_ts").with_columns(
        margin=pl.col("points_for") - pl.col("points_against"))
    t = t.with_columns(
        win=pl.when(pl.col("margin") > 0).then(1.0).when(pl.col("margin") < 0).then(0.0).otherwise(0.5),
        one_score=pl.col("margin").abs() <= 8)
    by_team, by_season = ["team"], ["team", "season"]
    cum = lambda e: e.cum_sum().over(by_season, order_by="game_end_ts")  # noqa: E731
    exprs = []
    for s in STATS:
        exprs += [pl.col(s).rolling_mean(4, min_samples=1).over(by_team, order_by="game_end_ts").alias(f"l4_{s}"),
                  pl.col(s).rolling_mean(8, min_samples=1).over(by_team, order_by="game_end_ts").alias(f"l8_{s}"),
                  (cum(pl.col(s).fill_null(0)) / cum(pl.col(s).is_not_null().cast(pl.Int64))).alias(f"std_{s}")]
    return t.with_columns(
        *exprs,
        std_games=cum(pl.lit(1)),
        std_point_diff=cum(pl.col("margin")) / cum(pl.lit(1)),
        std_fumble_rec_rate=cum(pl.col("fumbles_recovered")) / cum(pl.col("fumbles_game")),
        std_one_score_win=cum(pl.col("win") * pl.col("one_score")) / cum(pl.col("one_score").cast(pl.Int64)),
        std_turnover_margin=cum(pl.col("takeaways") - pl.col("giveaways")) / cum(pl.lit(1)),
    ).with_columns(std_epa_diff=pl.col("std_off_epa") - pl.col("std_def_epa"))


SHRUNK = [f"std_{s}" for s in STATS] + ["std_point_diff", "std_epa_diff"]
FORM_COLS = ([f"l4_{s}" for s in STATS] + [f"l8_{s}" for s in STATS] + SHRUNK
             + ["std_fumble_rec_rate", "std_one_score_win", "std_turnover_margin", "std_games"])


def season_priors(states: pl.DataFrame) -> pl.DataFrame:
    """(team, season) -> prior for each shrunk stat: last season's final value regressed toward that
    season's league mean. Season s's prior comes from season s−1 (1999 only supplies priors for 2000)."""
    final = states.sort("game_end_ts").group_by("team", "season").last()
    league = final.group_by("season").agg(pl.col(SHRUNK).mean().name.suffix("_lg"))
    return final.join(league, on="season").select(
        "team", season=pl.col("season") + 1,
        **{f"{c}_prior": PRIOR_TO_MEAN * pl.col(c) + (1 - PRIOR_TO_MEAN) * pl.col(f"{c}_lg") for c in SHRUNK})


def league_to_date(tg: pl.DataFrame) -> pl.DataFrame:
    """League-wide season-to-date off/def EPA after each finished team-game (point-in-time too)."""
    t = tg.filter(pl.col("game_end_ts").is_not_null() & pl.col("off_epa").is_not_null()).sort("game_end_ts")
    cum = lambda c: pl.col(c).cum_sum().over("season", order_by="game_end_ts")  # noqa: E731
    return t.select("season", lg_ts="game_end_ts",
                    lg_off_epa=cum("off_epa") / pl.int_range(1, pl.len() + 1).over("season", order_by="game_end_ts"),
                    lg_def_epa=cum("def_epa") / pl.int_range(1, pl.len() + 1).over("season", order_by="game_end_ts"))


def team_form(grid: pl.DataFrame, games: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """grid: game_id, mode, predict_time (+ kickoff_utc). Adds home_*/away_* form, rel-to-league EPA,
    prev-game QB ids (for context) and team_form_src_max_ts."""
    states = team_states(tg)
    priors = season_priors(states)
    lg = league_to_date(tg).sort("lg_ts")
    g = grid.join(games.select("game_id", "season", "home", "away"), on="game_id")
    out = g
    for side in ("home", "away"):
        st = states.select("team", ts=pl.col("game_end_ts"), st_season="season", qb_id="qb_id", *FORM_COLS).sort("ts")
        x = (g.select("game_id", "mode", "predict_time", "season", team=side).sort("predict_time")
             .join_asof(st, left_on="predict_time", right_on="ts", by="team", strategy="backward",
                        allow_exact_matches=False, check_sortedness=False)
             .join(priors, on=["team", "season"], how="left"))
        same = pl.col("st_season") == pl.col("season")
        n = pl.when(same).then(pl.col("std_games")).otherwise(0)
        shrunk = {c: pl.when(pl.col(f"{c}_prior").is_null()).then(pl.when(same).then(pl.col(c)))
                  .otherwise((n * pl.col(c).fill_null(0) + SHRINK_GAMES * pl.col(f"{c}_prior")) / (n + SHRINK_GAMES))
                  for c in SHRUNK}
        luck = {c: pl.when(same).then(pl.col(c)) for c in ("std_fumble_rec_rate", "std_one_score_win", "std_turnover_margin")}
        x = x.select("game_id", "mode", *[f"l{w}_{s}" for w in (4, 8) for s in STATS],
                     **shrunk, **luck, std_games=n, prev_qb_id="qb_id", ts="ts")
        out = out.join(x.rename(lambda c: c if c in ("game_id", "mode") else f"{side}_{c}"), on=["game_id", "mode"])
    out = (out.sort("predict_time").join_asof(lg, left_on="predict_time", right_on="lg_ts", by="season",
                                               strategy="backward", allow_exact_matches=False, check_sortedness=False))
    rel = {f"{side}_rel_{s}": pl.col(f"{side}_std_{s}") - pl.col(f"lg_{s}") for side in ("home", "away")
           for s in ("off_epa", "def_epa")}
    return out.with_columns(**rel, team_form_src_max_ts=pl.max_horizontal("home_ts", "away_ts", "lg_ts")).drop(
        "home_ts", "away_ts", "lg_ts", "season", "home", "away")
