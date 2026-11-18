"""Phase 9: point-in-time football features (incl. league-wide stats), tuesday weather gating."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from nflpm.features.build import HOME_SCHEMA, build_features
from nflpm.features.team_form import team_form
from nflpm.transform.games import predict_time

UTC, ET = timezone.utc, ZoneInfo("America/New_York")


def et(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(UTC)


# Week 1 (Sun Sep 13) everyone plays; Week 2: A–B Sun 1:00, C–D Sun 4:25, E–F Mon 8:15.
GAMES = [  # game_id, week, home, away, kickoff, minutes to game end, home pts, away pts
    ("w1_ab", 1, "ATL", "BUF", et(2026, 9, 13, 13), 190, 20, 10),
    ("w1_cd", 1, "CHI", "DAL", et(2026, 9, 13, 13), 190, 17, 14),
    ("w1_ef", 1, "DEN", "KC", et(2026, 9, 13, 16, 25), 190, 21, 24),
    ("w2_ab", 2, "ATL", "BUF", et(2026, 9, 20, 13), 190, 30, 3),
    ("w2_cd", 2, "CHI", "DAL", et(2026, 9, 20, 16, 25), 190, 10, 13),
    ("w2_ef", 2, "DEN", "KC", et(2026, 9, 21, 20, 15), 190, 7, 6),
]


def tables():
    games = pl.DataFrame([dict(game_id=g, season=2026, week=w, home=h, away=a, kickoff_utc=k, neutral_site=False,
                               home_score=hs, away_score=as_) for g, w, h, a, k, _, hs, as_ in GAMES])
    rows = []
    for g, w, h, a, k, mins, hs, as_ in GAMES:
        end = k + timedelta(minutes=mins)
        big = 1.0 if g == "w2_ab" else 0.0  # the 1:00 game of week 2 has an extreme EPA
        for team, pf, pa in ((h, hs, as_), (a, as_, hs)):
            rows.append(dict(game_id=g, season=2026, team=team, game_end_ts=end, points_for=pf, points_against=pa,
                             off_epa=0.1 + big, def_epa=0.0 + big, off_success=0.45, def_success=0.45,
                             fumbles_recovered=1, fumbles_game=2, takeaways=1, giveaways=1, qb_id=f"qb_{team}"))
    return games, pl.DataFrame(rows)


def grid_for(game_id, mode, games):
    k = games.filter(pl.col("game_id") == game_id)["kickoff_utc"][0]
    return pl.DataFrame([{"game_id": game_id, "mode": mode, "kickoff_utc": k, "predict_time": predict_time(k, mode)}])


def test_pregame_425_game_excludes_same_day_1pm_games_including_league_stats():
    games, tg = tables()
    f = team_form(grid_for("w2_cd", "pregame", games), games, tg).to_dicts()[0]
    assert f["team_form_src_max_ts"] < f["predict_time"]
    assert f["home_std_games"] == 1  # only week 1 for CHI
    # League-to-date = week-1 games only (all off_epa 0.1); the 1:00 game (off_epa 1.1) is not over yet.
    assert abs(f["home_rel_off_epa"] - (f["home_std_off_epa"] - 0.1)) < 1e-9


def test_tuesday_mnf_excludes_same_week_sunday_games():
    games, tg = tables()
    f = team_form(grid_for("w2_ef", "tuesday", games), games, tg).to_dicts()[0]
    assert f["predict_time"] == et(2026, 9, 15, 12)
    assert f["team_form_src_max_ts"] < et(2026, 9, 20, 13)  # nothing from Sunday Sep 20
    assert abs(f["home_rel_off_epa"] - (f["home_std_off_epa"] - 0.1)) < 1e-9


def test_tuesday_has_no_weather_pregame_does():
    games, tg = tables()
    sched = games.select("game_id", stadium_id=pl.lit("S1"), roof=pl.lit("open"), temp=pl.lit(61), wind=pl.lit(8))
    injuries = pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String, "gsis_id": pl.String,
                                    "report_status": pl.String, "date_modified": pl.Datetime("us", "UTC")})
    f = build_features(games, sched, tg, injuries, pl.DataFrame(schema=HOME_SCHEMA), now=et(2026, 9, 30, 0))
    tue = f.filter((pl.col("mode") == "tuesday") & (pl.col("game_id") == "w2_ef")).to_dicts()[0]
    pre = f.filter((pl.col("mode") == "pregame") & (pl.col("game_id") == "w2_ef")).to_dicts()[0]
    assert tue["temp"] is None and tue["wind"] is None and tue["roof_realized"] is None
    assert (pre["temp"], pre["wind"], pre["roof_realized"], pre["roof_class"]) == (61.0, 8.0, "open", "retractable")
    assert tue["home_qb_change"] is None  # no qualifying injury rows -> null
    assert f.select("game_id", "mode").is_unique().all()
