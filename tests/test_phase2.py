"""Phase 2: team resolution, clean games. Real-data counts run only where data_root has the files."""
from datetime import datetime, timezone

import polars as pl
import pytest

from nflpm.paths import data_root
from nflpm.transform.games import build_games
from nflpm.transform.teams import FRANCHISES, franchise, home_tz

# Every code/name observed 2026-11-18 (see teams.py docstring)
NFLVERSE = ("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO "
            "NYG NYJ OAK PHI PIT SD SEA SF STL TB TEN WAS").split()
KALSHI = ("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAC KC LA LAC LAR LV MIA MIN NE NO "
          "NYG NYJ PHI PIT SEA SF TB TEN WAS").split()
POLY_SLUG = [c.lower() for c in FRANCHISES]
POLY_OUTCOME = ["49ers", "Bears ", "LAR", "Rams", "Commanders ", "Jaguars"] + [v[1] for v in FRANCHISES.values()]


@pytest.mark.parametrize("source,codes", [("nflverse", NFLVERSE), ("kalshi", KALSHI),
                                          ("poly_slug", POLY_SLUG), ("poly_outcome", POLY_OUTCOME)])
def test_every_seen_code_resolves(source, codes):
    assert {franchise(c, source) for c in codes} <= set(FRANCHISES)


def test_aliases_and_relocations():
    assert franchise("JAC", "kalshi") == franchise("jax", "poly_slug") == "JAX"
    assert franchise("LAR", "kalshi") == franchise("LAR", "poly_outcome") == franchise("STL", "nflverse") == "LA"
    assert franchise("OAK", "nflverse") == "LV" and franchise("SD", "nflverse") == "LAC"
    assert home_tz("LA", 2015) == "America/Chicago" and home_tz("LA", 2016) == "America/Los_Angeles"
    with pytest.raises(KeyError):
        franchise("XYZ", "kalshi")


def sched(**over) -> dict:
    row = dict(game_id="g", season=2026, week=1, game_type="REG", gameday="2026-11-01", gametime="13:00",
               home_team="SEA", away_team="NE", location="Home", home_score=20, away_score=17,
               home_moneyline=-150, away_moneyline=130, spread_line=3.0, total_line=44.5)
    return {**row, **over}


def test_build_games_kickoff_tie_novig_neutral_relocation():
    g = build_games(pl.DataFrame([
        sched(game_id="a"),  # Nov 1 2026: DST ended at 2am, so 13:00 is EST
        sched(game_id="b", gameday="2026-10-04", gametime="09:30", location="Neutral", home_score=10, away_score=10),
        sched(game_id="c", season=2000, gameday="2000-09-03", home_team="OAK", away_team="SD",
              home_score=None, away_score=None, home_moneyline=None),
    ])).sort("game_id")
    a, b, c = g.to_dicts()
    assert a["kickoff_utc"] == datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)  # 13:00 EST
    assert b["kickoff_utc"] == datetime(2026, 10, 4, 13, 30, tzinfo=timezone.utc)  # 09:30 EDT
    assert a["home_win"] is True and b["home_win"] is None and c["home_win"] is None
    p_h, p_a = 150 / 250, 100 / 230
    assert a["line_home_prob"] == pytest.approx(p_h / (p_h + p_a))
    assert c["line_home_prob"] is None
    assert b["neutral_site"] and not a["neutral_site"]
    assert (c["home"], c["away"]) == ("LV", "LAC")


GAMES = data_root() / "clean/games.parquet"


@pytest.mark.skipif(not GAMES.exists(), reason="needs clean/games.parquet (home PC only)")
def test_real_season_counts():
    g = pl.read_parquet(GAMES)
    reg = dict(g.filter(pl.col("game_type") == "REG").group_by("season").len().iter_rows())
    post = dict(g.filter(pl.col("game_type") != "REG").group_by("season").len().iter_rows())
    for s, n in reg.items():
        want = 248 if s <= 2001 else 256 if s <= 2020 else 272
        assert n == want or (s == 2022 and n == 271), (s, n)  # BUF-CIN 2022 W17 cancelled, absent
    for s, n in post.items():
        assert n == (11 if s <= 2019 else 13), (s, n)
    ties = g.filter(pl.col("home_score") == pl.col("away_score"))
    assert ties["home_win"].null_count() == ties.height > 0
