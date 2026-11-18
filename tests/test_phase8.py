"""Phase 8: ASOF point-in-time, one row per (game, mode), A2 fallback, tuesday drop rule."""
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from nflpm.features.market import grid, market_features, targets
from nflpm.models.train import market_models
from nflpm.transform.games import predict_time

UTC = timezone.utc
KICK = datetime(2026, 11, 22, 18, tzinfo=UTC)  # Sun 13:00 EST
PT = predict_time(KICK, "pregame")


def home_row(venue, ts, mid, kind="candle_end", res=60, game="g"):
    return {"venue": venue, "game_id": game, "ts_kind": kind, "resolution_sec": res, "ts_utc": ts,
            "p_home_mid": mid, "p_home_bid": mid - 0.01, "p_home_ask": mid + 0.01, "quote_source": "bbo",
            "one_side": False}


def run(rows, modes=("pregame",)):
    g = pl.DataFrame([{"game_id": "g", "mode": m, "kickoff_utc": KICK, "predict_time": predict_time(KICK, m)}
                      for m in modes])
    return market_features(g, pl.DataFrame(rows))


def test_asof_never_uses_a_candle_ending_at_or_after_predict_time():
    f = run([home_row("kalshi", PT - timedelta(minutes=1), 0.40),
             home_row("kalshi", PT, 0.50),  # ends exactly at predict_time: not yet usable
             home_row("kalshi", PT + timedelta(seconds=1), 0.60)]).to_dicts()[0]
    assert f["kalshi_mid"] == 0.40 and f["kalshi_staleness_sec"] == 60
    assert f["market_src_max_ts"] < f["predict_time"]


def test_poly_prefers_fresh_snapshot_else_history_and_a2_fallback():
    fresh = run([home_row("kalshi", PT - timedelta(minutes=1), 0.40),
                 home_row("polymarket", PT - timedelta(minutes=30), 0.45, "history_point", 1800),
                 home_row("polymarket", PT - timedelta(minutes=4), 0.47, "snapshot", 0)]).to_dicts()[0]
    assert (fresh["poly_mid"], fresh["poly_source"], fresh["poly_resolution_sec"]) == (0.47, "snapshot", 0)
    old = run([home_row("kalshi", PT - timedelta(minutes=1), 0.40),
               home_row("polymarket", PT - timedelta(minutes=30), 0.45, "history_point", 1800),
               home_row("polymarket", PT - timedelta(minutes=20), 0.47, "snapshot", 0)]).to_dicts()[0]
    assert (old["poly_mid"], old["poly_source"]) == (0.45, "history")  # snapshot older than 600s
    assert old["gap_logit"] < 0

    no_poly = run([home_row("kalshi", PT - timedelta(minutes=1), 0.40)])
    games = pl.DataFrame({"game_id": ["g"], "line_home_prob": [0.42]})
    m = market_models(no_poly, games).to_dicts()[0]
    assert m["a2_fallback"] and m["p_A2"] == m["p_A"] == 0.40 and m["p_0"] == 0.42
    both = market_models(pl.DataFrame([fresh]), games).to_dicts()[0]
    assert not both["a2_fallback"] and 0.40 < both["p_A2"] < 0.47


def test_one_row_per_game_and_mode():
    rows = [home_row("kalshi", KICK - timedelta(days=d, minutes=m), 0.4) for d in range(6) for m in (0, 1)]
    f = run(rows, modes=("pregame", "tuesday"))
    assert f.height == 2 and f.select("game_id", "mode").is_unique().all()


def test_tuesday_row_dropped_when_previous_game_not_over(tmp_path, monkeypatch):
    monkeypatch.setenv("NFLPM_DATA_DIR", str(tmp_path))
    sun = datetime(2026, 11, 15, 18, tzinfo=UTC)
    tue_game = datetime(2026, 11, 18, 1, tzinfo=UTC)  # Tue 17 Nov 20:00 EST: tuesday pt = Tue 12:00
    games = pl.DataFrame({"game_id": ["a", "b", "c"], "home": ["SEA", "SEA", "SEA"], "away": ["NE", "BUF", "MIA"],
                          "kickoff_utc": [sun - timedelta(days=7), datetime(2026, 11, 17, 16, 30, tzinfo=UTC), tue_game]})
    # Game b kicks off Tue 11:30 EST, so it hasn't ended by Tue 12:00: game c's tuesday row must go.
    g = grid(games, {"c"}, now=tue_game + timedelta(days=1))
    assert set(zip(g["game_id"], g["mode"])) == {("c", "pregame")}
    assert (tmp_path / "logs/features/dropped.jsonl").exists()


def test_targets_are_last_value_at_or_before_kickoff():
    t = targets(pl.DataFrame({"game_id": ["g"], "kickoff_utc": [KICK]}),
                pl.DataFrame([home_row("kalshi", KICK - timedelta(minutes=1), 0.41),
                              home_row("kalshi", KICK, 0.43),
                              home_row("kalshi", KICK + timedelta(minutes=1), 0.9)])).to_dicts()[0]
    assert t["kalshi_close"] == 0.43 and t["poly_close"] is None


@pytest.mark.parametrize("x", [0.0, 1.0])
def test_gap_logit_clips(x):
    f = run([home_row("kalshi", PT - timedelta(minutes=1), 0.5),
             home_row("polymarket", PT - timedelta(minutes=2), x, "snapshot", 0)]).to_dicts()[0]
    assert abs(f["gap_logit"]) == pytest.approx(4.59511985, rel=1e-6)  # logit(0.99)
