"""Phase 7: validation must fail loudly on leakage, holdout rows, tuesday weather, bad prices."""
from datetime import datetime, timedelta, timezone

import polars as pl

from nflpm import holdout
from nflpm.validate.schemas import feature_errors, match_errors, schema_errors, training_matrix_errors

UTC = timezone.utc
PT = datetime(2026, 11, 17, 17, tzinfo=UTC)


def feats(**cols) -> pl.DataFrame:
    base = {"game_id": ["a", "b"], "mode": ["tuesday"] * 2, "predict_time": [PT, PT],
            "kalshi_mid": [0.5, 0.6], "kalshi_staleness_sec": [30.0, 40.0],
            "market_src_max_ts": [PT - timedelta(minutes=1)] * 2}
    return pl.DataFrame({**base, **cols})


def test_clean_features_pass():
    assert feature_errors(feats(), "tuesday") == []


def test_leaky_timestamp_fails():
    errs = feature_errors(feats(market_src_max_ts=[PT - timedelta(minutes=1), PT]), "tuesday")
    assert any("leakage" in e for e in errs)  # equal to predict_time is already a leak


def test_tuesday_weather_fails_but_pregame_weather_is_fine():
    assert any("weather" in e for e in feature_errors(feats(temp=[None, 61.0]), "tuesday"))
    assert feature_errors(feats(temp=[None, None]), "tuesday") == []
    assert feature_errors(feats(mode=["pregame"] * 2, temp=[55.0, 61.0]), "pregame") == []


def test_holdout_rows_and_line_columns_fail_in_training_matrix():
    k = pl.Series([holdout.start() - timedelta(days=1), holdout.start()])
    assert any("holdout" in e for e in training_matrix_errors(pl.DataFrame({"x": [1, 2]}), k))
    assert any("line_" in e for e in training_matrix_errors(pl.DataFrame({"line_spread": [1.0]}), k[:1]))
    assert training_matrix_errors(pl.DataFrame({"x": [1]}), k[:1]) == []
    assert any("label" in e for e in training_matrix_errors(pl.DataFrame({"home_win": [True]}), k[:1]))


def test_feature_hygiene():
    assert any("staleness" in e for e in feature_errors(feats(kalshi_staleness_sec=[None, 1.0]), "tuesday"))
    assert any("eval target" in e for e in feature_errors(feats(kalshi_close=[0.5, 0.5]), "tuesday"))
    assert any("line_" in e for e in feature_errors(feats(line_home_prob=[0.5, 0.5]), "tuesday"))
    assert any("unique" in e or "DUPLICATE" in e.upper() for e in feature_errors(feats(game_id=["a", "a"]), "tuesday"))


def test_prices_out_of_range_and_crossed_fail():
    df = pl.DataFrame({"venue": ["k", "k"], "market_id": ["m", "m"], "ts_kind": ["c", "c"], "resolution_sec": [60, 60],
                       "ts_utc": [PT, PT + timedelta(minutes=1)], "bid": [0.6, 0.1], "ask": [0.5, 0.2],
                       "mid": [0.55, 1.2], "last": [None, None]})
    errs = schema_errors("prices_long", df)
    assert any("bid<=ask" in e for e in errs) and any("in_range" in e for e in errs)


def test_double_matches_fail():
    mk = pl.DataFrame({"game_id": ["g", "g"], "event_ticker": ["E1", "E2"]})
    km = pl.DataFrame({"event_ticker": ["E1", "E1", "E2"]})
    mp = pl.DataFrame({"game_id": ["g"], "market_id": ["M"]})
    errs = match_errors(mk, km, mp)
    assert any("game_id" in e for e in errs) and any("exactly 2" in e for e in errs)
