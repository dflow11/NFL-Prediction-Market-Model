"""Phase 6: price tables from fixture raw files. No network."""
import json
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from nflpm.paths import write_raw
from nflpm.transform.match_games import match
from nflpm.transform.prices import collapse_home, kalshi_candles, polymarket_history, quote

UTC = timezone.utc
KICK = datetime(2026, 10, 11, 13, 30, tzinfo=UTC)  # London, neutral site: JAX "home"
TS = int((KICK - timedelta(hours=1)).timestamp())


def candle(bid, ask, last=None, live=True):
    k = "close_dollars" if live else "close"
    return {"end_period_ts": TS, "yes_bid": {k: bid}, "yes_ask": {k: ask}, "price": {k: last}}


def test_quote_rules():
    assert quote(0.40, 0.44, 0.41) == (0.40, 0.44, pytest.approx(0.42), "bbo")
    assert quote(0.0, 1.0, 0.41) == (None, None, 0.41, "last_trade")  # no book: last trade, flagged
    assert quote(None, None, None) == (None, None, None, None)  # null OHLC is missing, not 0


def test_kalshi_collapse_and_neutral_site_orientation(tmp_path):
    # Venue lists PHI first; nflverse says JAX is home -> orientation must follow nflverse.
    g = pl.DataFrame([dict(game_id="2026_05_PHI_JAX", home="JAX", away="PHI", kickoff_utc=KICK)])
    (m,), _ = match([dict(venue_id="EV", team_a="PHI", team_b="JAX", date_et=date(2026, 10, 11))], g)
    markets = {"PHI": "EV-PHI", "JAX": "EV-JAX"}
    mk = pl.DataFrame([{"game_id": m["game_id"], "home_market": markets[m["home"]],
                        "away_market": markets[m["away"]], "kickoff_utc": KICK}])
    write_raw(tmp_path / "raw/kalshi/candles/EV-JAX/1m/w.json.gz", json.dumps({"candlesticks": [candle("0.6000", "0.6200")]}).encode())
    write_raw(tmp_path / "raw/kalshi/candles/EV-PHI/1m/w.json.gz", json.dumps({"candlesticks": [candle("0.3700", "0.3900", live=False)]}).encode())
    long = pl.DataFrame(kalshi_candles(tmp_path, mk))
    (h,) = collapse_home(long).to_dicts()
    assert h["p_home_mid"] == pytest.approx((0.61 + (1 - 0.38)) / 2)
    assert h["p_home_bid"] == pytest.approx(max(0.60, 1 - 0.39))
    assert h["p_home_ask"] == pytest.approx(min(0.62, 1 - 0.37))
    assert not h["one_side"] and h["quote_source"] == "bbo" and h["resolution_sec"] == 60


def test_one_side_is_used_and_flagged(tmp_path):
    mk = pl.DataFrame([{"game_id": "g", "home_market": "H", "away_market": "A", "kickoff_utc": KICK}])
    write_raw(tmp_path / "raw/kalshi/candles/A/1m/w.json.gz", json.dumps({"candlesticks": [candle("0.3000", "0.3200")]}).encode())
    (h,) = collapse_home(pl.DataFrame(kalshi_candles(tmp_path, mk))).to_dicts()
    assert h["p_home_mid"] == pytest.approx(1 - 0.31) and h["one_side"]


def test_polymarket_complement_and_settlement_exclusion(tmp_path):
    mp = pl.DataFrame([{"game_id": "g", "home_token": "TH", "away_token": "TA", "kickoff_utc": KICK}])
    t0 = int((KICK - timedelta(hours=2)).timestamp())
    after = int((KICK + timedelta(hours=3, minutes=30)).timestamp())
    for tok, price, payout in (("TH", 0.58, 1), ("TA", 0.44, 0)):
        pts = [{"timestamp": t0, "price": price, "resolution_seconds": 60},
               {"timestamp": after, "price": 0.999, "resolution_seconds": 0},  # terminal tick
               {"timestamp": after + 60, "price": payout, "resolution_seconds": 0}]  # settlement
        write_raw(tmp_path / f"raw/polymarket/prices_history/{tok}/60s/c.json.gz", json.dumps({"data": pts}).encode())
    rows, settle = polymarket_history(tmp_path, mp, {"g": KICK + timedelta(hours=3, minutes=20)})
    assert len(rows) == 2  # only the two real prices
    (h,) = collapse_home(pl.DataFrame(rows)).to_dicts()
    assert h["p_home_mid"] == pytest.approx((0.58 + 1 - 0.44) / 2)
    assert h["p_home_bid"] is None and h["quote_source"] == "history"
    assert {(s["side"], s["payout"]) for s in settle} == {("home", 1.0), ("away", 0.0)}


def test_float_noise_is_not_a_cross_and_real_crosses_are_flagged(tmp_path):
    mk = pl.DataFrame([{"game_id": "g", "home_market": "H", "away_market": "A", "kickoff_utc": KICK}])
    write_raw(tmp_path / "raw/kalshi/candles/H/1m/w.json.gz", json.dumps({"candlesticks": [candle("0.2800", "0.2900")]}).encode())
    write_raw(tmp_path / "raw/kalshi/candles/A/1m/w.json.gz", json.dumps({"candlesticks": [candle("0.7100", "0.7200")]}).encode())
    (h,) = collapse_home(pl.DataFrame(kalshi_candles(tmp_path, mk))).to_dicts()
    assert not h["crossed"] and h["p_home_bid"] == 0.28 and h["p_home_ask"] == 0.29  # 1 − 0.71 == 0.29, not above it
    write_raw(tmp_path / "raw/kalshi/candles/H/1m/x.json.gz", json.dumps({"candlesticks": [{**candle("0.3000", "0.3100"), "end_period_ts": TS + 60}]}).encode())
    write_raw(tmp_path / "raw/kalshi/candles/A/1m/x.json.gz", json.dumps({"candlesticks": [{**candle("0.7200", "0.7300"), "end_period_ts": TS + 60}]}).encode())
    rows = collapse_home(pl.DataFrame(kalshi_candles(tmp_path, mk))).sort("ts_utc").to_dicts()
    assert rows[1]["crossed"] and rows[1]["p_home_bid"] is None and rows[1]["p_home_mid"] == pytest.approx(0.29)
