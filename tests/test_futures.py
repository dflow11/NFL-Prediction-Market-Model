"""Playoff futures: CON/SB games map to the finalists' champion markets; pre-matchup candles dropped."""
import gzip
import json
from datetime import datetime, timedelta, timezone

import polars as pl

from nflpm.ingest.kalshi import futures_matches, market_table
from nflpm.transform.prices import kalshi_candles

UTC = timezone.utc
K = datetime(2026, 2, 8, 23, 30, tzinfo=UTC)


def mkt(team, settled=None, event="KXSB-26"):
    return {"ticker": f"{event}-{team}", "event_ticker": event, "status": "finalized", "historical": True,
            "open_time": "2025-02-12T15:00:00Z", "settlement_ts": settled}


def test_super_bowl_maps_to_finalists_once_matchup_is_set():
    mt = market_table([mkt("SEA", "2026-02-09T03:30:06Z"), mkt("NE", "2026-02-09T03:30:06Z"),
                       mkt("DEN", "2026-01-25T23:09:50Z"), mkt("LA", "2026-01-26T03:02:26Z"),
                       mkt("PHI", "2025-02-10T03:26:25Z", event="KXSB-25"), mkt("NE", "2025-02-10T03:26:25Z", event="KXSB-25"),
                       mkt("SEA", "2025-02-10T03:26:25Z", event="KXSB-25")])
    games = pl.DataFrame([{"game_id": "2025_22_SEA_NE", "season": 2025, "game_type": "SB", "home": "NE", "away": "SEA",
                           "kickoff_utc": K}, {"game_id": "2025_10_X", "season": 2025, "game_type": "REG",
                                               "home": "NE", "away": "SEA", "kickoff_utc": K}])
    rows, missing = futures_matches(mt, games, taken=set())
    assert missing == [] and len(rows) == 1  # KXSB-25 settled before kickoff -> not a candidate
    r = rows[0]
    assert (r["event_ticker"], r["home_market"], r["away_market"]) == ("KXSB-26", "KXSB-26-NE", "KXSB-26-SEA")
    assert r["matchup_set_ts"] == datetime(2026, 1, 26, 3, 2, 26, tzinfo=UTC)  # last other team eliminated
    assert futures_matches(mt, games, taken={"2025_22_SEA_NE"}) == ([], [])  # already a KXNFLGAME event


def test_candles_before_matchup_set_are_dropped(tmp_path):
    t0 = datetime(2026, 1, 26, 3, tzinfo=UTC)
    for t in ("KXSB-26-NE", "KXSB-26-SEA"):
        d = tmp_path / f"raw/kalshi/candles/{t}/60m"
        d.mkdir(parents=True)
        candles = [{"end_period_ts": int((t0 + timedelta(hours=h)).timestamp()), "yes_bid": {"close": 0.4},
                    "yes_ask": {"close": 0.42}, "price": {"close": 0.41}} for h in range(3)]
        (d / "w.json.gz").write_bytes(gzip.compress(json.dumps({"candlesticks": candles}).encode()))
    match = pl.DataFrame([{"game_id": "g", "home_market": "KXSB-26-NE", "away_market": "KXSB-26-SEA",
                           "matchup_set_ts": t0 + timedelta(minutes=30)}])
    rows = kalshi_candles(tmp_path, match)
    assert {r["ts_utc"] for r in rows} == {t0 + timedelta(hours=1), t0 + timedelta(hours=2)}
