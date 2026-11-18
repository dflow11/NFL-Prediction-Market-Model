"""Phase 5: snapshot parsing and coverage. Fixtures written to tmp_path; no network."""
import json
from datetime import datetime, timedelta, timezone

import polars as pl

from nflpm.paths import write_raw
from nflpm.transform.snapshots import build_snapshots, coverage

UTC = timezone.utc
KICK = datetime(2026, 11, 20, 1, 15, tzinfo=UTC)


def snap(root, venue, t: datetime, body, http_date=True):
    part = {"url": "u", "status": 200, "fetched_at_utc": (t + timedelta(seconds=9)).isoformat(),
            "http_date": t.strftime("%a, %d %b %Y %H:%M:%S GMT") if http_date else None, "body": json.dumps(body)}
    write_raw(root / f"raw/snapshots/{venue}/{t:%Y%m%d}/{t:%H%M%S}Z.json.gz",
              json.dumps({"fetched_at_utc": t.isoformat(), "responses": [part]}).encode())


def kalshi_body(bid="0.4200", ask="0.4400"):
    return {"markets": [{"ticker": "K-HOU", "yes_bid_dollars": bid, "yes_ask_dollars": ask,
                         "yes_bid_size_fp": "10.00", "yes_ask_size_fp": "5.00"}]}


def setup(root):
    (root / "clean").mkdir(parents=True)
    pl.DataFrame({"game_id": ["g"], "home_market": ["K-HOU"], "away_market": ["K-IND"]}).write_parquet(
        root / "clean/match_kalshi.parquet")
    pl.DataFrame({"game_id": ["g"], "home_token": ["T-H"], "away_token": ["T-A"]}).write_parquet(
        root / "clean/match_polymarket.parquet")


def test_quotes_parse_best_levels_missing_sides_and_date_header(tmp_path):
    setup(tmp_path)
    t0 = KICK - timedelta(hours=1)
    snap(tmp_path, "kalshi", t0, kalshi_body())
    snap(tmp_path, "kalshi", t0 + timedelta(minutes=5), kalshi_body(bid="0.0000", ask="1.0000"), http_date=False)
    snap(tmp_path, "polymarket", t0, [{"asset_id": "T-A", "bids": [{"price": "0.50", "size": "3"}, {"price": "0.55", "size": "7"}],
                                       "asks": [{"price": "0.60", "size": "2"}, {"price": "0.57", "size": "9"}]}])
    s = build_snapshots(tmp_path).sort("venue", "ts_utc")
    k1, k2, p = s.to_dicts()
    assert (k1["game_id"], k1["side"], k1["bid"], k1["ask"], k1["bid_size"]) == ("g", "home", 0.42, 0.44, 10.0)
    assert k1["ts_utc"] == t0  # Date header, not fetched_at
    assert k2["bid"] is None and k2["ask"] is None and k2["ts_utc"] == t0 + timedelta(minutes=5, seconds=9)
    assert (p["side"], p["bid"], p["bid_size"], p["ask"], p["ask_size"]) == ("away", 0.55, 7.0, 0.57, 9.0)


def test_coverage_counts_gaps_without_interpolation(tmp_path):
    setup(tmp_path)
    start = KICK - timedelta(hours=3)
    for i in range(36):
        if not 10 <= i < 20:  # a 50-minute gap (PC asleep)
            snap(tmp_path, "kalshi", start + i * timedelta(minutes=5) + timedelta(seconds=4), kalshi_body())
    games = pl.DataFrame({"game_id": ["g"], "kickoff_utc": [KICK]})
    c = coverage(build_snapshots(tmp_path), games, now=KICK + timedelta(hours=1)).to_dicts()
    assert len(c) == 1
    row = c[0]
    assert (row["covered"], row["slots"], row["under_80"]) == (26, 36, True)
    missing = row["missing_slots"].split(";")
    assert len(missing) == 10 and missing[0] == (start + timedelta(minutes=50)).strftime("%H%MZ")
