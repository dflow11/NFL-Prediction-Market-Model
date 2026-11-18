"""clean/snapshots.parquet and reports/coverage.csv from raw/snapshots (`nflpm transform`)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import polars as pl

from nflpm.paths import REPO, UTC, data_root, parse_ts

SLOT = timedelta(minutes=5)
WINDOW = timedelta(hours=3)  # coverage window [kickoff − 3h, kickoff]
SCHEMA = {"venue": pl.String, "market_or_token": pl.String, "ts_utc": pl.Datetime("us", "UTC"),
          "bid": pl.Float64, "ask": pl.Float64, "bid_size": pl.Float64, "ask_size": pl.Float64}


def part_ts(part: dict) -> datetime:
    """Server Date header; fetched_at_utc when the header is missing."""
    return parsedate_to_datetime(part["http_date"]) if part.get("http_date") else parse_ts(part["fetched_at_utc"])


def _f(x) -> float | None:
    return None if x is None else float(x)


def kalshi_quotes(part: dict) -> list[dict]:
    out = []
    for m in json.loads(part["body"])["markets"]:
        bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
        out.append({"market_or_token": m["ticker"],
                    "bid": bid if bid else None,  # 0 = no bid
                    "ask": ask if ask and ask < 1 else None,  # 0 / 1.00 = no ask
                    "bid_size": _f(m.get("yes_bid_size_fp")) if bid else None,
                    "ask_size": _f(m.get("yes_ask_size_fp")) if ask and ask < 1 else None})
    return out


def polymarket_quotes(part: dict) -> list[dict]:
    out = []
    for book in json.loads(part["body"]):
        bid = max(book.get("bids") or [], key=lambda l: float(l["price"]), default=None)
        ask = min(book.get("asks") or [], key=lambda l: float(l["price"]), default=None)
        out.append({"market_or_token": book["asset_id"],
                    "bid": _f(bid and bid["price"]), "ask": _f(ask and ask["price"]),
                    "bid_size": _f(bid and bid["size"]), "ask_size": _f(ask and ask["size"])})
    return out


def read_snapshot_quotes(root: Path) -> pl.DataFrame:
    rows = []
    for venue, parse in (("kalshi", kalshi_quotes), ("polymarket", polymarket_quotes)):
        for f in sorted((root / "raw/snapshots" / venue).glob("*/*.json.gz")):
            for part in json.loads(gzip.decompress(f.read_bytes()))["responses"]:
                if part["status"] == 200:
                    ts = part_ts(part)
                    rows += [{"venue": venue, "ts_utc": ts, **q} for q in parse(part)]
    return pl.DataFrame(rows, schema=SCHEMA)


def sides(root: Path) -> pl.DataFrame:
    """market_or_token -> (game_id, side) from the Kalshi and Polymarket match tables."""
    frames = []
    for name, home, away in (("match_kalshi", "home_market", "away_market"),
                             ("match_polymarket", "home_token", "away_token")):
        p = root / f"clean/{name}.parquet"
        if p.exists():
            m = pl.read_parquet(p)
            frames += [m.select("game_id", pl.col(home).alias("market_or_token"), side=pl.lit("home")),
                       m.select("game_id", pl.col(away).alias("market_or_token"), side=pl.lit("away"))]
    return pl.concat(frames) if frames else pl.DataFrame(schema={"game_id": pl.String, "market_or_token": pl.String,
                                                                  "side": pl.String})


def build_snapshots(root: Path) -> pl.DataFrame:
    """Unmatched markets keep a null game_id/side rather than being dropped."""
    q = read_snapshot_quotes(root).join(sides(root), on="market_or_token", how="left")
    return q.select("venue", "game_id", "market_or_token", "side", "ts_utc", "bid", "ask", "bid_size", "ask_size")


def coverage(snaps: pl.DataFrame, games: pl.DataFrame, now: datetime) -> pl.DataFrame:
    """Per game and venue: share of 5-min slots in [kickoff − 3h, kickoff) holding a snapshot of that
    game, for every game kicking off since that venue's collection began. Gaps stay gaps."""
    rows = []
    n_slots = WINDOW // SLOT
    for venue in ("kalshi", "polymarket"):
        v = snaps.filter(pl.col("venue") == venue)
        if v.is_empty():
            continue
        began = v["ts_utc"].min()
        seen = v.filter(pl.col("game_id").is_not_null()).group_by("game_id").agg(pl.col("ts_utc").unique())
        ts_by_game = dict(seen.iter_rows())
        for g in games.filter((pl.col("kickoff_utc") > began) & (pl.col("kickoff_utc") <= now)).iter_rows(named=True):
            start = g["kickoff_utc"] - WINDOW
            hit = {int((t - start) // SLOT) for t in ts_by_game.get(g["game_id"], []) if start <= t < g["kickoff_utc"]}
            missing = [start + i * SLOT for i in range(n_slots) if i not in hit]
            rows.append({"game_id": g["game_id"], "venue": venue, "kickoff_utc": g["kickoff_utc"],
                         "slots": n_slots, "covered": len(hit), "share": len(hit) / n_slots,
                         "under_80": len(hit) / n_slots < 0.8,
                         "missing_slots": ";".join(t.strftime("%H%MZ") for t in missing)})
    return pl.DataFrame(rows, schema={"game_id": pl.String, "venue": pl.String, "kickoff_utc": pl.Datetime("us", "UTC"),
                                      "slots": pl.Int64, "covered": pl.Int64, "share": pl.Float64,
                                      "under_80": pl.Boolean, "missing_slots": pl.String})


def write_snapshots(games: pl.DataFrame) -> Path:
    root = data_root()
    snaps = build_snapshots(root)
    out = root / "clean/snapshots.parquet"
    snaps.write_parquet(out)
    (REPO / "reports").mkdir(exist_ok=True)
    coverage(snaps, games, datetime.now(UTC)).write_csv(REPO / "reports/coverage.csv")
    return out
