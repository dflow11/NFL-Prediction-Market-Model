"""Unified price tables (`nflpm transform`):
clean/prices_long.parquet, clean/prices_home.parquet, clean/settlements.parquet."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from nflpm.transform.games import game_ends

UTC = timezone.utc
LONG_KEY = ["venue", "market_id", "ts_kind", "resolution_sec", "ts_utc"]
HOME_KEY = ["venue", "game_id", "ts_kind", "resolution_sec", "ts_utc"]
LONG_SCHEMA = {"venue": pl.String, "game_id": pl.String, "market_id": pl.String, "side": pl.String,
               "ts_utc": pl.Datetime("us", "UTC"), "ts_kind": pl.String, "resolution_sec": pl.Int64,
               "bid": pl.Float64, "ask": pl.Float64, "mid": pl.Float64, "last": pl.Float64,
               "quote_source": pl.String}


def _load(f: Path) -> dict:
    return json.loads(gzip.decompress(f.read_bytes()))


def _px(ohlc: dict | None, field: str = "close") -> float | None:
    """Historical candles use `close`, live ones `close_dollars`; null stays null (never 0)."""
    if not ohlc:
        return None
    v = ohlc.get(field, ohlc.get(f"{field}_dollars"))
    return None if v is None else float(v)


def quote(bid: float | None, ask: float | None, last: float | None) -> tuple:
    """(bid, ask, mid, quote_source): bid 0 / ask 0 or ≥1 mean no quote; mid from bid/ask, else last trade."""
    bid = bid if bid else None
    ask = ask if ask and ask < 1 else None
    if bid is not None and ask is not None:
        return bid, ask, (bid + ask) / 2, "bbo"
    if last is not None:
        return bid, ask, last, "last_trade"
    return bid, ask, None, None


def kalshi_candles(root: Path, match: pl.DataFrame) -> list[dict]:
    rows = []
    for r in match.iter_rows(named=True):
        for side in ("home", "away"):
            t = r[f"{side}_market"]
            for f in sorted((root / f"raw/kalshi/candles/{t}").glob("*m/*.json.gz")):
                res = int(f.parent.name[:-1]) * 60
                for c in _load(f)["candlesticks"]:
                    bid, ask, mid, src = quote(_px(c.get("yes_bid")), _px(c.get("yes_ask")), _px(c.get("price")))
                    rows.append({"venue": "kalshi", "game_id": r["game_id"], "market_id": t, "side": side,
                                 "ts_utc": datetime.fromtimestamp(c["end_period_ts"], UTC), "ts_kind": "candle_end",
                                 "resolution_sec": res, "bid": bid, "ask": ask, "mid": mid,
                                 "last": _px(c.get("price")), "quote_source": src})
    return rows


def polymarket_history(root: Path, match: pl.DataFrame, ends: dict[str, datetime]) -> tuple[list[dict], list[dict]]:
    """Price points (resolution_seconds > 0) before game end; settlement = the last resolution-0
    point at/after kickoff (post-game payout, never a price)."""
    rows, settle = [], []
    for r in match.iter_rows(named=True):
        end = ends[r["game_id"]]
        for side in ("home", "away"):
            tok = r[f"{side}_token"]
            final = None
            for f in sorted((root / f"raw/polymarket/prices_history/{tok}").glob("*s/*.json.gz")):
                for p in _load(f)["data"]:
                    ts = datetime.fromtimestamp(p["timestamp"], UTC)
                    if not p.get("resolution_seconds"):
                        if ts >= r["kickoff_utc"] and (final is None or ts > final[0]):
                            final = (ts, p["price"])
                    elif ts < end:
                        rows.append({"venue": "polymarket", "game_id": r["game_id"], "market_id": tok, "side": side,
                                     "ts_utc": ts, "ts_kind": "history_point", "resolution_sec": p["resolution_seconds"],
                                     "bid": None, "ask": None, "mid": float(p["price"]), "last": None,
                                     "quote_source": "history"})
            if final:
                settle.append({"venue": "polymarket", "game_id": r["game_id"], "market_id": tok, "side": side,
                               "payout": float(final[1]), "settled_ts": final[0]})
    return rows, settle


def snapshot_rows(snaps: pl.DataFrame, ends: dict[str, datetime]) -> list[dict]:
    rows = []
    for s in snaps.filter(pl.col("game_id").is_not_null()).iter_rows(named=True):
        if s["ts_utc"] >= ends[s["game_id"]]:
            continue
        bid, ask, mid, src = quote(s["bid"], s["ask"], None)
        rows.append({"venue": s["venue"], "game_id": s["game_id"], "market_id": s["market_or_token"],
                     "side": s["side"], "ts_utc": s["ts_utc"], "ts_kind": "snapshot", "resolution_sec": 0,
                     "bid": bid, "ask": ask, "mid": mid, "last": None, "quote_source": src})
    return rows


def kalshi_settlements(markets: pl.DataFrame, match: pl.DataFrame) -> pl.DataFrame:
    sides = pl.concat([match.select("game_id", market_ticker="home_market", side=pl.lit("home")),
                       match.select("game_id", market_ticker="away_market", side=pl.lit("away"))])
    return (markets.filter(pl.col("settlement_ts").is_not_null()).join(sides, on="market_ticker")
            .select(venue=pl.lit("kalshi"), game_id="game_id", market_id="market_ticker", side="side",
                    payout=pl.col("settlement_value_dollars").cast(pl.Float64),
                    settled_ts=pl.col("settlement_ts").str.to_datetime(time_zone="UTC")))


def collapse_home(long: pl.DataFrame) -> pl.DataFrame:
    """One row per (venue, game, ts_kind, resolution, ts): home probability from both team markets.
    mid = mean(home, 1 − away); bid = max(home_bid, 1 − away_ask); ask = min(home_ask, 1 − away_bid).
    A single available side is used alone and flagged (one_side)."""
    def side(s: str, p: str) -> pl.DataFrame:
        return long.filter(pl.col("side") == s).select(
            *HOME_KEY, **{f"{p}b": "bid", f"{p}a": "ask", f"{p}m": "mid", f"{p}s": "quote_source"})
    j = side("home", "h").join(side("away", "a"), on=HOME_KEY, how="full", coalesce=True)
    return j.select(
        *HOME_KEY,
        p_home_mid=pl.mean_horizontal("hm", 1 - pl.col("am")),
        p_home_bid=pl.max_horizontal("hb", 1 - pl.col("aa")),
        p_home_ask=pl.min_horizontal("ha", 1 - pl.col("ab")),
        quote_source=pl.when((pl.col("hs") == "last_trade") | (pl.col("as") == "last_trade"))
        .then(pl.lit("last_trade")).otherwise(pl.coalesce("hs", "as")),
        one_side=pl.col("hm").is_null() | pl.col("am").is_null(),
    ).filter(pl.col("p_home_mid").is_not_null()).sort(HOME_KEY)


def build_prices(root: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    clean = root / "clean"
    mk, mp = pl.read_parquet(clean / "match_kalshi.parquet"), pl.read_parquet(clean / "match_polymarket.parquet")
    ends = dict(game_ends(pl.concat([mk.select("game_id", "kickoff_utc"), mp.select("game_id", "kickoff_utc")])
                          .unique("game_id")).iter_rows())
    hist, pm_settle = polymarket_history(root, mp, ends)
    snaps = pl.read_parquet(clean / "snapshots.parquet") if (clean / "snapshots.parquet").exists() else None
    rows = kalshi_candles(root, mk) + hist + (snapshot_rows(snaps, ends) if snaps is not None else [])
    long = pl.DataFrame(rows, schema=LONG_SCHEMA).unique(LONG_KEY, keep="first", maintain_order=True).sort(LONG_KEY)
    settle = pl.concat([kalshi_settlements(pl.read_parquet(clean / "kalshi_markets.parquet"), mk),
                        pl.DataFrame(pm_settle, schema={"venue": pl.String, "game_id": pl.String, "market_id": pl.String,
                                                        "side": pl.String, "payout": pl.Float64,
                                                        "settled_ts": pl.Datetime("us", "UTC")})])
    return long, collapse_home(long), settle


def write_prices(root: Path) -> list[Path]:
    long, home, settle = build_prices(root)
    out = [root / "clean/prices_long.parquet", root / "clean/prices_home.parquet", root / "clean/settlements.parquet"]
    for df, p in zip((long, home, settle), out):
        df.write_parquet(p)
    return out
