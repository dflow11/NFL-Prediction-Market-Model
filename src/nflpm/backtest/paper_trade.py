"""Kalshi paper trading (`nflpm papertrade`): hold to settlement, one decision per game per mode at
predict_time from the 1-min candles ending before it. Paper-only research; small samples."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from nflpm import holdout
from nflpm.fees import kalshi_fee
from nflpm.models.evaluate import mean_ci, md_table
from nflpm.paths import REPO, append_jsonl, config, data_root
from nflpm.transform.games import MODES

LABEL = "paper-only, small sample (ROI CI ≈ ±20% at 100 trades)"


def quotes_at(decisions: pl.DataFrame, long: pl.DataFrame, match: pl.DataFrame) -> pl.DataFrame:
    """Per decision row: home/away yes bid/ask from each market's last 1-min candle ending before predict_time."""
    c = long.filter((pl.col("venue") == "kalshi") & (pl.col("ts_kind") == "candle_end") & (pl.col("resolution_sec") == 60))
    out = decisions.join(match.select("game_id", "home_market", "away_market"), on="game_id")
    for side in ("home", "away"):
        q = c.select(**{f"{side}_market": "market_id"}, ts_utc="ts_utc", **{f"{side}_bid": "bid", f"{side}_ask": "ask"}).sort("ts_utc")
        out = (out.sort("predict_time").join_asof(q, left_on="predict_time", right_on="ts_utc", by=f"{side}_market",
                                                  strategy="backward", allow_exact_matches=False, check_sortedness=False)
               .rename({"ts_utc": f"{side}_quote_ts"}))
    return out


def decide(row: dict, p_home: float) -> dict | None:
    """Cheapest equivalent contract per side, taker fee, edge; trade the larger edge above edge_buffer."""
    cfg = config()["paper_trade"]
    hb, ha, ab, aa = row["home_bid"], row["home_ask"], row["away_bid"], row["away_ask"]
    legs = {"home": [(ha, "home_yes_ask"), (None if ab is None else 1 - ab, "away_no=1-away_yes_bid")],
            "away": [(aa, "away_yes_ask"), (None if hb is None else 1 - hb, "home_no=1-home_yes_bid")]}
    best = None
    for side, opts in legs.items():
        opts = [(c, how) for c, how in opts if c is not None and 0 < c < 1]
        if not opts:
            continue
        cost, how = min(opts)
        n = math.floor(cfg["stake_usd"] / cost)
        if n < 1:
            continue
        fee = kalshi_fee(row["predict_time"], cost, n)
        p = p_home if side == "home" else 1 - p_home
        edge = p - cost - fee / n
        if best is None or edge > best["edge"]:
            best = {"side": side, "cost": cost, "leg": how, "contracts": n, "fee": fee, "edge": edge}
    return best if best and best["edge"] > cfg["edge_buffer"] else None


def top_size(trade: dict, row: dict, snaps: pl.DataFrame) -> float | None:
    """Top-of-book size for the leg bought, from a snapshot within max_snapshot_age_sec before predict_time."""
    market = row[f"{trade['leg'].split('_')[0]}_market"]
    col = "ask_size" if trade["leg"].endswith("_ask") else "bid_size"
    max_age = config()["collector"]["max_snapshot_age_sec"]
    s = snaps.filter((pl.col("market_or_token") == market) & (pl.col("ts_utc") < row["predict_time"])
                     & ((row["predict_time"] - pl.col("ts_utc")).dt.total_seconds() <= max_age))
    return None if s.is_empty() else s.sort("ts_utc")[col][-1]


def settle(trades: pl.DataFrame, settlements: pl.DataFrame, games: pl.DataFrame, log: Path) -> pl.DataFrame:
    """Kalshi market results: payout per contract of the side bought (ties $0.50). Fair-price /
    cancelled settlements are excluded and logged; disagreements with nflverse scores are logged."""
    k = settlements.filter(pl.col("venue") == "kalshi").select("game_id", side="side", payout="payout")
    t = trades.join(k, on=["game_id", "side"], how="left").join(games.select("game_id", "home_score", "away_score"), on="game_id")
    keep = []
    for r in t.iter_rows(named=True):
        if r["payout"] not in (0.0, 0.5, 1.0):
            append_jsonl(log, {"game_id": r["game_id"], "mode": r["mode"], "model": r["model"], "reason": "not settled 0/0.5/1",
                               "payout": r["payout"]})
            continue
        if r["home_score"] is not None:
            margin = (r["home_score"] - r["away_score"]) * (1 if r["side"] == "home" else -1)
            want = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            if want != r["payout"]:
                append_jsonl(log, {"game_id": r["game_id"], "reason": "Kalshi result != nflverse score",
                                   "payout": r["payout"], "nflverse": want})
        keep.append(r)
    if not keep:
        return pl.DataFrame()
    s = pl.DataFrame(keep)
    return s.with_columns(staked=pl.col("contracts") * pl.col("cost") + pl.col("fee"),
                          pnl=pl.col("contracts") * (pl.col("payout") - pl.col("cost")) - pl.col("fee"))


def run(preds: pl.DataFrame, long: pl.DataFrame, match: pl.DataFrame, snaps: pl.DataFrame, targets: pl.DataFrame,
        settlements: pl.DataFrame, games: pl.DataFrame, log: Path) -> pl.DataFrame:
    rows = quotes_at(preds.filter(~holdout.is_holdout()), long, match)
    rows = rows.join(targets.select("game_id", "kalshi_close"), on="game_id", how="left")
    trades = []
    for r in rows.iter_rows(named=True):
        for model in ("B", "F"):
            p = r[f"p_{model}"]
            if p is None:
                continue
            t = decide(r, p)
            if t:
                size = top_size(t, r, snaps)
                close = None if r["kalshi_close"] is None else (r["kalshi_close"] if t["side"] == "home" else 1 - r["kalshi_close"])
                trades.append({"game_id": r["game_id"], "mode": r["mode"], "model": model, "kickoff_utc": r["kickoff_utc"],
                               **t, "top_size": size, "size_unknown": size is None,
                               "size_short": size is not None and size < t["contracts"],
                               "clv": None if close is None else close - t["cost"]})
    return settle(pl.DataFrame(trades), settlements, games, log) if trades else pl.DataFrame()


def summary(s: pl.DataFrame) -> dict:
    pnl, staked = s["pnl"].to_numpy(), s["staked"].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pnl), (2000, len(pnl)))
    roi_b = pnl[idx].sum(1) / staked[idx].sum(1)
    cum = np.cumsum(s.sort("kickoff_utc")["pnl"].to_numpy())
    clv = s["clv"].drop_nulls().to_numpy()
    c, clo, chi = mean_ci(clv)
    return {"trades": len(pnl), "hit_rate": float((s["payout"] == 1).mean()), "pnl_usd": float(pnl.sum()),
            "roi": float(pnl.sum() / staked.sum()), "roi_lo": float(np.percentile(roi_b, 2.5)),
            "roi_hi": float(np.percentile(roi_b, 97.5)), "max_drawdown_usd": float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max()),
            "mean_clv": c, "clv_lo": clo, "clv_hi": chi, "size_unknown": int(s["size_unknown"].sum()),
            "size_short": int(s["size_short"].sum())}


def report(trades: pl.DataFrame, mode: str) -> str:
    rows = [{"model": m, **summary(trades.filter(pl.col("model") == m))}
            for m in ("B", "F") if trades.height and trades.filter(pl.col("model") == m).height]
    body = md_table(pl.DataFrame(rows)) if rows else "No trades."
    return (f"# Paper trading (Kalshi) — {mode}\n\n**{LABEL}.** Pre-holdout rows only; hold to settlement; "
            f"taker fees; stake ${config()['paper_trade']['stake_usd']}, edge buffer {config()['paper_trade']['edge_buffer']}. "
            "F is a sanity check.\n\n" + body + "\n")


def papertrade() -> list[Path]:
    root, out = data_root(), []
    clean = root / "clean"
    long, match = pl.read_parquet(clean / "prices_long.parquet"), pl.read_parquet(clean / "match_kalshi.parquet")
    snaps, settlements = pl.read_parquet(clean / "snapshots.parquet"), pl.read_parquet(clean / "settlements.parquet")
    games, targets = pl.read_parquet(clean / "games.parquet"), pl.read_parquet(root / "features/targets.parquet")
    for mode in MODES:
        preds = pl.read_parquet(root / f"models/preds_{mode}.parquet")
        trades = run(preds, long, match, snaps, targets, settlements, games, root / "logs/papertrade/excluded.jsonl")
        if trades.height:
            trades.write_parquet(root / f"models/trades_{mode}.parquet")
        out.append(REPO / f"reports/papertrade_{mode}.md")
        out[-1].write_text(report(trades, mode))
    return out
