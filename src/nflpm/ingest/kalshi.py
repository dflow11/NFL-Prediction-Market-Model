"""Kalshi KXNFLGAME: market metadata -> match to games -> candles (`nflpm ingest kalshi`)."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta

import polars as pl

from nflpm.http import Http
from nflpm.ingest.nflverse import read_schedules
from nflpm.paths import REPO, UTC, data_root, parse_ts, stamp
from nflpm.transform.games import MODES, build_games, predict_time
from nflpm.transform.match_games import in_season, match
from nflpm.transform.teams import franchise

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"
MAX_PERIODS = 5000  # per candlestick request (docs)


def list_markets(http: Http, run: str) -> list[dict]:
    """Every KXNFLGAME market, live and historical (pre-cutoff events list no nested markets, so
    the markets listings are the source for both). Raw pages saved; live wins on overlap."""
    out: dict[str, dict] = {}
    for kind, path in (("historical", "/historical/markets"), ("live", "/markets")):
        cursor, n = "", 0
        while True:
            params = dict(series_ticker=SERIES, limit=1000, **({"cursor": cursor} if cursor else {}))
            body = http.request("GET", BASE + path, policy="nocache", params=params,
                                raw_path=data_root() / f"raw/kalshi/markets/{run}/{kind}_page_{n}.json.gz").json()
            n += 1
            out.update({m["ticker"]: {**m, "historical": kind == "historical"} for m in body["markets"]})
            cursor = body.get("cursor")
            if not cursor or not body["markets"]:
                break
    return list(out.values())


def team_of(ticker: str) -> str | None:
    try:
        return franchise(ticker.rsplit("-", 1)[1], "kalshi")
    except KeyError:
        return None


def event_date(event_ticker: str):
    """KXNFLGAME-26SEP17DETBUF -> 2026-09-17 (ET date)."""
    return datetime.strptime(event_ticker.split("-")[1][:7].title(), "%y%b%d").date()


def market_table(markets: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{
        "market_ticker": m["ticker"], "event_ticker": m["event_ticker"], "team": team_of(m["ticker"]),
        "team_code": m["ticker"].rsplit("-", 1)[1], "event_date": event_date(m["event_ticker"]),
        "status": m["status"], "result": m.get("result"),
        "settlement_value_dollars": m.get("settlement_value_dollars"), "settlement_ts": m.get("settlement_ts"),
        "close_time": m.get("close_time"), "expected_expiration_time": m.get("expected_expiration_time"),
        "volume": float(m.get("volume_fp") or 0), "historical": m["historical"],
        "rules_primary": m.get("rules_primary"),
        "rules_secondary": m.get("rules_secondary"),
    } for m in markets], infer_schema_length=None)


def events_for_match(mt: pl.DataFrame) -> list[dict]:
    out = []
    for (ev,), g in mt.group_by("event_ticker"):
        e = {"venue_id": ev, "date_et": g["event_date"][0], "team_codes": ";".join(g["team_code"].sort())}
        teams = g["team"].to_list()
        if len(teams) != 2:
            e["reason"] = "not_2_markets"
        elif None in teams:
            e["reason"] = "unknown_team"
        elif g["volume"].sum() == 0 and g["settlement_ts"].is_not_null().all():
            e["reason"] = "no_volume"  # settled without a trade, e.g. a duplicate listing of a game
        else:
            e["team_a"], e["team_b"] = teams
        out.append(e)
    return sorted(out, key=lambda e: e["venue_id"])


def candle_windows(kickoff: datetime) -> list[tuple[int, datetime, datetime]]:
    """1-min [pt−60m, pt] per mode and [kickoff−3h, kickoff]; 60-min [kickoff−8d, kickoff].
    A window inside another window of the same interval is dropped (no duplicate data)."""
    w = [(1, kickoff - timedelta(hours=3), kickoff), (60, kickoff - timedelta(days=8), kickoff)]
    w += [(1, p - timedelta(minutes=60), p) for p in (predict_time(kickoff, m) for m in MODES)]
    w = list(dict.fromkeys(w))
    w = [x for x in w if not any(y != x and y[0] == x[0] and y[1] <= x[1] and x[2] <= y[2] for y in w)]
    assert all((e - s).total_seconds() / 60 / iv <= MAX_PERIODS for iv, s, e in w)
    return w


def fetch_candles(http: Http, market: dict, kickoff: datetime) -> int:
    """Settled markets only (immutable). Returns network requests made."""
    t = market["market_ticker"]
    path = f"/historical/markets/{t}/candlesticks" if market["historical"] else f"/series/{SERIES}/markets/{t}/candlesticks"
    made = 0
    for iv, s, e in candle_windows(kickoff):
        raw = data_root() / f"raw/kalshi/candles/{t}/{iv}m/{stamp(s)}_{stamp(e)}.json.gz"
        made += not raw.exists()
        http.request("GET", BASE + path, policy="immutable", raw_path=raw,
                     params=dict(start_ts=int(s.timestamp()), end_ts=int(e.timestamp()), period_interval=iv))
    return made


def ingest_kalshi() -> int:
    http, run = Http(), stamp(datetime.now(UTC))
    cutoff = http.request("GET", BASE + "/historical/cutoff", policy="nocache",
                          raw_path=data_root() / f"raw/kalshi/cutoff/{run}.json.gz").json()
    http.request("GET", BASE + f"/series/{SERIES}", policy="nocache",
                 raw_path=data_root() / f"raw/kalshi/series/{SERIES}_{run}.json.gz")
    fees = http.request("GET", BASE + "/series/fee_changes", policy="nocache",
                        params=dict(series_ticker=SERIES, show_historical="true"),
                        raw_path=data_root() / f"raw/kalshi/fee_changes/{SERIES}_{run}.json.gz").json()
    mt = market_table(list_markets(http, run))
    clean = data_root() / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    mt.write_parquet(clean / "kalshi_markets.parquet")

    games = build_games(read_schedules())
    matched, unmatched = match(events_for_match(mt), games)
    rows = []
    for m in matched:
        side = {r["team"]: r["market_ticker"] for r in mt.filter(pl.col("event_ticker") == m["venue_id"]).iter_rows(named=True)}
        rows.append({"game_id": m["game_id"], "event_ticker": m["venue_id"], "home_market": side[m["home"]],
                     "away_market": side[m["away"]], "match_confidence": m["match_confidence"],
                     "kickoff_utc": m["kickoff_utc"]})
    pl.DataFrame(rows).write_parquet(clean / "match_kalshi.parquet")
    report = REPO / "reports/unmatched_kalshi.csv"
    report.parent.mkdir(exist_ok=True)
    with report.open("w", newline="") as f:
        w = csv.DictWriter(f, ["venue_id", "date_et", "team_codes", "reason", "candidates", "in_season"])
        w.writeheader()
        for u in unmatched:
            w.writerow({k: u.get(k, "") for k in w.fieldnames[:-1]} | {"in_season": in_season(u["date_et"], games)})

    settled = {r["market_ticker"]: r for r in mt.filter(pl.col("settlement_ts").is_not_null()).iter_rows(named=True)}
    made = sum(fetch_candles(http, settled[t], r["kickoff_utc"])
               for r in rows for t in (r["home_market"], r["away_market"]) if t in settled)
    n_in = sum(1 for e in events_for_match(mt) if in_season(e["date_et"], games))
    n_unm_in = sum(1 for u in unmatched if in_season(u["date_et"], games))
    print(f"cutoff={cutoff.get('market_settled_ts')} markets={mt.height} events={mt['event_ticker'].n_unique()} "
          f"in_season_events={n_in} matched={len(rows)} unmatched_in_season={n_unm_in} "
          f"unmatched_total={len(unmatched)} match_rate={len(rows) / max(n_in, 1):.4f}")
    print(f"settled_markets={len(settled)} candle_requests={made} fee_changes={fees.get('series_fee_change_arr')}")
    return 0
