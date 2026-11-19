"""One snapshot pass of Kalshi markets and Polymarket order books (`nflpm collect`)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import httpx

from nflpm.http import Http
from nflpm.ingest.kalshi import FUTURES, SERIES
from nflpm.ingest.polymarket import list_gamma
from nflpm.paths import UTC, append_jsonl, config, data_root, parse_ts, write_raw

KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
BOOKS = "https://clob.polymarket.com/books"
TOKEN_TTL = 3600  # refresh the open-token list hourly
BOOKS_BATCH = 100  # POST /books has no documented max; stay modest


def wrap(r: httpx.Response) -> dict:
    """Response body kept byte-exact as text, with when we got it and the server's Date."""
    return {"url": str(r.url), "status": r.status_code, "http_date": r.headers.get("date"),
            "fetched_at_utc": datetime.now(UTC).isoformat(), "body": r.text}


def kalshi(http: Http, now: datetime) -> tuple[list[dict], int]:
    """Open markets of KXNFLGAME and the playoff futures series (CON/SB games trade there).
    min/max_close_ts can't be combined with status=open (API docs), and each open list is small,
    so whole lists are stored; horizon filtering and game matching happen downstream."""
    parts, n = [], 0
    for series in (SERIES, *(s for v in FUTURES.values() for s in v)):
        cursor = ""
        while True:
            params = dict(series_ticker=series, status="open", limit=1000)
            r = http.request("GET", KALSHI, policy="snapshot",
                             params={**params, **({"cursor": cursor} if cursor else {})})
            parts.append(wrap(r))
            body = r.json()
            n += len(body["markets"])
            cursor = body.get("cursor")
            if not cursor:
                break
    return parts, n


def open_tokens(http: Http, now: datetime) -> list[dict]:
    """Open NFL moneyline tokens with game start; cached in state/, refreshed hourly."""
    f = data_root() / "state/pm_open_tokens.json"
    try:
        st = json.loads(f.read_text())
        if now.timestamp() - st["refreshed_at"] < TOKEN_TTL:
            return st["tokens"]
    except FileNotFoundError:
        pass
    toks = [{"token_id": t, "slug": m["slug"], "game_start": m["gameStartTime"]}
            for m in list_gamma(http, ("false",)) for t in json.loads(m["clobTokenIds"])]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"refreshed_at": now.timestamp(), "tokens": toks}))
    os.replace(tmp, f)
    return toks


def polymarket(http: Http, now: datetime) -> tuple[list[dict], int]:
    """Books every run within near_kickoff_hours of kickoff; the rest of the horizon at minute 0."""
    c = config()["collector"]
    horizon, near = timedelta(days=c["horizon_days"]), timedelta(hours=c["near_kickoff_hours"])
    want = [t["token_id"] for t in open_tokens(http, now)
            if now < parse_ts(t["game_start"]) <= now + horizon
            and (parse_ts(t["game_start"]) - now <= near or now.minute == 0)]
    parts = [wrap(http.request("POST", BOOKS, policy="snapshot",
                               json=[{"token_id": t} for t in want[i:i + BOOKS_BATCH]]))
             for i in range(0, len(want), BOOKS_BATCH)]
    return parts, len(want)


def collect() -> int:
    """Both venues, independently. Exit code 1 if either failed."""
    now = datetime.now(UTC)
    http, failed = Http(), False
    for venue, fn in (("kalshi", kalshi), ("polymarket", polymarket)):
        t0 = time.monotonic()
        try:
            parts, n = fn(http, now)
            if parts:
                body = {"fetched_at_utc": now.isoformat(), "responses": parts}
                write_raw(data_root() / f"raw/snapshots/{venue}/{now:%Y%m%d}/{now:%H%M%S}Z.json.gz",
                          json.dumps(body).encode())
            rec = {"ok": True, "count": n, "written": bool(parts)}
        except Exception as e:  # one venue's failure must not stop the other
            failed = True
            rec = {"ok": False, "error": repr(e)[:500]}
        append_jsonl(data_root() / f"logs/collector/{now:%Y%m%d}.jsonl",
                     {"ts": now.isoformat(), "venue": venue,
                      "latency_ms": round((time.monotonic() - t0) * 1000), **rec})
    return 1 if failed else 0
