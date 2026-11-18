# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Phase 0: one snapshot pass of Kalshi markets and Polymarket order books.

    uv run scripts/collect.py      # run by nflpm-collect.timer every 5 minutes

Each venue fails independently; exit code 1 if either failed.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

# Mirrors config.yaml defaults; Phase 1 reads config.yaml (stdlib has no YAML parser).
DATA = Path(os.environ.get("NFLPM_DATA_DIR") or "~/nflpm-data").expanduser()
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
GAMMA = "https://gamma-api.polymarket.com/markets"
BOOKS = "https://clob.polymarket.com/books"
NFL_TAG = 450  # Gamma /sports -> nfl.primaryTagId (checked 2026-11-18)
HORIZON = timedelta(days=8)  # collector.horizon_days
NEAR = timedelta(hours=3)  # collector.near_kickoff_hours
TOKEN_TTL = 3600  # refresh the open-token list hourly
BOOKS_BATCH = 100  # POST /books has no documented max; stay modest
RPS = 5
GAME_SLUG = re.compile(r"nfl-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}")
UTC = timezone.utc


def stamp(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def write_raw(path: Path, body: bytes) -> None:
    """Gzip `body` byte-exact to `path`; atomic, raises FileExistsError instead of overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def new_gamma_dir() -> Path:
    while True:  # pm_archive.py may list Gamma in the same second
        d = DATA / "raw/polymarket/gamma" / stamp(datetime.now(UTC))
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            time.sleep(1)


class Http:
    """Shared client: global rate cap, backoff on 429/5xx honoring Retry-After."""

    def __init__(self) -> None:
        self.c = httpx.Client(timeout=20, headers={"User-Agent": "nflpm-research/0.1"})
        self.last = 0.0

    def req(self, method: str, url: str, **kw) -> httpx.Response:
        for attempt in range(6):
            time.sleep(max(0.0, self.last + 1 / RPS - time.monotonic()))
            self.last = time.monotonic()
            try:
                r = self.c.request(method, url, **kw)
            except httpx.TransportError:
                if attempt == 5:
                    raise
                time.sleep(min(60, 2**attempt))
                continue
            if r.status_code != 429 and r.status_code < 500:
                return r.raise_for_status()
            try:
                wait = float(r.headers["Retry-After"])
            except (KeyError, ValueError):
                wait = 2**attempt
            time.sleep(min(60.0, wait))
        return r.raise_for_status()


def wrap(r: httpx.Response) -> dict:
    """Response body kept byte-exact as text, with when we got it and the server's Date."""
    return {"url": str(r.url), "status": r.status_code, "http_date": r.headers.get("date"),
            "fetched_at_utc": datetime.now(UTC).isoformat(), "body": r.text}


def kalshi(http: Http, now: datetime) -> tuple[list[dict], int]:
    # min/max_close_ts can't be combined with status=open (API docs), and the open
    # KXNFLGAME list is small, so the whole list is stored; horizon filtering happens downstream.
    parts, n, cursor = [], 0, ""
    while True:
        params = dict(series_ticker="KXNFLGAME", status="open", limit=1000)
        r = http.req("GET", KALSHI, params={**params, **({"cursor": cursor} if cursor else {})})
        parts.append(wrap(r))
        body = r.json()
        n += len(body["markets"])
        cursor = body.get("cursor")
        if not cursor:
            return parts, n


def open_tokens(http: Http, now: datetime) -> list[dict]:
    """Open NFL moneyline tokens with game start, cached in state/ and refreshed hourly."""
    f = DATA / "state/pm_open_tokens.json"
    try:
        st = json.loads(f.read_text())
        if now.timestamp() - st["refreshed_at"] < TOKEN_TTL:
            return st["tokens"]
    except FileNotFoundError:
        pass
    d, toks, offset, n = new_gamma_dir(), [], 0, 0
    while True:
        params = dict(tag_id=NFL_TAG, sports_market_types="moneyline", closed="false",
                      order="id", ascending="true", limit=100, offset=offset)
        r = http.req("GET", GAMMA, params=params)
        write_raw(d / f"page_{n}.json.gz", r.content)
        n += 1
        page = r.json()
        if not page:
            break
        offset += len(page)
        toks += [{"token_id": t, "slug": m["slug"], "game_start": m["gameStartTime"]}
                 for m in page if GAME_SLUG.fullmatch(m["slug"]) and m.get("gameStartTime")
                 for t in json.loads(m["clobTokenIds"])]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"refreshed_at": now.timestamp(), "tokens": toks}))
    os.replace(tmp, f)
    return toks


def polymarket(http: Http, now: datetime) -> tuple[list[dict], int]:
    """Books every run within NEAR of kickoff; the rest of the horizon only at minute 0."""
    want = [t["token_id"] for t in open_tokens(http, now)
            if now < ts(t["game_start"]) <= now + HORIZON
            and (ts(t["game_start"]) - now <= NEAR or now.minute == 0)]
    parts = [wrap(http.req("POST", BOOKS, json=[{"token_id": t} for t in want[i:i + BOOKS_BATCH]]))
             for i in range(0, len(want), BOOKS_BATCH)]
    return parts, len(want)


def main() -> None:
    now = datetime.now(UTC)
    http, failed = Http(), False
    logf = DATA / f"logs/collector/{now:%Y%m%d}.jsonl"
    logf.parent.mkdir(parents=True, exist_ok=True)
    for venue, fn in (("kalshi", kalshi), ("polymarket", polymarket)):
        t0 = time.monotonic()
        try:
            parts, n = fn(http, now)
            if parts:
                body = {"fetched_at_utc": now.isoformat(), "responses": parts}
                write_raw(DATA / f"raw/snapshots/{venue}/{now:%Y%m%d}/{now:%H%M%S}Z.json.gz",
                          json.dumps(body).encode())
            rec = {"ok": True, "count": n, "written": bool(parts)}
        except Exception as e:  # one venue's failure must not stop the other
            failed = True
            rec = {"ok": False, "error": repr(e)[:500]}
        with logf.open("a") as log:
            log.write(json.dumps({"ts": now.isoformat(), "venue": venue,
                                  "latency_ms": round((time.monotonic() - t0) * 1000), **rec}) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
