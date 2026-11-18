# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Phase 0: archive Polymarket NFL moneyline price history before fine grains expire.

    uv run scripts/pm_archive.py [slug ...]   # no slugs = every 2026 game whose window has started

Writes the canonical raw layout (docs/spec.md) so later phases reuse these files.
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
GAMMA = "https://gamma-api.polymarket.com/markets"
HISTORY = "https://data-api.polymarket.com/v2/prices-history"
NFL_TAG = 450  # Gamma /sports -> nfl.primaryTagId (checked 2026-11-18)
SEASON_START = datetime(2026, 9, 1, tzinfo=timezone.utc)  # drops preseason games
GRAINS = [60, 300, 1800, 10800]  # polymarket.grain_ladder_sec
FLOOR = {60: 7, 300: 60, 1800: 90}  # retention floors in days; 10800 is permanent
SKIP_AFTER_FLOOR = timedelta(days=14)
PRE, POST = timedelta(days=8), timedelta(hours=6)
RPS = 5
GAME_SLUG = re.compile(r"nfl-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}")  # excludes season-series etc.
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
        os.link(tmp, path)  # atomic and refuses an existing target, unlike rename
    finally:
        tmp.unlink()


def new_gamma_dir() -> Path:
    while True:  # collect.py may refresh Gamma in the same second
        d = DATA / "raw/polymarket/gamma" / stamp(datetime.now(UTC))
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            time.sleep(1)


class Http:
    """Shared client: global rate cap, backoff on 429/5xx honoring Retry-After."""

    def __init__(self) -> None:
        self.c = httpx.Client(timeout=30, headers={"User-Agent": "nflpm-research/0.1"})
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


def discover(http: Http) -> list[dict]:
    """All 2026 regular/post-season NFL moneyline markets, open and closed. Raw pages saved."""
    out, d, n = {}, new_gamma_dir(), 0
    for closed in ("true", "false"):
        offset = 0
        while True:
            params = dict(tag_id=NFL_TAG, sports_market_types="moneyline", closed=closed,
                          start_date_min="2026-06-01T00:00:00Z", order="id", ascending="true",
                          limit=100, offset=offset)
            r = http.req("GET", GAMMA, params=params)
            write_raw(d / f"page_{n}.json.gz", r.content)
            n += 1
            page = r.json()
            if not page:
                break
            out.update({m["id"]: m for m in page})
            offset += len(page)
    return [m for m in out.values()
            if GAME_SLUG.fullmatch(m["slug"]) and m.get("gameStartTime")
            and ts(m["gameStartTime"]) >= SEASON_START]


def chunks(start: datetime, created: datetime, now: datetime):
    """UTC-day pieces of [start-8d, start+6h], completed and not wholly before market creation."""
    ws, we = start - PRE, start + POST
    day = ws.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < we:
        cs, ce = max(day, ws), min(day + timedelta(days=1), we)
        if created < ce <= now:
            yield cs, ce
        day += timedelta(days=1)


def history_dir(token: str) -> Path:
    return DATA / "raw/polymarket/prices_history" / token


def exhausted() -> set[tuple[str, str]]:
    """(token, chunk_start) pairs where every grain came back empty: never retried."""
    out = set()
    for f in (DATA / "logs/pm_archive").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "all_empty":
                out.add((e["token_id"], e["chunk_start"]))
    return out


def fetch(http: Http, token: str, cs: datetime, ce: datetime, bucket: int) -> tuple[list[bytes], int]:
    """All pages for one chunk at one grain; returns (raw pages, number of price points)."""
    params = dict(token_id=token, start=int(cs.timestamp()), end=int(ce.timestamp()),
                  bucket_seconds=bucket)
    pages, points = [], 0
    while True:
        r = http.req("GET", HISTORY, params=params)
        pages.append(r.content)
        body = r.json()
        # resolution_seconds 0 = settlement/terminal point, not a price at this grain
        points += sum(1 for p in body["data"] if p.get("resolution_seconds"))
        cursor = (body.get("pagination") or {}).get("next_cursor")
        if not cursor:
            return pages, points
        params = {**params, "cursor": cursor}


def report(games: list[dict], empty: set[tuple[str, str]]) -> None:
    print(f"\n{'game':<26} {'kickoff_utc':<17} " + " ".join(f"{b:>6}s" for b in GRAINS)
          + "  empty  finest")
    for g in sorted(games, key=lambda g: g["gameStartTime"]):
        toks = json.loads(g["clobTokenIds"])
        counts = {b: sum(1 for t in toks for f in (history_dir(t) / f"{b}s").glob("*.json.gz")
                         if "_p" not in f.name) for b in GRAINS}
        n_empty = sum(1 for t, _ in empty if t in toks)
        finest = next((f"{b}s" for b in GRAINS if counts[b]), "-")
        print(f"{g['slug']:<26} {stamp(ts(g['gameStartTime'])):<17} "
              + " ".join(f"{counts[b]:>7}" for b in GRAINS) + f"  {n_empty:>5}  {finest}")


def main(only: list[str]) -> None:
    now = datetime.now(UTC)
    logf = DATA / f"logs/pm_archive/{now:%Y%m%d}.jsonl"
    logf.parent.mkdir(parents=True, exist_ok=True)
    log = logf.open("a")

    def emit(**e) -> None:
        log.write(json.dumps({"ts": datetime.now(UTC).isoformat(), **e}) + "\n")
        log.flush()

    http = Http()
    games = [g for g in discover(http)
             if ts(g["gameStartTime"]) - PRE <= now and (not only or g["slug"] in only)]
    empty = exhausted()
    requests = saved = errors = 0
    # Most recent first: their 1-min data is the next to expire.
    for g in sorted(games, key=lambda g: g["gameStartTime"], reverse=True):
        created = ts(g["createdAt"])
        for token in json.loads(g["clobTokenIds"]):
            for cs, ce in chunks(ts(g["gameStartTime"]), created, now):
                key = (token, stamp(cs))
                if key in empty or any(history_dir(token).glob(f"*s/{stamp(cs)}.json.gz")):
                    continue
                ctx = dict(slug=g["slug"], token_id=token, chunk_start=stamp(cs))
                for b in GRAINS:
                    if b in FLOOR and now - ce > timedelta(days=FLOOR[b]) + SKIP_AFTER_FLOOR:
                        continue
                    try:
                        pages, points = fetch(http, token, cs, ce, b)
                    except httpx.HTTPError as e:  # not marked exhausted: retried next run
                        errors += 1
                        emit(event="error", bucket_seconds=b, error=repr(e)[:300], **ctx)
                        break
                    requests += len(pages)
                    if points:
                        base = history_dir(token) / f"{b}s"
                        for i, p in enumerate(pages[1:], 1):
                            write_raw(base / f"{stamp(cs)}_p{i}.json.gz", p)
                        write_raw(base / f"{stamp(cs)}.json.gz", pages[0])  # last: marks complete
                        saved += 1
                        break
                    emit(event="empty", bucket_seconds=b, **ctx)
                else:
                    emit(event="all_empty", **ctx)
                    empty.add(key)
    emit(event="run", games=len(games), history_requests=requests, chunks_saved=saved,
         errors=errors)
    report(games, empty)
    print(f"\ngames={len(games)} history_requests={requests} chunks_saved={saved} errors={errors}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
