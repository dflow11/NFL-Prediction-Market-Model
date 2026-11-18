"""Polymarket Gamma discovery and price-history archive (`nflpm archive`)."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from nflpm.http import Http
from nflpm.paths import UTC, append_jsonl, config, data_root, parse_ts, stamp, write_raw

GAMMA = "https://gamma-api.polymarket.com/markets"
HISTORY = "https://data-api.polymarket.com/v2/prices-history"
NFL_TAG = 450  # Gamma /sports -> nfl.primaryTagId (checked 2026-11-18)
FLOOR_DAYS = {60: 7, 300: 60, 1800: 90}  # retention floors; 3h and 12h are permanent
SKIP_AFTER_FLOOR = timedelta(days=14)
PRE, POST = timedelta(days=8), timedelta(hours=6)
GAME_SLUG = re.compile(r"nfl-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}")  # excludes season-series etc.


def new_gamma_dir() -> Path:
    while True:  # archive and collector may list Gamma in the same second
        d = data_root() / "raw/polymarket/gamma" / stamp(datetime.now(UTC))
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            time.sleep(1)


def list_gamma(http: Http, closed: tuple[str, ...], **extra) -> list[dict]:
    """Every NFL game-moneyline market in a Gamma listing; all pages saved raw."""
    out, d, n = {}, new_gamma_dir(), 0
    for c in closed:
        offset = 0
        while True:
            params = dict(tag_id=NFL_TAG, sports_market_types="moneyline", closed=c,
                          order="id", ascending="true", limit=100, offset=offset, **extra)
            page = http.request("GET", GAMMA, policy="nocache", params=params,
                                raw_path=d / f"page_{n}.json.gz").json()
            n += 1
            if not page:
                break
            out.update({m["id"]: m for m in page})
            offset += len(page)
    return [m for m in out.values() if GAME_SLUG.fullmatch(m["slug"]) and m.get("gameStartTime")]


def chunks(start: datetime, created: datetime, now: datetime):
    """UTC-day pieces of [start-8d, start+6h] that are completed and not wholly before creation."""
    ws, we = start - PRE, start + POST
    day = ws.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < we:
        cs, ce = max(day, ws), min(day + timedelta(days=1), we)
        if created < ce <= now:
            yield cs, ce
        day += timedelta(days=1)


def grains(chunk_end: datetime, now: datetime) -> list[int]:
    """Finest to coarsest, dropping grains whose floor the chunk passed more than 14 days ago."""
    return [b for b in config()["polymarket"]["grain_ladder_sec"]
            if b not in FLOOR_DAYS or now - chunk_end <= timedelta(days=FLOOR_DAYS[b]) + SKIP_AFTER_FLOOR]


def history_dir(token: str) -> Path:
    return data_root() / "raw/polymarket/prices_history" / token


def exhausted() -> set[tuple[str, str]]:
    """(token, chunk_start) where every grain came back empty: never retried."""
    out = set()
    for f in (data_root() / "logs/pm_archive").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "all_empty":
                out.add((e["token_id"], e["chunk_start"]))
    return out


def fetch_chunk(http: Http, token: str, cs: datetime, ce: datetime, bucket: int) -> tuple[list[bytes], int]:
    """All pages for one chunk at one grain: (raw pages, number of price points)."""
    params = dict(token_id=token, start=int(cs.timestamp()), end=int(ce.timestamp()),
                  bucket_seconds=bucket)
    pages, points = [], 0
    while True:
        r = http.request("GET", HISTORY, policy="nocache", params=params)
        pages.append(r.content)
        body = r.json()
        # resolution_seconds 0 = settlement/terminal point, not a price at this grain
        points += sum(1 for p in body["data"] if p.get("resolution_seconds"))
        cursor = (body.get("pagination") or {}).get("next_cursor")
        if not cursor:
            return pages, points
        params = {**params, "cursor": cursor}


def report(games: list[dict], empty: set[tuple[str, str]]) -> None:
    ladder = config()["polymarket"]["grain_ladder_sec"]
    print(f"\n{'game':<26} {'kickoff_utc':<17} " + " ".join(f"{b:>6}s" for b in ladder) + "  empty  finest")
    for g in sorted(games, key=lambda g: g["gameStartTime"]):
        toks = json.loads(g["clobTokenIds"])
        counts = {b: sum(1 for t in toks for f in (history_dir(t) / f"{b}s").glob("*.json.gz")
                         if "_p" not in f.name) for b in ladder}
        n_empty = sum(1 for t, _ in empty if t in toks)
        finest = next((f"{b}s" for b in ladder if counts[b]), "-")
        print(f"{g['slug']:<26} {stamp(parse_ts(g['gameStartTime'])):<17} "
              + " ".join(f"{counts[b]:>7}" for b in ladder) + f"  {n_empty:>5}  {finest}")


def archive(only: list[str]) -> int:
    """Archive every completed, not-yet-stored chunk of every started game window. Exit code."""
    now = datetime.now(UTC)
    logf = data_root() / f"logs/pm_archive/{now:%Y%m%d}.jsonl"

    def emit(**e) -> None:
        append_jsonl(logf, {"ts": datetime.now(UTC).isoformat(), **e})

    season = config()["seasons"]["current"]
    season_start = datetime(season, 9, 1, tzinfo=UTC)  # drops preseason
    http = Http()
    games = [g for g in list_gamma(http, ("true", "false"), start_date_min=f"{season}-06-01T00:00:00Z")
             if season_start <= parse_ts(g["gameStartTime"]) <= now + PRE
             and (not only or g["slug"] in only)]
    empty = exhausted()
    requests = saved = errors = 0
    # Most recent first: their 1-min data is the next to expire.
    for g in sorted(games, key=lambda g: g["gameStartTime"], reverse=True):
        for token in json.loads(g["clobTokenIds"]):
            for cs, ce in chunks(parse_ts(g["gameStartTime"]), parse_ts(g["createdAt"]), now):
                key = (token, stamp(cs))
                if key in empty or any(history_dir(token).glob(f"*s/{stamp(cs)}.json.gz")):
                    continue
                ctx = dict(slug=g["slug"], token_id=token, chunk_start=stamp(cs))
                for b in grains(ce, now):
                    try:
                        pages, points = fetch_chunk(http, token, cs, ce, b)
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
    emit(event="run", games=len(games), history_requests=requests, chunks_saved=saved, errors=errors)
    report(games, empty)
    print(f"\ngames={len(games)} history_requests={requests} chunks_saved={saved} errors={errors}")
    return 1 if errors else 0
