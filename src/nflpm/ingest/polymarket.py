"""Polymarket Gamma discovery and price-history archive (`nflpm archive`)."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from nflpm.http import Http
from nflpm.paths import REPO, UTC, append_jsonl, config, data_root, parse_ts, stamp, write_raw
from nflpm.transform.match_games import ET, in_season, match
from nflpm.transform.teams import franchise

GAMMA = "https://gamma-api.polymarket.com/markets"
HISTORY = "https://data-api.polymarket.com/v2/prices-history"
NFL_TAG = 450  # Gamma /sports -> nfl.primaryTagId (checked 2026-11-18)
FLOOR_DAYS = {60: 7, 300: 60, 1800: 90}  # retention floors; 3h and 12h are permanent
SKIP_AFTER_FLOOR = timedelta(days=14)
PRE, POST = timedelta(days=8), timedelta(hours=6)


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
    return [m for m in out.values() if is_game(m)]


def teams(m: dict) -> list[str] | None:
    """Franchises for the two outcomes, in outcome (= clobTokenIds) order; None if not two teams."""
    try:
        outs = json.loads(m["outcomes"])
        return [franchise(o, "poly_outcome") for o in outs] if len(outs) == 2 else None
    except KeyError:
        return None


def is_game(m: dict) -> bool:
    """A game-winner market: both outcomes are NFL teams (season-series etc. are Yes/No)."""
    return bool(m.get("gameStartTime")) and teams(m) is not None


def in_test_era(game_start: str, seasons: tuple[int, ...]) -> bool:
    """Regular/post-season games of `seasons`: on or after Sep 1 of the game's own season (no preseason)."""
    s = season_of(game_start)
    return s in seasons and parse_ts(game_start) >= datetime(s, 9, 1, tzinfo=UTC)


def season_of(game_start: str) -> int:
    t = parse_ts(game_start)
    return t.year if t.month >= 6 else t.year - 1


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
    """Per game: chunks on disk by grain, empty chunks, finest grain; then games by season × finest."""
    ladder = config()["polymarket"]["grain_ladder_sec"]
    by_season: dict[tuple[int, str], int] = {}
    print(f"\n{'game':<30} {'kickoff_utc':<17} " + " ".join(f"{b:>6}s" for b in ladder) + "  empty  finest")
    for g in sorted(games, key=lambda g: g["gameStartTime"]):
        toks = json.loads(g["clobTokenIds"])
        counts = {b: sum(1 for t in toks for f in (history_dir(t) / f"{b}s").glob("*.json.gz")
                         if "_p" not in f.name) for b in ladder}
        n_empty = sum(1 for t, _ in empty if t in toks)
        finest = next((f"{b}s" for b in ladder if counts[b]), "-")
        key = (season_of(g["gameStartTime"]), finest)
        by_season[key] = by_season.get(key, 0) + 1
        print(f"{g['slug']:<30} {stamp(parse_ts(g['gameStartTime'])):<17} "
              + " ".join(f"{counts[b]:>7}" for b in ladder) + f"  {n_empty:>5}  {finest}")
    print("\nseason  finest  games")
    for (season, finest), n in sorted(by_season.items()):
        print(f"{season:<7} {finest:<7} {n}")


def match_polymarket(markets: list[dict]) -> None:
    """clean/polymarket_markets.parquet (with rules text), clean/match_polymarket.parquet,
    reports/unmatched_polymarket.csv; logs the matched-game count next to Kalshi's."""
    from nflpm.ingest.nflverse import read_schedules
    from nflpm.transform.games import build_games

    clean = data_root() / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in markets:
        (ta, tb), (ka, kb) = teams(m), json.loads(m["clobTokenIds"])
        rows.append({"market_id": m["id"], "slug": m["slug"], "question": m.get("question"),
                     "team_a": ta, "team_b": tb, "token_a": ka, "token_b": kb,
                     "game_start": parse_ts(m["gameStartTime"]), "created_at": parse_ts(m["createdAt"]),
                     "closed": m.get("closed"), "uma_resolution_status": m.get("umaResolutionStatus"),
                     "outcome_prices": m.get("outcomePrices"), "description": m.get("description")})
    pl.DataFrame(rows).write_parquet(clean / "polymarket_markets.parquet")

    games = build_games(read_schedules())
    events = [{"venue_id": r["market_id"], "slug": r["slug"], "team_a": r["team_a"], "team_b": r["team_b"],
               "date_et": r["game_start"].astimezone(ET).date()} for r in rows]
    matched, unmatched = match(events, games)
    by_id = {r["market_id"]: r for r in rows}
    out = []
    for e in matched:
        r = by_id[e["venue_id"]]
        tok = {r["team_a"]: r["token_a"], r["team_b"]: r["token_b"]}
        out.append({"game_id": e["game_id"], "market_id": r["market_id"], "slug": r["slug"],
                    "home_token": tok[e["home"]], "away_token": tok[e["away"]],
                    "match_confidence": e["match_confidence"], "kickoff_utc": e["kickoff_utc"]})
    pl.DataFrame(out).write_parquet(clean / "match_polymarket.parquet")
    report_path = REPO / "reports/unmatched_polymarket.csv"
    report_path.parent.mkdir(exist_ok=True)
    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["venue_id", "slug", "date_et", "team_a", "team_b", "reason", "candidates", "in_season"])
        w.writeheader()
        for u in unmatched:
            w.writerow({k: u.get(k, "") for k in w.fieldnames[:-1]} | {"in_season": in_season(u["date_et"], games)})
    kalshi = clean / "match_kalshi.parquet"
    k_games = set(pl.read_parquet(kalshi)["game_id"]) if kalshi.exists() else set()
    p_games = {o["game_id"] for o in out}
    msg = dict(event="match", polymarket_matched=len(p_games), unmatched=len(unmatched),
               kalshi_matched=len(k_games), both=len(p_games & k_games),
               kalshi_only=len(k_games - p_games), polymarket_only=len(p_games - k_games))
    append_jsonl(data_root() / f"logs/pm_archive/{datetime.now(UTC):%Y%m%d}.jsonl",
                 {"ts": datetime.now(UTC).isoformat(), **msg})
    print("\n" + " ".join(f"{k}={v}" for k, v in msg.items() if k != "event"))


def archive(only: list[str]) -> int:
    """Archive every completed, not-yet-stored chunk of every started game window. Exit code."""
    now = datetime.now(UTC)
    logf = data_root() / f"logs/pm_archive/{now:%Y%m%d}.jsonl"

    def emit(**e) -> None:
        append_jsonl(logf, {"ts": datetime.now(UTC).isoformat(), **e})

    current = config()["seasons"]["current"]
    seasons = (current - 1, current)  # the test era: last season + this one
    http = Http()
    games = [g for g in list_gamma(http, ("true", "false"), start_date_min=f"{seasons[0]}-06-01T00:00:00Z")
             if in_test_era(g["gameStartTime"], seasons) and parse_ts(g["gameStartTime"]) <= now + PRE
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
    if not only:  # a slug-limited run must not overwrite the full clean tables
        match_polymarket(games)
    report(games, empty)
    print(f"\ngames={len(games)} history_requests={requests} chunks_saved={saved} errors={errors}")
    return 1 if errors else 0
