"""Match venue events to nflverse games (shared by Kalshi and Polymarket).

Key = unordered team pair + ET kickoff date ± 1 day. Orientation (home/away) always comes from
nflverse, never from venue order, which also covers neutral sites.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from zoneinfo import ZoneInfo

import polars as pl

ET = ZoneInfo("America/New_York")


def match(events: list[dict], games: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    """events: dicts with venue_id, team_a, team_b (franchises), date_et (date); a `reason` key marks
    an event already unmatchable. Returns (matched + game_id/home/away/match_confidence, unmatched + reason)."""
    by_pair: dict[frozenset, list[dict]] = defaultdict(list)
    for g in games.filter(pl.col("kickoff_utc").is_not_null()).iter_rows(named=True):
        by_pair[frozenset((g["home"], g["away"]))].append(
            {**g, "date_et": g["kickoff_utc"].astimezone(ET).date()})
    matched, unmatched = [], []
    for e in events:
        if e.get("reason"):
            unmatched.append(e)
            continue
        cands = [g for g in by_pair[frozenset((e["team_a"], e["team_b"]))]
                 if abs((g["date_et"] - e["date_et"]).days) <= 1]
        if len(cands) != 1:
            unmatched.append({**e, "reason": "multiple_candidates" if cands else "no_candidate",
                              "candidates": ";".join(g["game_id"] for g in cands)})
            continue
        g = cands[0]
        matched.append({**e, "game_id": g["game_id"], "home": g["home"], "away": g["away"],
                        "kickoff_utc": g["kickoff_utc"],
                        "match_confidence": 1.0 if g["date_et"] == e["date_et"] else 0.9})
    dup = {gid for gid, n in Counter(m["game_id"] for m in matched).items() if n > 1}
    unmatched += [{**m, "reason": "game_matched_twice"} for m in matched if m["game_id"] in dup]
    return [m for m in matched if m["game_id"] not in dup], unmatched


def in_season(d: date, games: pl.DataFrame) -> bool:
    """True if `d` is within a day of some season's first-to-last game (excludes preseason)."""
    spans = (games.filter(pl.col("kickoff_utc").is_not_null())
             .group_by("season").agg(pl.col("kickoff_utc").min().alias("lo"), pl.col("kickoff_utc").max().alias("hi")))
    return any((lo.astimezone(ET).date() - d).days <= 1 and (d - hi.astimezone(ET).date()).days <= 1
               for lo, hi in spans.select("lo", "hi").iter_rows())
