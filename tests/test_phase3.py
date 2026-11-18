"""Phase 3: predict times, Kalshi ticker parsing, matching, candle windows. No network."""
from datetime import date, datetime, timedelta, timezone

import polars as pl

from nflpm.ingest.kalshi import candle_windows, event_date, events_for_match, team_of
from nflpm.transform.games import predict_time
from nflpm.transform.match_games import match

UTC = timezone.utc


def test_tuesday_predict_time_is_latest_tuesday_noon_et_strictly_before():
    mnf = datetime(2026, 11, 24, 1, 15, tzinfo=UTC)  # Mon 23 Nov 20:15 EST
    assert predict_time(mnf, "tuesday") == datetime(2026, 11, 17, 17, tzinfo=UTC)  # Tue 17 Nov 12:00 EST
    tue_night = datetime(2026, 11, 18, 0, 0, tzinfo=UTC)  # Tue 17 Nov 19:00 EST
    assert predict_time(tue_night, "tuesday") == datetime(2026, 11, 17, 17, tzinfo=UTC)
    tue_noon = datetime(2026, 11, 17, 17, tzinfo=UTC)  # kickoff exactly Tue 12:00 -> previous week
    assert predict_time(tue_noon, "tuesday") == datetime(2026, 11, 10, 17, tzinfo=UTC)
    sept = datetime(2026, 9, 13, 17, tzinfo=UTC)  # EDT
    assert predict_time(sept, "tuesday") == datetime(2026, 9, 8, 16, tzinfo=UTC)
    assert predict_time(mnf, "pregame") == mnf - timedelta(minutes=60)


def test_kalshi_ticker_parsing():
    assert event_date("KXNFLGAME-26SEP17DETBUF") == date(2026, 9, 17)
    assert team_of("KXNFLGAME-25OCT06KCJAC-JAC") == "JAX"
    assert team_of("KXNFLGAME-26NOV19INDHOU-XXX") is None


def games(*rows) -> pl.DataFrame:
    return pl.DataFrame([dict(game_id=g, home=h, away=a, kickoff_utc=k) for g, h, a, k in rows])


def test_match_orientation_confidence_and_failures():
    g = games(("n", "JAX", "PHI", datetime(2026, 10, 11, 13, 30, tzinfo=UTC)),  # neutral, London
              ("m", "SEA", "NE", datetime(2026, 9, 11, 0, 20, tzinfo=UTC)))  # Thu 20:20 ET Sep 10
    events = [
        dict(venue_id="ev1", team_a="PHI", team_b="JAX", date_et=date(2026, 10, 11)),  # reversed order
        dict(venue_id="ev2", team_a="NE", team_b="SEA", date_et=date(2026, 9, 11)),  # +1 day
        dict(venue_id="ev3", team_a="NE", team_b="BUF", date_et=date(2026, 9, 11)),
        dict(venue_id="ev4", date_et=date(2026, 9, 11), reason="unknown_team"),
    ]
    matched, unmatched = match(events, g)
    by = {m["venue_id"]: m for m in matched}
    assert (by["ev1"]["home"], by["ev1"]["away"], by["ev1"]["match_confidence"]) == ("JAX", "PHI", 1.0)
    assert by["ev2"]["match_confidence"] == 0.9
    assert {u["venue_id"]: u["reason"] for u in unmatched} == {"ev3": "no_candidate", "ev4": "unknown_team"}


def test_game_claimed_twice_goes_to_unmatched():
    g = games(("m", "SEA", "NE", datetime(2026, 9, 11, 0, 20, tzinfo=UTC)))
    ev = [dict(venue_id=v, team_a="NE", team_b="SEA", date_et=date(2026, 9, 10)) for v in ("a", "b")]
    matched, unmatched = match(ev, g)
    assert not matched and {u["reason"] for u in unmatched} == {"game_matched_twice"}


def test_events_need_exactly_two_known_team_markets():
    mt = pl.DataFrame({"event_ticker": ["E1", "E1", "E2", "E3", "E3", "E4", "E4"],
                       "event_date": [date(2026, 9, 10)] * 7,
                       "team": ["NE", "SEA", "NE", "NE", None, "NE", "SEA"],
                       "team_code": ["NE", "SEA", "NE", "NE", "ZZ", "NE", "SEA"],
                       "volume": [5.0, 0, 1, 1, 1, 0, 0],
                       "settlement_ts": ["t"] * 7})
    r = {e["venue_id"]: e.get("reason") for e in events_for_match(mt)}
    assert r == {"E1": None, "E2": "not_2_markets", "E3": "unknown_team", "E4": "no_volume"}


def test_candle_windows_drop_contained_pregame_window():
    k = datetime(2026, 11, 20, 1, 15, tzinfo=UTC)
    w = candle_windows(k)
    assert (1, k - timedelta(hours=3), k) in w and (60, k - timedelta(days=8), k) in w
    assert (1, k - timedelta(hours=2), k - timedelta(hours=1)) not in w  # inside [k−3h, k]
    assert (1, datetime(2026, 11, 17, 16, tzinfo=UTC), datetime(2026, 11, 17, 17, tzinfo=UTC)) in w
    assert len(w) == 3
