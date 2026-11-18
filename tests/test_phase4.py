"""Phase 4: Polymarket game-market detection and season assignment. No network."""
import json

from nflpm.ingest.polymarket import is_game, season_of, teams


def market(outcomes, slug="nfl-x-y-2025-11-16", start="2025-11-16 21:25:00+00"):
    return {"slug": slug, "outcomes": json.dumps(outcomes), "gameStartTime": start}


def test_game_markets_are_two_team_outcomes_whatever_the_slug():
    assert teams(market(["Chiefs", "Broncos "], slug="nfl-kc-den-2025-11-16-656")) == ["KC", "DEN"]
    assert teams(market(["LAR", "Seahawks"])) == ["LA", "SEA"]
    assert not is_game(market(["Yes", "No"], slug="will-the-rams-win-the-season-series"))
    assert not is_game(market(["Chiefs", "Broncos"], start=None))
    assert not is_game(market(["Chiefs", "Broncos", "Draw"]))


def test_season_of_game_start():
    assert season_of("2025-09-05 00:20:00+00") == 2025
    assert season_of("2026-02-08 23:30:00+00") == 2025  # Super Bowl belongs to the prior season
    assert season_of("2026-09-10 00:20:00+00") == 2026
