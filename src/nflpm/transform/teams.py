"""Canonical franchise table: every team code/name seen in nflverse, Kalshi and Polymarket.

Franchise ids are current nflverse codes. Codes checked 2026-11-18 against nflverse schedules
1999-2026, Kalshi KXNFLGAME market-ticker suffixes (2025-26), and Polymarket Gamma NFL moneyline
slugs and outcome names (2025-26).
"""
from __future__ import annotations

from typing import Literal

# franchise -> (full name, short name, home time zone)
FRANCHISES: dict[str, tuple[str, str, str]] = {
    "ARI": ("Arizona Cardinals", "Cardinals", "America/Phoenix"),
    "ATL": ("Atlanta Falcons", "Falcons", "America/New_York"),
    "BAL": ("Baltimore Ravens", "Ravens", "America/New_York"),
    "BUF": ("Buffalo Bills", "Bills", "America/New_York"),
    "CAR": ("Carolina Panthers", "Panthers", "America/New_York"),
    "CHI": ("Chicago Bears", "Bears", "America/Chicago"),
    "CIN": ("Cincinnati Bengals", "Bengals", "America/New_York"),
    "CLE": ("Cleveland Browns", "Browns", "America/New_York"),
    "DAL": ("Dallas Cowboys", "Cowboys", "America/Chicago"),
    "DEN": ("Denver Broncos", "Broncos", "America/Denver"),
    "DET": ("Detroit Lions", "Lions", "America/Detroit"),
    "GB": ("Green Bay Packers", "Packers", "America/Chicago"),
    "HOU": ("Houston Texans", "Texans", "America/Chicago"),
    "IND": ("Indianapolis Colts", "Colts", "America/Indiana/Indianapolis"),
    "JAX": ("Jacksonville Jaguars", "Jaguars", "America/New_York"),
    "KC": ("Kansas City Chiefs", "Chiefs", "America/Chicago"),
    "LA": ("Los Angeles Rams", "Rams", "America/Los_Angeles"),
    "LAC": ("Los Angeles Chargers", "Chargers", "America/Los_Angeles"),
    "LV": ("Las Vegas Raiders", "Raiders", "America/Los_Angeles"),
    "MIA": ("Miami Dolphins", "Dolphins", "America/New_York"),
    "MIN": ("Minnesota Vikings", "Vikings", "America/Chicago"),
    "NE": ("New England Patriots", "Patriots", "America/New_York"),
    "NO": ("New Orleans Saints", "Saints", "America/Chicago"),
    "NYG": ("New York Giants", "Giants", "America/New_York"),
    "NYJ": ("New York Jets", "Jets", "America/New_York"),
    "PHI": ("Philadelphia Eagles", "Eagles", "America/New_York"),
    "PIT": ("Pittsburgh Steelers", "Steelers", "America/New_York"),
    "SEA": ("Seattle Seahawks", "Seahawks", "America/Los_Angeles"),
    "SF": ("San Francisco 49ers", "49ers", "America/Los_Angeles"),
    "TB": ("Tampa Bay Buccaneers", "Buccaneers", "America/New_York"),
    "TEN": ("Tennessee Titans", "Titans", "America/Chicago"),
    "WAS": ("Washington Commanders", "Commanders", "America/New_York"),
}

# Old nflverse code -> (franchise, first season under the new code, old home time zone)
RELOCATIONS = {
    "STL": ("LA", 2016, "America/Chicago"),
    "SD": ("LAC", 2017, "America/Los_Angeles"),
    "OAK": ("LV", 2020, "America/Los_Angeles"),
}

_ALIASES = {
    "nflverse": {**{c: c for c in FRANCHISES}, **{old: f for old, (f, _, _) in RELOCATIONS.items()}},
    "kalshi": {**{c: c for c in FRANCHISES}, "JAC": "JAX", "LAR": "LA"},
    "poly_slug": {c.lower(): c for c in FRANCHISES},
    "poly_outcome": {**{short: c for c, (_, short, _) in FRANCHISES.items()}, "LAR": "LA"},
}

Source = Literal["nflverse", "kalshi", "poly_slug", "poly_outcome"]


def franchise(code: str, source: Source) -> str:
    """Canonical franchise for a venue's team code or name. KeyError if unknown: never guess."""
    return _ALIASES[source][code.strip()]


def home_tz(team: str, season: int) -> str:
    """Home time zone of a franchise in a season (the Rams were in St. Louis through 2015)."""
    for _, (f, first, tz) in RELOCATIONS.items():
        if f == team and season < first:
            return tz
    return FRANCHISES[team][2]


def aliases(source: Source) -> dict[str, str]:
    """Code -> franchise map for vectorized use (pl.Expr.replace_strict)."""
    return dict(_ALIASES[source])
