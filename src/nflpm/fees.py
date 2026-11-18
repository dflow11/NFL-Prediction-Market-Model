"""Kalshi taker fee: ceil_to_cent(rate × C × P × (1−P)) per order, effective-dated from config."""
from __future__ import annotations

import math
from datetime import datetime

from nflpm.paths import config, parse_ts


def kalshi_rate(when: datetime) -> float:
    rows = sorted(config()["fees"]["kalshi"], key=lambda r: r["effective_from"] or "")
    rate = None
    for r in rows:
        if r["effective_from"] is None or parse_ts(r["effective_from"]) <= when:
            rate = r["taker_rate"]
    if rate is None:
        raise ValueError(f"no Kalshi fee row effective at {when}")
    return rate


def kalshi_fee(when: datetime, price: float, contracts: int) -> float:
    """Dollars for one order of `contracts` at `price` (round(…, 6) first so float noise can't add a cent)."""
    return math.ceil(round(kalshi_rate(when) * contracts * price * (1 - price) * 100, 6)) / 100
