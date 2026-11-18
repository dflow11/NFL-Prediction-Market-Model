"""Phase 12: NO-side cost, fee rounding, tie settlement. No network."""
from datetime import datetime, timezone

import polars as pl
import pytest

from nflpm.backtest.paper_trade import decide, settle
from nflpm.fees import kalshi_fee

UTC = timezone.utc
T = datetime(2026, 10, 1, tzinfo=UTC)


def test_fee_rounding_examples_from_spec():
    assert kalshi_fee(T, 0.50, 1) == 0.02
    assert kalshi_fee(T, 0.50, 100) == 1.75  # 175.00000000000003 cents must not round up to 1.76
    assert kalshi_fee(datetime(2025, 9, 1, tzinfo=UTC), 0.50, 100) == 1.75  # pre-2026 row


def test_no_side_cost_uses_one_minus_yes_bid():
    # Home YES ask 0.60, but away YES bid 0.45 -> buying away NO costs 0.55: cheaper equivalent for "home".
    row = {"predict_time": T, "home_bid": 0.30, "home_ask": 0.60, "away_bid": 0.45, "away_ask": 0.70}
    t = decide(row, p_home=0.70)
    assert t["side"] == "home" and t["cost"] == pytest.approx(0.55) and t["leg"].startswith("away_no")
    assert t["contracts"] == 18 and t["fee"] == kalshi_fee(T, 0.55, 18)
    assert t["edge"] == pytest.approx(0.70 - 0.55 - t["fee"] / 18)
    assert decide(row, p_home=0.56) is None  # edge below buffer after fees


def test_ties_settle_at_half(tmp_path):
    trades = pl.DataFrame([{"game_id": "g", "mode": "pregame", "model": "B", "side": "home", "cost": 0.40,
                            "contracts": 25, "fee": 0.42, "kickoff_utc": T}])
    settlements = pl.DataFrame([{"venue": "kalshi", "game_id": "g", "side": "home", "payout": 0.5}])
    games = pl.DataFrame([{"game_id": "g", "home_score": 20, "away_score": 20}])
    (s,) = settle(trades, settlements, games, tmp_path / "log.jsonl").to_dicts()
    assert s["pnl"] == pytest.approx(25 * (0.5 - 0.40) - 0.42)
    assert not (tmp_path / "log.jsonl").exists()  # tie agrees with nflverse


def test_cancelled_settlement_excluded_and_logged(tmp_path):
    trades = pl.DataFrame([{"game_id": "g", "mode": "pregame", "model": "B", "side": "home", "cost": 0.4,
                            "contracts": 25, "fee": 0.42, "kickoff_utc": T}])
    settlements = pl.DataFrame([{"venue": "kalshi", "game_id": "g", "side": "home", "payout": 0.37}])
    games = pl.DataFrame([{"game_id": "g", "home_score": None, "away_score": None}])
    assert settle(trades, settlements, games, tmp_path / "log.jsonl").height == 0
    assert "not settled" in (tmp_path / "log.jsonl").read_text()
