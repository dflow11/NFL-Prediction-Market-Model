"""Phase 10: F never fits 2025+, B walk-forward coverage, huge λ makes B = A. No network."""
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import nflpm.models.train as T
from nflpm import holdout

UTC = timezone.utc


def synthetic_feats(seasons=range(2000, 2027), per_season=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for i in range(per_season):
            k = datetime(s, 9, 10, 17, tzinfo=UTC) + timedelta(days=7 * (i // 4), hours=i % 4)
            x = rng.normal()
            rows.append({"game_id": f"{s}_{i}", "mode": "pregame", "kickoff_utc": k, "predict_time": k - timedelta(hours=1),
                         "home_l4_off_epa": x, "away_l4_off_epa": -x, "neutral_site": 0, "away_west_to_east_early": 0,
                         "temp": None, "wind": None, "roof_class": "outdoors", "roof_realized": None,
                         "season": s, "week": 1 + i // 4, "home_win": bool(rng.random() < 1 / (1 + np.exp(-x)))})
    d = pl.DataFrame(rows)
    games = d.select("game_id", "season", "home_win", line_home_prob=pl.lit(0.5))
    return d.drop("season", "week", "home_win"), games, d


def test_F_fit_never_sees_2025_plus(monkeypatch):
    feats, games, d = synthetic_feats()
    seen = []
    for cls in (T.LGBMClassifier, T.LogisticRegression):
        orig = cls.fit
        monkeypatch.setattr(cls, "fit", lambda self, X, y, *a, _o=orig, **k: (seen.append(len(y)), _o(self, X, y, *a, **k))[1])
    T.train_F(feats, games)
    n_train = d.filter(pl.col("season").is_between(2000, 2023)).height
    assert n_train in seen  # learners fit on exactly the 2000-2023 rows
    assert max(seen) == n_train  # nothing larger (no 2024+/2025+ rows) ever reached a fit


def b_rows(lam_weeks=10):
    rng = np.random.default_rng(1)
    rows = []
    start = holdout.start() - timedelta(weeks=lam_weeks)
    for w in range(1, lam_weeks + 3):  # last 2 weeks fall in the holdout
        for i in range(6):
            k = start + timedelta(weeks=w - 1, hours=i)
            km = float(rng.uniform(0.2, 0.8))
            rows.append({"game_id": f"g{w}_{i}", "mode": "pregame", "season": 2025, "week": w, "kickoff_utc": k,
                         "predict_time": k - timedelta(hours=1), "game_end_ts": k + timedelta(hours=3),
                         "home_win": bool(rng.random() < km), "kalshi_mid": km,
                         "poly_mid": None if i == 5 else km + 0.01, "p_F": float(rng.uniform(0.2, 0.8)),
                         "gap_logit": float(rng.normal(0, 0.05)), "poly_staleness_sec": 60.0})
    return pl.DataFrame(rows)


def test_B_predicts_every_post_burn_in_pre_holdout_row():
    d = b_rows()
    b = T.walk_forward_B(d)
    burn = T.config()["model_b"]["burn_in_weeks"]
    want = d.filter((pl.col("week") > burn) & ~holdout.is_holdout() & pl.col("poly_mid").is_not_null())
    assert set(b["game_id"]) == set(want["game_id"])
    assert b["p_B"].is_between(0, 1).all()


def test_B_with_huge_lambda_is_A(monkeypatch):
    cfg = T.config()
    monkeypatch.setitem(cfg, "model_b", {**cfg["model_b"], "l2_lambda": 1e12})
    d = b_rows()
    b = T.walk_forward_B(d).join(d.select("game_id", "kalshi_mid"), on="game_id")
    assert b["p_B"].to_numpy() == pytest.approx(b["kalshi_mid"].to_numpy(), abs=1e-6)
