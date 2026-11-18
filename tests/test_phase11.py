"""Phase 11: every report section renders on fixtures; holdout eval refuses without a manifest."""
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from nflpm import holdout
from nflpm.models import evaluate as E

UTC = timezone.utc


def fixture_frame(n=120, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.2, 0.85, n)
    rows = []
    for i in range(n):
        kick = holdout.start() - timedelta(days=200) + timedelta(days=i)
        y = bool(rng.random() < k[i])
        rows.append({"game_id": f"g{i}", "mode": "tuesday", "season": 2025, "week": 1 + i // 14, "kickoff_utc": kick,
                     "predict_time": kick - timedelta(days=3), "home_win": None if i == 0 else y,
                     "p_A": k[i], "p_A2": k[i] + 0.005, "a2_fallback": False, "p_0": k[i] - 0.01,
                     "p_F": float(np.clip(k[i] + rng.normal(0, 0.05), 0.05, 0.95)), "p_F_lr": k[i], "p_F_lgbm": k[i],
                     "p_B": None if i < 10 else k[i] + 0.002, "b_fit_n": 50,
                     "kalshi_mid": k[i], "poly_mid": float(np.clip(k[i] + rng.normal(0, 0.02), 0.01, 0.99)),
                     "gap_logit": float(rng.normal(0, 0.05)), "poly_resolution_sec": 10800 if i % 2 else 300,
                     "poly_source": "history", "kalshi_close": float(np.clip(k[i] + rng.normal(0, 0.03), 0.01, 0.99)),
                     "poly_close": float(np.clip(k[i] + rng.normal(0, 0.03), 0.01, 0.99)), "line_home_prob": k[i] - 0.01})
    return pl.DataFrame(rows)


def test_every_section_renders(tmp_path):
    text = E.render(fixture_frame(), "tuesday", "Evaluation", tmp_path / "cal.png", tmp_path / "none.csv")
    for h in ("## 1. Coverage", "## 2. Primary — closing-line value", "## 3. Primary — lead-lag", "## 4. Secondary",
              "## 5. Calibration", "## 6. Favorite–longshot", "## 7. Disagreement", "## 8. Caveats"):
        assert h in text, h
    assert "line (alt. close)" in text  # tuesday mode uses the line as an alternate close
    assert "inconclusive" in text or "lower" in text or "higher" in text
    assert (tmp_path / "cal.png").exists()


def test_scoring_set_drop_log():
    s, drops = E.scoring_set(fixture_frame())
    assert dict(drops)["tie or unplayed"] == 1 and dict(drops)["no B (burn-in, no Polymarket, or missing input)"] == 9
    assert s.height == 110


def test_stats_helpers():
    assert E.verdict(-0.1, 0.2) == "inconclusive" and E.verdict(-0.3, -0.1) == "lower"
    lo, hi = E.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97
    assert E.mde(np.array([1.0, -1.0] * 50)) == pytest.approx(2.8 * np.std([1.0, -1.0] * 50, ddof=1) / 10)


def test_holdout_eval_refuses_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("NFLPM_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="no models/frozen/manifest.json"):
        E.evaluate_holdout("pregame")


def test_holdout_eval_refuses_on_mismatched_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("NFLPM_DATA_DIR", str(tmp_path))
    p = E.manifest_path()
    p.parent.mkdir(parents=True)
    p.write_text('{"git_commit": "deadbeef", "config_sha256": "x", "models": {}, "frozen_at": "t"}')
    with pytest.raises(SystemExit, match="does not match"):
        E.evaluate_holdout("pregame")
