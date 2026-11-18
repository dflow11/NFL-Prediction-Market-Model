"""Evaluation. Phase 8: cross-venue lead-lag."""
from __future__ import annotations

import numpy as np
import polars as pl

from nflpm import holdout
from nflpm.features.market import logit

N_BOOT = 2000


def slope_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """OLS slope of y on x with a percentile bootstrap 95% CI (rows resampled)."""
    def slope(i):
        xx, yy = x[i], y[i]
        v = np.var(xx)
        return np.cov(xx, yy, bias=True)[0, 1] / v if v > 0 else np.nan
    rng = np.random.default_rng(seed)
    boots = [slope(rng.integers(0, len(x), len(x))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(slope(np.arange(len(x)))), float(lo), float(hi)


def lead_lag(feats: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Pre-holdout rows: Δk = logit(kalshi_close) − logit(kalshi_mid), Δp likewise for poly, each
    regressed on gap_logit; stratified by poly resolution (≤300s vs ≥1800s)."""
    d = (feats.join(targets.select("game_id", "kalshi_close", "poly_close"), on="game_id")
         .filter(~holdout.is_holdout())
         .drop_nulls(["kalshi_mid", "poly_mid", "kalshi_close", "poly_close", "gap_logit"])
         .with_columns(dk=logit(pl.col("kalshi_close")) - logit(pl.col("kalshi_mid")),
                       dp=logit(pl.col("poly_close")) - logit(pl.col("poly_mid"))))
    strata = {"all": pl.lit(True), "poly_res<=300s": pl.col("poly_resolution_sec") <= 300,
              "poly_res>=1800s": pl.col("poly_resolution_sec") >= 1800}
    rows = []
    for name, cond in strata.items():
        s = d.filter(cond)
        for target in ("dk", "dp"):
            if s.height >= 3:
                b, lo, hi = slope_ci(s["gap_logit"].to_numpy(), s[target].to_numpy())
            else:
                b = lo = hi = float("nan")
            rows.append({"mode": feats["mode"][0] if feats.height else None, "stratum": name, "target": target,
                         "n": s.height, "slope": b, "ci_lo": lo, "ci_hi": hi})
    return pl.DataFrame(rows)
