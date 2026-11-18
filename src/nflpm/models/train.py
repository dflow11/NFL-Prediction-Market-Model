"""Model predictions. Market-only models (Phase 8): A, A2; Model 0 is the line baseline."""
from __future__ import annotations

import polars as pl

from nflpm.features.market import inv_logit, logit


def market_models(feats: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """A = kalshi_mid. A2 = inv_logit(½·logit k + ½·logit p), falling back to A (flagged) without poly.
    Model 0 = line_home_prob (baseline only; joined from games, never a feature)."""
    return feats.join(games.select("game_id", "line_home_prob"), on="game_id", how="left").select(
        "game_id", "mode", "kickoff_utc",
        p_A="kalshi_mid",
        p_A2=pl.when(pl.col("poly_mid").is_not_null())
        .then(inv_logit(0.5 * logit(pl.col("kalshi_mid")) + 0.5 * logit(pl.col("poly_mid"))))
        .otherwise("kalshi_mid"),
        a2_fallback=pl.col("poly_mid").is_null() & pl.col("kalshi_mid").is_not_null(),
        p_0="line_home_prob",
    )
