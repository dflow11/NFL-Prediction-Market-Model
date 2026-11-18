"""Models (`nflpm train`): market-only A/A2/0, football model F (frozen), offset model B (walk-forward)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from scipy.special import logit as np_logit
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nflpm import holdout
from nflpm.features.market import CLIP, inv_logit, logit
from nflpm.models import offset_logit
from nflpm.paths import config, data_root
from nflpm.transform.games import MODES, game_ends
from nflpm.validate.schemas import training_matrix_errors

F_TRAIN = range(2000, 2024)  # frozen: never refit on 2025+
F_CALIB = 2024
CONTEXT = ["neutral_site", "away_west_to_east_early", "temp", "wind"]


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


def football_matrix(f: pl.DataFrame) -> pl.DataFrame:
    """Team-form + context columns as floats (no market, line, id or timestamp columns)."""
    cols = [c for c in f.columns if c.startswith(("home_", "away_")) and not c.endswith("_src_max_ts")]
    return f.select(
        pl.col(cols + [c for c in CONTEXT if c in f.columns]).cast(pl.Float64),
        *[(pl.col("roof_class") == v).cast(pl.Float64).alias(f"roof_{v}") for v in ("dome", "outdoors", "retractable")],
        *[(pl.col("roof_realized") == v).cast(pl.Float64).alias(f"roof_is_{v}") for v in ("open", "closed")],
    )


def _calibrate(p: np.ndarray, y: np.ndarray):
    """Platt or isotonic, chosen by 2-fold log loss within the calibration season only; refit on all of it."""
    fits = {"platt": lambda: LogisticRegression(),
            "isotonic": lambda: IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)}
    z = np_logit(np.clip(p, *CLIP)).reshape(-1, 1)
    half = np.arange(len(y)) % 2 == 0
    score = {}
    for name, make in fits.items():
        losses = []
        for tr in (half, ~half):
            m = make().fit(z[tr] if name == "platt" else p[tr], y[tr])
            q = m.predict_proba(z[~tr])[:, 1] if name == "platt" else m.predict(p[~tr])
            losses.append(log_loss(y[~tr], np.clip(q, 1e-6, 1 - 1e-6)))
        score[name] = float(np.mean(losses))
    kind = min(score, key=score.get)
    return kind, (fits[kind]().fit(z if kind == "platt" else p, y)), score


def _apply_cal(kind: str, cal, p: np.ndarray) -> np.ndarray:
    if kind == "platt":
        return cal.predict_proba(np_logit(np.clip(p, *CLIP)).reshape(-1, 1))[:, 1]
    return cal.predict(p)


def train_F(feats: pl.DataFrame, games: pl.DataFrame) -> dict:
    """Fit LR and LightGBM on F_TRAIN, calibrate each on F_CALIB; F = the lower calibrated 2024 log loss."""
    d = feats.join(games.select("game_id", "season", "home_win"), on="game_id").filter(pl.col("home_win").is_not_null())
    X, y = football_matrix(d), d["home_win"].cast(pl.Int8).to_numpy()
    tr, ca = d["season"].is_in(list(F_TRAIN)).to_numpy(), (d["season"] == F_CALIB).to_numpy()
    errs = training_matrix_errors(X.filter(pl.Series(tr)), d.filter(pl.Series(tr))["kickoff_utc"])
    if errs:
        raise ValueError(errs)
    Xn = X.to_numpy()
    learners = {
        "lr": make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), StandardScaler(),
                            LogisticRegression(max_iter=5000)),
        "lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1),
    }
    out = {"columns": X.columns, "models": {}, "train_seasons": [F_TRAIN.start, F_TRAIN.stop - 1], "calib_season": F_CALIB}
    for name, m in learners.items():
        m.fit(Xn[tr], y[tr])
        kind, cal, scores = _calibrate(m.predict_proba(Xn[ca])[:, 1], y[ca])
        out["models"][name] = {"model": m, "cal_kind": kind, "cal": cal, "cal_cv_logloss": scores}
    out["chosen"] = min(out["models"], key=lambda n: min(out["models"][n]["cal_cv_logloss"].values()))
    return out


def predict_F(bundle: dict, feats: pl.DataFrame) -> dict[str, np.ndarray]:
    X = football_matrix(feats).select(bundle["columns"]).to_numpy()
    preds = {n: _apply_cal(m["cal_kind"], m["cal"], m["model"].predict_proba(X)[:, 1]) for n, m in bundle["models"].items()}
    return {"p_F": preds[bundle["chosen"]], **{f"p_F_{n}": p for n, p in preds.items()}}


def b_design(d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """x = [1, F_logit − logit(kalshi_mid), gap_logit, log1p(poly_staleness_sec)]; offset = logit(kalshi_mid)."""
    k = d.select(logit(pl.col("kalshi_mid"))).to_series().to_numpy()
    f = np_logit(np.clip(d["p_F"].to_numpy(), *CLIP))
    x = np.column_stack([np.ones(len(d)), f - k, d["gap_logit"].to_numpy(), np.log1p(d["poly_staleness_sec"].to_numpy())])
    return x, k


def walk_forward_B(d: pl.DataFrame, include_holdout: bool = False) -> pl.DataFrame:
    """Weekly refit on rows with game_end_ts < the week's first predict_time; the first burn_in_weeks of the
    first market season get no prediction. Holdout rows are neither fit nor predicted unless include_holdout."""
    cfg = config()["model_b"]
    ok = d.filter(pl.col("kalshi_mid").is_not_null() & pl.col("poly_mid").is_not_null() & pl.col("p_F").is_not_null()
                  & pl.col("poly_staleness_sec").is_not_null())
    if not include_holdout:
        ok = ok.filter(~holdout.is_holdout())
    first_season = ok["season"].min() if ok.height else None
    out = []
    for (season, week), wk in ok.group_by("season", "week", maintain_order=True):
        if season == first_season and week <= cfg["burn_in_weeks"]:
            continue
        fit_rows = ok.filter((pl.col("game_end_ts") < wk["predict_time"].min()) & pl.col("home_win").is_not_null())
        errs = training_matrix_errors(pl.DataFrame({"x": [0] * fit_rows.height}), fit_rows["kickoff_utc"]) if not include_holdout else []
        if errs:
            raise ValueError(errs)
        x, off = b_design(fit_rows)
        w = offset_logit.fit(x, off, fit_rows["home_win"].cast(pl.Float64).to_numpy(), cfg["l2_lambda"])
        xw, offw = b_design(wk)
        out.append(wk.select("game_id", "mode").with_columns(p_B=pl.Series(offset_logit.predict(w, xw, offw)),
                                                             b_fit_n=pl.lit(fit_rows.height)))
    return pl.concat(out) if out else pl.DataFrame(schema={"game_id": pl.String, "mode": pl.String,
                                                           "p_B": pl.Float64, "b_fit_n": pl.Int32})


def predictions(feats: pl.DataFrame, games: pl.DataFrame, bundle: dict, include_holdout: bool = False) -> pl.DataFrame:
    d = (feats.join(games.select("game_id", "season", "week", "home_win"), on="game_id")
         .join(game_ends(games), on="game_id", how="left").sort("predict_time"))
    d = d.with_columns(**{k: pl.Series(v) for k, v in predict_F(bundle, d).items()})
    mm = market_models(d, games)
    b = walk_forward_B(d, include_holdout)
    return (d.select("game_id", "mode", "season", "week", "kickoff_utc", "predict_time", "home_win", "p_F", "p_F_lr", "p_F_lgbm")
            .join(mm.drop("kickoff_utc"), on=["game_id", "mode"]).join(b, on=["game_id", "mode"], how="left"))


def train_all() -> list[Path]:
    root = data_root()
    games = pl.read_parquet(root / "clean/games.parquet")
    (root / "models").mkdir(exist_ok=True)
    out = []
    for mode in MODES:
        feats = pl.read_parquet(root / f"features/features_{mode}.parquet")
        bundle = train_F(feats, games)
        with open(root / f"models/F_{mode}.pkl", "wb") as fh:
            pickle.dump(bundle, fh)
        out.append(root / f"models/preds_{mode}.parquet")
        predictions(feats, games, bundle).write_parquet(out[-1])
        print(mode, "F chosen:", bundle["chosen"],
              {n: m["cal_kind"] + f" {min(m['cal_cv_logloss'].values()):.4f}" for n, m in bundle["models"].items()})
    return out
