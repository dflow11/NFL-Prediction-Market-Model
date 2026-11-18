"""Evaluation (`nflpm eval`) and the holdout protocol (`nflpm freeze`, `nflpm eval --holdout`).

Primary evidence = CLV and cross-venue lead-lag. Brier / log loss / accuracy are secondary and always
come with paired-bootstrap CIs and the MDE; a CI containing 0 is "inconclusive", never "no edge"/"edge".
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from nflpm import holdout
from nflpm.features.market import logit
from nflpm.paths import REPO, UTC, config, data_root

N_BOOT = 2000
MODELS = ["0", "A", "A2", "F", "B"]
CAVEATS = [
    "Model 0's lines (nflverse `line_*`) name no book and no capture time; they may include news after predict_time.",
    "2025 Polymarket features are ≥3-hour resolution (fine-grained history had expired before archiving began).",
    "Paper research only; small samples — read every CI before any point estimate.",
]


# ---------- statistics ----------

def slope_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """OLS slope of y on x with a percentile bootstrap 95% CI (rows resampled)."""
    def slope(i):
        xx, yy = x[i], y[i]
        v = np.var(xx)
        return np.cov(xx, yy, bias=True)[0, 1] / v if v > 0 else np.nan
    if len(x) < 3:
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    lo, hi = np.nanpercentile([slope(rng.integers(0, len(x), len(x))) for _ in range(n_boot)], [2.5, 97.5])
    return float(slope(np.arange(len(x)))), float(lo), float(hi)


def mean_ci(v: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    if len(v) == 0:
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def mde(diff: np.ndarray) -> float:
    """Minimum detectable effect at 80% power: 2.8 × sd(per-game diff) / √N."""
    return float(2.8 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan


def verdict(lo: float, hi: float) -> str:
    return "inconclusive" if lo <= 0 <= hi or np.isnan(lo) else ("lower" if hi < 0 else "higher")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------- data ----------

def eval_frame(preds: pl.DataFrame, feats: pl.DataFrame, targets: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    return (preds.join(feats.select("game_id", "mode", "kalshi_mid", "poly_mid", "gap_logit", "poly_resolution_sec",
                                    "poly_source"), on=["game_id", "mode"], how="left")
            .join(targets.select("game_id", "kalshi_close", "poly_close"), on="game_id", how="left")
            .join(games.select("game_id", "line_home_prob"), on="game_id", how="left"))


def scoring_set(d: pl.DataFrame) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
    """Rows where 0, A, A2, F and B all exist and the game has a winner. Returns (rows, drop log)."""
    filters = [("tie or unplayed", pl.col("home_win").is_not_null()),
               ("no Kalshi price (A)", pl.col("p_A").is_not_null()),
               ("no line (Model 0)", pl.col("p_0").is_not_null()),
               ("no F", pl.col("p_F").is_not_null()),
               ("no B (burn-in, no Polymarket, or missing input)", pl.col("p_B").is_not_null())]
    drops = []
    for reason, cond in filters:
        n = d.height
        d = d.filter(cond)
        drops.append((reason, n - d.height))
    return d, drops


# ---------- report sections ----------

def md_table(df: pl.DataFrame, fmt: str = "{:.4f}") -> str:
    def cell(v):
        return fmt.format(v) if isinstance(v, float) and not np.isnan(v) else ("—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.iter_rows())


def coverage_section(d: pl.DataFrame, drops, scored: pl.DataFrame, coverage_csv: Path) -> str:
    by = d.group_by("season").agg(rows=pl.len(), kalshi=pl.col("kalshi_mid").is_not_null().sum(),
                                  poly=pl.col("poly_mid").is_not_null().sum()).sort("season")
    res = d.filter(pl.col("poly_mid").is_not_null()).group_by("season", "poly_source", "poly_resolution_sec").len().sort(
        "season", "poly_resolution_sec")
    s = "## 1. Coverage\n\n" + md_table(by) + "\n\nPolymarket feature resolution:\n\n" + md_table(res)
    s += f"\n\nScoring set N = **{scored.height}**. Drops (in order):\n\n" + md_table(
        pl.DataFrame(drops, schema=["filter", "dropped"], orient="row"))
    if coverage_csv.exists():
        c = pl.read_csv(coverage_csv)
        if c.height:
            s += "\n\nSnapshot coverage ([kickoff−3h, kickoff), 5-min slots):\n\n" + md_table(
                c.group_by("venue").agg(games=pl.len(), mean_share=pl.col("share").mean(), under_80=pl.col("under_80").sum()))
    return s


def clv_section(d: pl.DataFrame, mode: str) -> str:
    rows = []
    closes = [("kalshi_close", "Kalshi close")] + ([("line_home_prob", "line (alt. close)")] if mode == "tuesday" else [])
    for col, label in closes:
        for m in ("F", "B", "A2"):
            x = d.drop_nulls([f"p_{m}", "kalshi_mid", col])
            dd = x.select(logit(pl.col(f"p_{m}")) - logit(pl.col("kalshi_mid"))).to_series().to_numpy()
            dk = x.select(logit(pl.col(col)) - logit(pl.col("kalshi_mid"))).to_series().to_numpy()
            b, blo, bhi = slope_ci(dd, dk)
            mu, mlo, mhi = mean_ci(np.sign(dd) * dk)
            rows.append({"close": label, "model": m, "n": len(dd), "slope": b, "slope_lo": blo, "slope_hi": bhi,
                         "mean_sign_d_x_dk": mu, "mean_lo": mlo, "mean_hi": mhi, "verdict": verdict(mlo, mhi)})
    return ("## 2. Primary — closing-line value\n\nd = logit(model) − logit(kalshi_mid); Δ = logit(close) − logit(kalshi_mid). "
            "Positive slope / mean(sign(d)·Δ) means the close moved toward the model.\n\n" + md_table(pl.DataFrame(rows)))


def lead_lag(d: pl.DataFrame) -> pl.DataFrame:
    """Δk = logit(kalshi_close) − logit(kalshi_mid), Δp likewise for poly, each regressed on gap_logit;
    stratified by poly resolution (≤300s vs ≥1800s). Pre-holdout rows only."""
    x = (d.filter(~holdout.is_holdout())
         .drop_nulls(["kalshi_mid", "poly_mid", "kalshi_close", "poly_close", "gap_logit"])
         .with_columns(dk=logit(pl.col("kalshi_close")) - logit(pl.col("kalshi_mid")),
                       dp=logit(pl.col("poly_close")) - logit(pl.col("poly_mid"))))
    strata = {"all": pl.lit(True), "poly_res<=300s": pl.col("poly_resolution_sec") <= 300,
              "poly_res>=1800s": pl.col("poly_resolution_sec") >= 1800}
    rows = []
    for name, cond in strata.items():
        s = x.filter(cond)
        for t in ("dk", "dp"):
            b, lo, hi = slope_ci(s["gap_logit"].to_numpy(), s[t].to_numpy())
            rows.append({"stratum": name, "target": t, "n": s.height, "slope": b, "ci_lo": lo, "ci_hi": hi})
    return pl.DataFrame(rows)


def secondary_section(s: pl.DataFrame) -> str:
    y = s["home_win"].cast(pl.Float64).to_numpy()
    rows = []
    for m in MODELS:
        p = s[f"p_{m}"].to_numpy()
        rows.append({"model": m, "n": len(y), "log_loss": float(logloss(p, y).mean()) if len(y) else np.nan,
                     "brier": float(brier(p, y).mean()) if len(y) else np.nan,
                     "accuracy": float(((p > 0.5) == (y == 1)).mean()) if len(y) else np.nan})
    diffs = []
    for m in ("B", "A2"):
        diff = brier(s[f"p_{m}"].to_numpy(), y) - brier(s["p_A"].to_numpy(), y)
        mu, lo, hi = mean_ci(diff)
        diffs.append({"comparison": f"Brier({m}) − Brier(A)", "n": len(diff), "mean": mu, "ci_lo": lo, "ci_hi": hi,
                      "mde_80pct": mde(diff), "verdict": verdict(lo, hi)})
    return ("## 4. Secondary — scoring on the scoring set\n\n" + md_table(pl.DataFrame(rows))
            + f"\n\nPaired bootstrap ({N_BOOT} resamples):\n\n" + md_table(pl.DataFrame(diffs)))


def calibration_plot(s: pl.DataFrame, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = s["home_win"].cast(pl.Float64).to_numpy()
    series = {f"model {m}": s[f"p_{m}"].to_numpy() for m in MODELS} | {"venue Polymarket": s["poly_mid"].to_numpy()}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    bins = np.linspace(0, 1, 11)
    for name, p in series.items():
        ok = ~np.isnan(p)
        idx = np.clip(np.digitize(p[ok], bins) - 1, 0, 9)
        xs = [p[ok][idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [y[ok][idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", lw=1, label=name)
    ax.set(xlabel="predicted P(home win)", ylabel="observed home win rate", title="Calibration (10 bins)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"## 5. Calibration\n\n![calibration]({path.name})  (model A = Kalshi venue)"


def favorite_section(d: pl.DataFrame) -> str:
    x = d.drop_nulls(["kalshi_mid", "home_win"]).with_columns(
        fav_price=pl.max_horizontal("kalshi_mid", 1 - pl.col("kalshi_mid")),
        fav_won=pl.when(pl.col("kalshi_mid") >= 0.5).then(pl.col("home_win")).otherwise(~pl.col("home_win")))
    rows = []
    for lo in np.arange(0.5, 1.0, 0.1):
        b = x.filter(pl.col("fav_price").is_between(lo, lo + 0.1, closed="left" if lo < 0.9 else "both"))
        k, n = int(b["fav_won"].sum()), b.height
        wl, wh = wilson(k, n)
        rows.append({"favorite_price": f"{lo:.0%}–{lo + 0.1:.0%}", "n": n, "win_rate": k / n if n else np.nan,
                     "wilson_lo": wl, "wilson_hi": wh})
    return "## 6. Favorite–longshot (Kalshi price at predict_time)\n\n" + md_table(pl.DataFrame(rows))


def disagreement_section(d: pl.DataFrame) -> str:
    edges = config()["gap_buckets_pts"]
    x = d.drop_nulls(["kalshi_mid", "poly_mid", "home_win"]).with_columns(
        gap_pts=(pl.col("kalshi_mid") - pl.col("poly_mid")).abs() * 100)
    y = lambda b: b["home_win"].cast(pl.Float64).to_numpy()  # noqa: E731
    rows = []
    bounds = [(-np.inf, edges[0])] + list(zip(edges, edges[1:])) + [(edges[-1], np.inf)]
    for lo, hi in bounds:
        b = x.filter((pl.col("gap_pts") >= lo) & (pl.col("gap_pts") < hi))
        row = {"gap_pts": f"[{lo}, {hi})", "n": b.height}
        for venue, col in (("kalshi", "kalshi_mid"), ("poly", "poly_mid")):
            mu, l, h = mean_ci(brier(b[col].to_numpy(), y(b)))
            row |= {f"{venue}_brier": mu, f"{venue}_lo": l, f"{venue}_hi": h}
        rows.append(row)
    return "## 7. Disagreement buckets (|Kalshi − Polymarket|, probability points)\n\n" + md_table(pl.DataFrame(rows))


def render(d: pl.DataFrame, mode: str, title: str, plot_path: Path, coverage_csv: Path) -> str:
    scored, drops = scoring_set(d)
    parts = [f"# {title} — {mode}\n\nGenerated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Rows: {d.height}.",
             coverage_section(d, drops, scored, coverage_csv),
             clv_section(d, mode),
             "## 3. Primary — lead-lag\n\ngap_logit = logit(kalshi_mid) − logit(poly_mid); Δk, Δp regressed on it (bootstrap 95% CIs). Kalshi following Polymarket shows as Δk slope → −1; Polymarket following Kalshi as Δp slope → +1.\n\n" + md_table(lead_lag(d)),
             secondary_section(scored),
             calibration_plot(scored, plot_path),
             favorite_section(d),
             disagreement_section(d),
             "## 8. Caveats\n\n" + "\n".join(f"- {c}" for c in CAVEATS)]
    return "\n\n".join(parts) + "\n"


def load(mode: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    root = data_root()
    return (pl.read_parquet(root / f"models/preds_{mode}.parquet"), pl.read_parquet(root / f"features/features_{mode}.parquet"),
            pl.read_parquet(root / "features/targets.parquet"), pl.read_parquet(root / "clean/games.parquet"))


def evaluate(mode: str) -> Path:
    """Default eval: pre-holdout rows only."""
    d = eval_frame(*load(mode)).filter(~holdout.is_holdout())
    out = REPO / f"reports/eval_{mode}.md"
    out.write_text(render(d, mode, "Evaluation (pre-holdout)", REPO / f"reports/eval_{mode}_calibration.png",
                          REPO / "reports/coverage.csv"))
    return out


# ---------- holdout protocol ----------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def fingerprint() -> dict:
    """Current code + config + frozen-model identity."""
    root = data_root()
    dirty = _git("status", "--porcelain", "--", "src", "config.yaml", "pyproject.toml", "uv.lock")
    return {"git_commit": _git("rev-parse", "HEAD"), "code_dirty": bool(dirty),
            "config_sha256": _sha(REPO / "config.yaml"),
            "models": {p.name: _sha(p) for p in sorted((root / "models").glob("F_*.pkl"))}}


def manifest_path() -> Path:
    return data_root() / "models/frozen/manifest.json"


def freeze() -> Path:
    fp = fingerprint()
    if fp["code_dirty"]:
        raise SystemExit("refusing to freeze: uncommitted changes in src/, config.yaml or the lockfile")
    p = manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**fp, "frozen_at": datetime.now(UTC).isoformat()}, indent=2))
    return p


def check_manifest() -> dict:
    p = manifest_path()
    if not p.exists():
        raise SystemExit("holdout eval refused: no models/frozen/manifest.json (run `nflpm freeze`)")
    m, fp = json.loads(p.read_text()), fingerprint()
    bad = [k for k in ("git_commit", "config_sha256", "models") if m[k] != fp[k]] + (["code_dirty"] if fp["code_dirty"] else [])
    if bad:
        raise SystemExit(f"holdout eval refused: manifest does not match current {bad}")
    return m


def evaluate_holdout(mode: str, force: bool = False) -> Path:
    from nflpm.models.train import predictions
    m = check_manifest()
    lock = data_root() / "models/frozen/holdout.lock"  # one JSON line per holdout run
    done = [json.loads(line) for line in lock.read_text().splitlines()] if lock.exists() else []
    rerun = any(r["mode"] == mode for r in done)
    if rerun and not force:
        raise SystemExit(f"holdout already evaluated for {mode} ({lock}); re-run requires --force")
    root = data_root()
    with open(root / f"models/F_{mode}.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    _, feats, targets, games = load(mode)
    preds = predictions(feats, games, bundle, include_holdout=True)  # frozen procedure, walk-forward refits
    d = eval_frame(preds, feats, targets, games).filter(holdout.is_holdout())
    title = "Holdout evaluation" + (" — FORCED RE-RUN" if rerun else "")
    text = render(d, mode, title, REPO / f"reports/holdout_eval_{mode}_calibration.png", REPO / "reports/coverage.csv")
    text += f"\nManifest: commit `{m['git_commit']}`, config `{m['config_sha256'][:12]}`, frozen {m['frozen_at']}.\n"
    out = REPO / f"reports/holdout_eval_{mode}.md"
    out.write_text(text)
    with lock.open("a") as fh:
        fh.write(json.dumps({"mode": mode, "at": datetime.now(UTC).isoformat(), "forced": rerun}) + "\n")
    return out
