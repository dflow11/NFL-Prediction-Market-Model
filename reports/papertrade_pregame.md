# Paper trading (Kalshi) — pregame

**paper-only, small sample (ROI CI ≈ ±20% at 100 trades).** Pre-holdout rows only; hold to settlement; taker fees; stake $10, edge buffer 0.01. F is a sanity check.

| model | trades | hit_rate | pnl_usd | roi | roi_lo | roi_hi | max_drawdown_usd | mean_clv | clv_lo | clv_hi | size_unknown | size_short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 29 | 0.4138 | -69.4000 | -0.2390 | -0.5812 | 0.0994 | 94.6300 | -0.0034 | -0.0066 | 0.0000 | 29 | 0 |
| F | 338 | 0.3757 | -339.8800 | -0.0984 | -0.2358 | 0.0506 | 367.6200 | -0.0039 | -0.0051 | -0.0029 | 338 | 0 |
