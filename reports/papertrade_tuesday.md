# Paper trading (Kalshi) — tuesday

**paper-only, small sample (ROI CI ≈ ±20% at 100 trades).** Pre-holdout rows only; hold to settlement; taker fees; stake $10, edge buffer 0.01. F is a sanity check.

| model | trades | hit_rate | pnl_usd | roi | roi_lo | roi_hi | max_drawdown_usd | mean_clv | clv_lo | clv_hi | size_unknown | size_short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 31 | 0.3871 | -108.7900 | -0.3523 | -0.6441 | -0.0316 | 108.7900 | 0.0023 | -0.0172 | 0.0206 | 31 | 0 |
| F | 337 | 0.3917 | -98.9100 | -0.0287 | -0.1677 | 0.1180 | 283.8100 | 0.0026 | -0.0012 | 0.0068 | 337 | 0 |
