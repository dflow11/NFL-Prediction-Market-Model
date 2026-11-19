# NFL Prediction Market ML — Build Spec v2

## Context
Two-person student project. Research + paper trading only: read-only public market data, no order placement, no API keys expected (if an endpoint demands auth, stop and ask).

Core question: does a football model, the Kalshi–Polymarket gap, or a blend of venues carry information the Kalshi price doesn't already have?

The test era is ~430 games (2025 + 2026 to date) — too few to resolve realistic Brier/ROI edges against an efficient market (simulated power 6–37%; ROI 95% CI ≈ ±20% at 100 trades). So **primary evidence = closing-line value (CLV) and cross-venue lead-lag**. Brier, log loss and ROI are secondary and always shown with CIs and the minimum detectable effect (MDE).

Runtime + storage: Diego's home PC (CachyOS Linux; the machine dual-boots Windows). All data lives under `data_root` on that machine. Nothing data-bearing runs on GitHub.

## Rules for Claude Code
- Build phase by phase; meet each phase's acceptance criteria before moving on. **Phase 0 is urgent — do it first, in one session.**
- Don't add features, sources, or abstractions not listed here. Ask if something's ambiguous.
- Verify endpoints/params against official docs before coding. "Verified facts" were checked 2026-11-17; still confirm exact param names.
- Point-in-time is non-negotiable: every feature group carries `src_max_ts`, which must be `< predict_time`. A game's data becomes available at its END (`game_end_ts`), never at kickoff.
- `raw/` is immutable: write temp file → atomic rename; attempting to overwrite raises. Store response bodies byte-exact, gzipped (`.json.gz`).
- Holdout rows (kickoff ≥ `holdout.start_utc`) never enter fitting, tuning, or default eval.
- Never interpolate over collection gaps; record them.
- Filenames use UTC, colon-free timestamps: `20261117T014400Z`.
- Small typed functions, Python 3.11+, `uv`. Tests never touch the network (mock httpx; fixtures in `tests/fixtures/`).
- After each phase, output the summary block at the bottom.

## Verified facts (2026-11-17)
**Kalshi** — base `https://api.elections.kalshi.com/trade-api/v2`
- NFL game series `KXNFLGAME`. One event per game; **two markets per event, one per team**. Market ticker suffix = team code (`KXNFLGAME-26SEP27HOUIND-HOU`, `…-IND`). Event ticker = date + away + home (`25OCT06KCJAC` = KC at JAC).
- Candles: `end_period_ts` = inclusive END of the period; `yes_bid`, `yes_ask`, `price` OHLC, `volume`, `open_interest`. `period_interval` ∈ {1, 60, 1440}. Batch candles (≤100 tickers) and event-level candle endpoints exist.
- Units differ: live candle endpoints return integer cents plus `*_dollars` strings; historical candles return dollar strings. Normalize to float dollars in [0,1].
- Live/historical split: `GET /historical/cutoff`. Markets settled before `market_settled_ts` (and their candles) only via `GET /historical/markets` and `GET /historical/markets/{ticker}/candlesticks`.
- Fees: taker = ceil_to_cent(0.07 × C × P × (1−P)) per order (1 contract @ 50¢ → $0.02; 100 @ 50¢ → $1.75). Maker fees only on series that list them. No settlement fee. A "Get Series Fee Changes" endpoint exists (exchange section).
- Football ties: each team market settles at $0.50. Cancelled / rescheduled >2 weeks: fair-price settlement per rules.

**Polymarket Global** (signal-only: U.S. users can't open positions there)
- Gamma `https://gamma-api.polymarket.com`, public, no auth. `GET /sports` and `GET /sports/market-types` exist.
- History: Data API v2 `GET https://data-api.polymarket.com/v2/prices-history`. Params: `token_id`; exactly one of `start`(+`end`, exclusive, window ≤15 days) / `interval` / `as_of`; `bucket_seconds` (60–86400); `limit` ≤10,000; `cursor`. No `fidelity`. Points: `price` (0–1; docs don't say mid vs last), `timestamp`, `resolution_seconds`. **No bid/ask, no sizes.**
- Retention floors: 1-min ≥7 days, 5-min ≥60 days, 30-min ≥90 days; 3-h and 12-h permanent. An expired grain requested explicitly returns an empty page (no fallback).
- Resolved markets end with a settlement point (payout 0/1/fraction) on the final page — post-game, never a price.
- Old CLOB `/prices-history` returns only ≥12-h data for resolved markets — don't use it.

**nflverse**
- `temp`, `wind` = measured at the stadium (PFR), outdoor/open only. `roof` = realized status (`closed`/`open` = the retractable-roof decision that day). `location` ∈ {Home, Neutral}. Line columns name no book and no capture time.

## Stack
- Python 3.11+, `uv`; `polars`; `duckdb` (ASOF JOIN); Parquet
- `httpx` + `tenacity`; `nflreadpy`; `pandera` (polars backend); `typer`
- `scikit-learn`, `lightgbm`; `scipy` (already a scikit-learn dependency) for Model B — no statsmodels
- `pytest`
- Scheduling: systemd user timers on the home PC. GitHub Actions = CI tests only.

## Storage (home PC)
- `data_root` = env `NFLPM_DATA_DIR`, else `config.yaml: paths.data_root` (default `~/nflpm-data`, on the CachyOS NVMe; expand `~`).
```
{data_root}/raw/{nflverse,kalshi,polymarket,snapshots}/   # immutable
{data_root}/clean/  features/  models/  logs/  state/  cache/
```
- `logs/` (append-only run logs) and `state/` (mutable working state) are never raw data.
- `reports/` lives in the repo; humans commit it by hand.
- Expected gaps: PC off, asleep, or booted into Windows. Collection works in any CachyOS session (KDE or Hyprland).

## Repo Layout
```
nfl-predmarket/
├── pyproject.toml  config.yaml  CLAUDE.md  .gitignore
├── scripts/pm_archive.py  scripts/collect.py        # Phase 0, PEP 723 inline deps
├── ops/systemd/  ops/install.sh  ops/README.md
├── .github/workflows/ci.yml                          # tests only
├── src/nflpm/
│   ├── paths.py            # data_root, write_raw (atomic, no-overwrite, gzip)
│   ├── http.py             # backoff, per-host rate limit, cache policy
│   ├── fees.py  holdout.py
│   ├── ingest/{nflverse,kalshi,polymarket,odds_snapshots}.py   # odds_snapshots = TODO stub only
│   ├── collect/snapshots.py
│   ├── transform/{teams,games,match_games,prices,snapshots}.py
│   ├── features/{market,team_form,context,build}.py
│   ├── validate/schemas.py
│   ├── models/{offset_logit,train,evaluate}.py
│   ├── backtest/paper_trade.py
│   └── cli.py   # nflpm archive|collect|ingest|transform|features|validate|train|freeze|eval|papertrade|notify
├── reports/
└── tests/
```

## config.yaml (create these keys; values are defaults)
```yaml
paths: {data_root: ~/nflpm-data}
seasons: {first: 1999, current: 2026}
modes:
  pregame: {offset_minutes: 60}
  tuesday: {weekday: TUE, time_et: "12:00"}
holdout: {start_utc: "2026-12-02T12:00:00Z"}      # 2026 Week 13 → Super Bowl; humans may change before any eval
rate_limits_rps: {kalshi: 5, polymarket: 5, nflverse: 2}   # lower if docs say so
polymarket: {variant: global, role: signal_only, grain_ladder_sec: [60, 300, 1800, 10800]}
collector: {cadence_min: 5, near_kickoff_hours: 3, far_cadence_min: 60, horizon_days: 8, max_snapshot_age_sec: 600}
game_end_fallback_minutes: 270                    # kickoff + 4h30 when last-play time is unavailable
model_b: {burn_in_weeks: 4, l2_lambda: 10.0}      # frozen before holdout unlock
paper_trade: {venue: kalshi, stake_usd: 10, edge_buffer: 0.01}
fees:
  kalshi:   # effective-dated; fill from the fee schedule + Get Series Fee Changes; flag unknown dates
    - {effective_from: null, taker_rate: 0.07, rounding: ceil_cent_per_order}
gap_buckets_pts: [1, 3]                           # |gap| <1, 1–3, ≥3 probability points; pre-committed
ops: {notify_cmd: "notify-send nflpm '{unit} failed'", backup_dir: null}
```

---

## Phase 0 — URGENT: archive expiring data + start collecting
Why: Polymarket's fine-grained history expires and bid/ask is never stored historically. Week 10's 1-min data hits its floor Nov 19–23; Week 1's 30-min data around Dec 9.

Two standalone scripts (PEP 723 inline deps: `httpx` only; run via `uv run scripts/<name>.py`). They write the canonical raw layout, so later phases reuse the files and never re-fetch.

1. `scripts/pm_archive.py`
   - Discover NFL game-winner (moneyline) markets via Gamma (`/sports` → NFL, `/sports/market-types` → moneyline; verify field names, e.g. outcomes ↔ `clobTokenIds`, game start time). Save Gamma pages to `raw/polymarket/gamma/{fetched_at}/page_{n}.json.gz`.
   - Scope on first run: every 2026-season game. Later runs: any game whose window has started.
   - Window per game = [game_start − 8d, game_start + 6h], split into UTC-day chunks. Skip chunks before market creation.
   - For every **completed** chunk with no file on disk, try grains finest → coarsest with explicit `bucket_seconds` (follow `next_cursor`). Save the first non-empty response to `raw/polymarket/prices_history/{token_id}/{bucket}s/{chunk_start}.json.gz` and stop.
   - Skip a grain when the chunk is more than 14 days past its floor (60s → 7d, 300s → 60d, 1800s → 90d).
   - Log empty responses to `logs/pm_archive/{date}.jsonl`. A chunk where every grain came back empty is logged and never retried.
   - Print a resolution report: game × finest grain obtained.
2. `scripts/collect.py` (one invocation = one snapshot pass)
   - Kalshi, every run: `GET /markets?series_ticker=KXNFLGAME&status=open`, limited to games within `horizon_days` (use close-time filters if the API has them; verify params; paginate).
   - Polymarket: open NFL moneyline tokens with game start within `horizon_days`. Token list cached in `state/pm_open_tokens.json`, refreshed hourly. Fetch CLOB order books (batch endpoint if available; verify) every run for tokens within `near_kickoff_hours` of game start; others only on runs at minute 0.
   - Save to `raw/snapshots/{venue}/{YYYYMMDD}/{HHMMSS}Z.json.gz`, wrapping the body with `fetched_at_utc` and the HTTP `Date` header.
   - The two venues fail independently. Append a run record (ok/fail, counts, latency) to `logs/collector/{YYYYMMDD}.jsonl`.
3. `ops/systemd/` + `ops/install.sh`
   - `install.sh` renders absolute paths (repo dir, `command -v uv`), copies units to `~/.config/systemd/user/`, and enables them.
   - `nflpm-collect.timer`: `OnCalendar=*:0/5`, `AccuracySec=10s`, `Persistent=false`.
   - `nflpm-archive.timer`: `OnCalendar=*-*-* 03:30:00 America/Chicago`, `Persistent=true`.
   - `ops/README.md` covers: `loginctl enable-linger $USER`; disable idle suspend (KDE Power Management + hypridle); `systemctl --user list-timers`; `journalctl --user -u 'nflpm-*'`.

**Accept:** archive ran for every 2026 game to date, with the resolution report; both timers active; ≥3 consecutive snapshot files per venue; re-running the archive makes zero history requests for chunks already on disk.

## Phase 1 — Scaffold, paths, HTTP client, CI
- `pyproject.toml` (uv), `config.yaml`, typer CLI skeleton, `.gitignore` (`data/`, `.env`).
- `CLAUDE.md` (≤40 lines): the Rules section, `data_root`, how to run tests.
- `paths.py`: resolve `data_root`; `write_raw()` (atomic, gzip, raises on overwrite).
- `http.py`: one shared httpx client.
  - tenacity backoff on 429/5xx: 1s → 2s → 4s … max 60s, 6 tries; honor `Retry-After`.
  - Per-host requests/sec cap.
  - **Cache policy per call:**
    - `immutable` — settled markets / completed windows only. Check raw first; a hit makes no network call.
    - `snapshot` — always fetch; write a new timestamped raw file.
    - `nocache` — listings, pagination, open markets. Always fetch; never served from cache.
- Port Phase 0 logic behind `nflpm archive` and `nflpm collect` (same raw layout; Phase 0 files stay valid). Point the timers at `uv run nflpm …`.
- `.github/workflows/ci.yml`: `uv sync` + `pytest` on push/PR. No data, no secrets, no network.

**Accept:** `nflpm --help` works. Tests pass: immutable hit skips the network; `snapshot`/`nocache` always hit the network; `write_raw` refuses overwrite. CI green.

## Phase 2 — Schedules, teams, clean games (no pbp yet)
- `nflreadpy.load_schedules(1999–current)`; point nflreadpy's cache at `{data_root}/cache/` (verify option name).
  - Completed seasons → `raw/nflverse/schedules_{season}.parquet` (immutable).
  - Current season → `raw/nflverse/schedules_{season}/{fetched_at}.parquet`; transforms use the latest file.
- `teams.py`: canonical franchise table covering every code/name seen in nflverse, Kalshi (market-ticker suffixes) and Polymarket (outcome names, slugs).
  - Relocations OAK→LV, SD→LAC, STL→LA; full and short names.
  - Each team's home time zone.
- `clean/games.parquet`: `game_id, season, week, game_type, kickoff_utc, home, away, neutral_site, home_score, away_score, home_win (null if tie/unplayed), line_home_prob, line_spread, line_total`.
  - `close_*` → `line_*`: capture time is unverified.
  - Kickoff = `gameday` + `gametime` in America/New_York → UTC.
  - `line_home_prob` = no-vig moneyline probability (multiplicative normalization).

**Accept:** every team code resolves; ties → null. Season-aware counts:
- Regular season: 248 (1999–2001), 256 (2002–2020), 272 (2021+). 2022 = 271 played (BUF–CIN Week 17 cancelled: accept the row present with nulls, or absent).
- Playoffs: 11 (≤2019), 13 (2020+).

## Phase 3 — Kalshi: metadata → match → candles
- Call `GET /historical/cutoff` once per run; route settled-before-cutoff markets to `/historical/*`.
- Confirm `KXNFLGAME` via `/series`. List events with nested markets (paginate, `nocache`). Store each event's rules text.
- `match_games.py` (shared with Phase 4):
  - Key = unordered team pair + ET kickoff date ± 1 day. Orientation always comes from nflverse home/away, never from venue order (covers neutral sites).
  - Kalshi team codes come from market-ticker suffixes; the date comes from the event ticker.
  - `match_confidence`: 1.0 for exact date, 0.9 for ±1 day.
  - Anything else, or multiple candidates → `reports/unmatched_kalshi.csv`. Never drop silently.
- Candles per matched market (prefer batch/event endpoints; respect max-candle limits by splitting windows):
  - 1-min over [pt − 60m, pt] for every predict_time in every mode, plus [kickoff − 3h, kickoff].
  - 60-min over [kickoff − 8d, kickoff].
  - Raw stored per (market, interval, window). Settled → `immutable`.
- Pull series fee changes for `KXNFLGAME` → raw; fill `fees.kalshi` rows.

**Accept:** ≥98% of 2025 + 2026-to-date events matched; each matched event maps exactly 2 team markets to home/away; unmatched report written; a re-run makes zero candle requests for settled markets.

## Phase 4 — Polymarket: metadata → match → history
- `nflpm archive` covers both seasons. 2025 will mostly yield 3-h data — expected; report it.
- Match via `match_games.py`, mapping outcome names through `teams.py`. Unmatched → `reports/unmatched_polymarket.csv`.
- Store each market's rules text (`description`).

**Accept:** Polymarket matched-game count vs Kalshi logged; resolution report by season; zero re-fetches of chunks on disk.

## Phase 5 — Snapshot collector, hardened
- `nflpm collect`: within `near_kickoff_hours`, also fetch each Kalshi market's orderbook (top levels) for sizes.
- `transform/snapshots.py` → `clean/snapshots.parquet`: `venue, game_id, market_or_token, side, ts_utc, bid, ask, bid_size, ask_size`.
  - `ts_utc` = HTTP `Date` header (fallback `fetched_at_utc`).
- Coverage → `reports/coverage.csv`: per game and venue, share of 5-min slots in [kickoff − 3h, kickoff] with a snapshot. Flag games under 80%.

**Accept:** coverage rows for every game since collection began; a fixture with a gap shows missing slots (no interpolation).

## Phase 6 — Unified price tables
- `clean/prices_long.parquet`: `venue, game_id, market_id, side(home|away), ts_utc, ts_kind(candle_end|history_point|snapshot), resolution_sec, bid, ask, mid, last, quote_source(bbo|history|last_trade)`.
  - Unique on (venue, market_id, ts_kind, resolution_sec, ts_utc).
- `clean/prices_home.parquet`, one row per (venue, game_id, ts_kind, resolution_sec, ts_utc), with `p_home_mid, p_home_bid, p_home_ask, quote_source`:
  - Kalshi: `p_home_mid` = mean(home_yes_mid, 1 − away_yes_mid). If only one side exists, use it and flag. `p_home_bid` = max(home_yes_bid, 1 − away_yes_ask); `p_home_ask` = min(home_yes_ask, 1 − away_yes_bid).
  - Polymarket history: `p_home_mid` = mean(home_token_price, 1 − away_token_price) at the same timestamp and grain; bid/ask null.
  - Snapshots: BBO mids for both venues.
  - Mid from bid/ask is preferred; last trade only when bid/ask is missing (flagged).
  - Null OHLC ≠ 0: treat as missing.
- Keep Polymarket settlement points and any point with ts ≥ game end out of price tables. Venue results (Kalshi results, Polymarket settlement points) go to `clean/settlements.parquet`.

**Accept:** tests for the Kalshi two-market collapse, Polymarket token complement, settlement-point exclusion, and a neutral-site game listed in reverse order oriented to the nflverse home team.

## Phase 7 — Validation (pandera)
Schemas for games, prices_long, prices_home, snapshots, features. Fail loudly on:
- prices outside [0,1]; `bid > ask`
- duplicate keys (as defined above); duplicate (game_id, mode) in features
- a game_id matched to >1 Kalshi event or >1 Polymarket moneyline market; a Kalshi event without exactly 2 team markets
- leakage:
  - any `src_max_ts ≥ predict_time`
  - any `line_*` column in a model feature matrix
  - `temp`/`wind`/realized roof in tuesday-mode features
- any holdout row in a training matrix
- any market feature missing its `*_staleness_sec`

**Accept:** fixtures for leaky timestamps, holdout leakage, and Tuesday-mode weather each fail validation.

## Phase 8 — Market features, market-only models, market evaluation
This phase answers the venue question before any football modeling.
- **predict_time:**
  - `pregame` = kickoff − 60m.
  - `tuesday` = latest Tuesday 12:00 America/New_York strictly before kickoff. Assert it's after both teams' previous `game_end_ts`; otherwise drop the row and log it.
- `features/market.py` via DuckDB ASOF JOIN on `ts_utc ≤ predict_time` (Kalshi `ts_utc` = candle end):
  - `kalshi_mid, kalshi_bid, kalshi_ask, kalshi_staleness_sec`
  - `poly_mid, poly_staleness_sec, poly_resolution_sec, poly_source` — prefer a snapshot within `max_snapshot_age_sec`, else history.
  - `gap_logit = logit(kalshi_mid) − logit(poly_mid)`, with probabilities clipped to [0.01, 0.99].
  - Eval targets only, never features: `kalshi_close`, `poly_close` = last value with ts ≤ kickoff.
    *Deviation (approved 2026-11-18):* `poly_close` uses ts < kickoff; Polymarket's point stamped at kickoff spikes as the book clears.
- **Models:**
  - A = `kalshi_mid`.
  - A2 = inv_logit(0.5·logit(kalshi_mid) + 0.5·logit(poly_mid)); falls back to A when poly is missing (flagged).
  - Model 0 = `line_home_prob` (baseline only).
- **Lead-lag** (pre-holdout rows): regress Δk = logit(kalshi_close) − logit(kalshi_mid) and Δp = logit(poly_close) − logit(poly_mid) on `gap_logit`. Report slopes with bootstrap CIs, stratified by poly resolution (≤300s vs ≥1800s).

**Accept:** ASOF fixture test (a candle ending 1s after predict_time is never used); one feature row per (game_id, mode); A2 fallback flagged.

## Phase 9 — Football data + features
- Ingest `load_pbp` and `load_injuries` (2009+) with Phase 2's raw rules.
- Aggregate pbp once → `clean/team_games.parquet`, one row per team-game: off/def EPA/play, success rate, points, turnovers, fumbles recovered/lost, QB dropbacks by player, `game_end_ts`. Nothing downstream reloads full pbp.
- `game_end_ts` = wall-clock time of the last play if derivable (verify the pbp time-of-day field and its timezone); else kickoff + `game_end_fallback_minutes`.
- `team_form.py`: only team-games with `game_end_ts < predict_time` — this applies to league-wide stats too.
  - Rolling last 4, last 8 and season-to-date off/def EPA and success rate.
  - Point differential vs EPA differential.
  - Luck: fumble recovery rate, one-score record, turnover margin.
  - Early-season shrinkage: blend the prior season toward the league mean. 1999 only supplies priors for 2000.
- `context.py`:
  - Rest days, short-week flag, bye flag; West→East early-kickoff flag (from team time zones).
  - `neutral_site` (home-field and travel features = 0 when neutral).
  - Roof class (dome/outdoors/retractable).
  - `temp`, `wind` and the realized retractable-roof status — **pregame mode only**, null in tuesday mode.
  - QB change flag = the previous game's starter (most dropbacks) listed Out/Doubtful in this game week's injury rows with `date_modified < predict_time`. Null if no qualifying rows (always null before 2009). LR uses 0 + a missing indicator; LightGBM uses NaN.
- `build.py` → `features/features_{mode}.parquet`, one row per (game_id, mode); `src_max_ts` per feature group.

**Accept:** test that pregame features for a 4:25 ET game exclude that day's 1:00 games, including in league-wide stats; test that a tuesday-mode MNF game excludes the same week's Sunday games; test that tuesday mode has no weather.

## Phase 10 — Models F and B
- Label `home_win`; drop ties from fitting and scoring. Paper-trade P&L still settles ties at $0.50.
- **F:** logistic regression, then LightGBM, on team_form + context. Train 2000–2023. Calibrate on 2024 (Platt or isotonic, chosen on 2024 only). Frozen — never refit on 2025+.
- **B** (`models/offset_logit.py`, scipy L-BFGS):
  - logit p = logit(kalshi_mid) + w·x, with x = [1, F_logit − logit(kalshi_mid), gap_logit, log1p(poly_staleness_sec)].
  - L2 penalty on all w (`l2_lambda`), so with no evidence B = A.
  - Walk-forward: refit weekly on rows with `game_end_ts <` that week's first predict_time. The first `burn_in_weeks` of 2025 get no B predictions.
- **Scoring set** = rows where 0, A, A2, F and B all exist (post-burn-in, pre-holdout). Report N and drop reasons per filter.

**Accept:** test that F's fit contains no 2025+ rows; B predicts every post-burn-in, pre-holdout row that has `kalshi_mid`, `poly_mid` and F; test that B with a very large λ reproduces A.

## Phase 11 — Evaluation + holdout protocol
`evaluate.py` → `reports/eval_{mode}.md` + plots, pre-holdout rows only:
1. **Coverage:** N by season / venue / poly resolution; drop reasons; snapshot coverage summary.
2. **Primary — CLV:** for F, B and A2, deviation d = logit(model) − logit(kalshi_mid), target Δk = logit(kalshi_close) − logit(kalshi_mid). Report the slope of Δk on d and mean(sign(d)·Δk), with bootstrap CIs. Tuesday mode also uses `line_home_prob` as an alternate close.
3. **Primary — lead-lag:** Phase 8 slopes, final numbers.
4. **Secondary:**
   - Log loss, Brier and accuracy for 0, A, A2, F, B on the scoring set.
   - Paired bootstrap (≥2,000 resamples) CIs on Brier(B) − Brier(A) and Brier(A2) − Brier(A).
   - Next to each CI: MDE at 80% power = 2.8 × sd(per-game diff) / √N.
   - A CI containing 0 is reported as "inconclusive" — never "no edge" or "edge".
5. Calibration plots (10 bins) per model and per venue.
6. **Favorite-longshot:** one row per game, keyed on the favorite's price (80–90¢ favorite ↔ 10–20¢ underdog). Win rate with Wilson CI and N.
7. **Disagreement:** `gap_buckets_pts` buckets. N and per-venue Brier with CIs per bucket.
8. **Required caveats:** Model 0's lines have unverified timing and may include post-predict_time news; 2025 Polymarket features are ≥3-h resolution.

**Holdout protocol:**
- `nflpm freeze` writes `models/frozen/manifest.json` (git commit, config hash, model hashes).
- `nflpm eval --holdout` requires a manifest matching the current code and config.
  - It evaluates holdout rows with the frozen procedure (walk-forward refits allowed; code and config frozen).
  - It writes `reports/holdout_eval_{mode}.md` and `models/frozen/holdout.lock`.
  - Re-runs refuse without `--force`; forced runs are stamped in the report.

**Accept:** every section renders on fixtures; holdout eval refuses without a manifest (test).

## Phase 12 — Paper trading (Kalshi only)
`paper_trade.py`: hold to settlement; one decision per game per mode at predict_time, using the 1-min candle ending ≤ predict_time. Run for B and for F (F is a sanity check).
- **Cost of the cheapest equivalent contract:**
  - `cost_home` = min(home_yes_ask, 1 − away_yes_bid)
  - `cost_away` = min(away_yes_ask, 1 − home_yes_bid)
- `contracts` = floor(stake_usd / cost); `fee` = `fees.kalshi_fee(date, cost, contracts)` (taker, ceil to the cent per order).
- `edge_side` = p_side − cost_side − fee / contracts. Trade the side with the larger edge if it exceeds `edge_buffer`.
- **Settlement** from Kalshi market results: ties $0.50 per side; cancellations per rules → excluded and logged. Cross-check against nflverse scores; log mismatches.
- **Size:** if a snapshot within `max_snapshot_age_sec` exists, record top-of-book size and flag trades where size < contracts; otherwise mark `size_unknown`.
- **Report:** trades, hit rate, ROI, max drawdown, bootstrap CI on ROI, mean trade CLV (signed Kalshi close − entry) with CI. Label: "paper-only, small sample (ROI CI ≈ ±20% at 100 trades)".

**Accept:** tests that NO-side cost uses 1 − yes_bid; fee rounding (1 @ 50¢ = $0.02; 100 @ 50¢ = $1.75); ties settle at $0.50.

## Phase 13 — Ops on the home PC
- `nflpm-weekly.timer`: `OnCalendar=Tue *-*-* 12:45:00 America/New_York`, `Persistent=true`. Runs `ingest → transform → features → validate → eval` (eval excludes holdout). Validation errors exit non-zero.
- Every service gets `OnFailure=nflpm-notify@%n.service` → `nflpm notify --unit <name>`, which logs to the journal and runs `ops.notify_cmd` (best-effort).
- `ops.backup_dir` (optional): a daily rsync of `raw/` to another disk — snapshots and fine-grained Polymarket history are irreplaceable. If unset, the weekly run prints a warning.
- Update `ops/README.md`; CI stays tests-only.

**Accept:** `systemctl --user list-timers` shows collect, archive and weekly timers; a forced validation failure triggers the notify command.

---

## Out of scope (do not build)
- Live trading, order placement, wallet/API-key trading auth
- Trading on any Polymarket venue; Polymarket US ingest
- On-chain Polymarket trade reconstruction (would give fine-grained 2025 prices — needs human approval)
- In-game/live price modeling; social sentiment; player props
- Cloud storage; GitHub-hosted data or pipeline runs
- Paid odds feeds (stub `ingest/odds_snapshots.py` with a TODO only)

## Open questions (defaults in place — don't block on these)
1. Holdout start: default 2026-12-02 (Week 13 → Super Bowl, ~105 games). Alternative: the whole 2027 season.
2. Partner access to data: default Tailscale + SSH into the home PC.
3. `ops.backup_dir`: unset.
4. On-chain reconstruction for 2025 Polymarket prices: default no.

## Summary block (output after every phase)
```
PHASE: <n> — <name>
STATUS: done | blocked
FILES CHANGED: <list>
DECISIONS MADE: <anything not specified in this doc>
API DEVIATIONS: <endpoints/params that differed from this spec>
DATA STATS: <row counts, match rates, date ranges, resolution/coverage>
TESTS: <passed/failed>
OPEN QUESTIONS: <for the humans>
```
