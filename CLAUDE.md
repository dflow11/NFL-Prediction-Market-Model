NFL prediction-market research (paper only). Build spec: `docs/spec.md`; build phase by phase.

## Rules
- Meet each phase's acceptance criteria before the next. Print the spec's summary block after each phase.
- Nothing beyond the spec: no extra features, sources or abstractions. Ask when ambiguous.
- Verify endpoints and params against the official docs before coding. Read-only public data; if an endpoint wants auth, stop and ask.
- Point-in-time: every feature group carries `src_max_ts < predict_time`. A game's data becomes available at `game_end_ts`, never at kickoff.
- `raw/` is immutable: write through `nflpm.paths.write_raw` (atomic, gzip, byte-exact, raises on overwrite).
- Holdout rows (kickoff ≥ `holdout.start_utc`) never enter fitting, tuning or the default eval.
- Never interpolate over collection gaps; record them.
- Filenames use UTC, colon-free timestamps: `20261117T014400Z` (`nflpm.paths.stamp`).
- HTTP only through `nflpm.http.Http` with an explicit cache policy (`immutable` / `snapshot` / `nocache`).
- Small typed functions, Python 3.11+, `uv`. Tests never touch the network (httpx.MockTransport; fixtures in `tests/fixtures/`).

## Data
`data_root` = env `NFLPM_DATA_DIR`, else `config.yaml: paths.data_root` (default `~/nflpm-data`).
Layout: `raw/{nflverse,kalshi,polymarket,snapshots}/` (immutable), `clean/ features/ models/ logs/ state/ cache/`.
Collection runs on systemd user timers on the home PC (`ops/README.md`); GitHub runs CI tests only.

## Run
- `uv sync`, `uv run pytest -q`, `uv run nflpm --help`

## Agent skills

### Issue tracker

Issues live in GitHub Issues on this repo (`gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
