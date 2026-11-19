# Ops (home PC, CachyOS)

Collection only runs while the PC is up in CachyOS (either KDE or Hyprland). Any time it's off,
asleep, or booted into Windows becomes a recorded gap. Those gaps are never interpolated.

## Install

```sh
ops/install.sh                    # renders units into ~/.config/systemd/user, enables timers
loginctl enable-linger "$USER"    # timers keep running when you're logged out
```

| Timer | Schedule | Runs |
|---|---|---|
| `nflpm-collect.timer` | every 5 min (`*:0/5`), not caught up after downtime | `nflpm collect` |
| `nflpm-archive.timer` | daily 03:30 America/Chicago, catches up after downtime | `nflpm archive` |
| `nflpm-weekly.timer` | Tue 12:45 America/New_York, catches up | `backup → ingest → transform → features → validate → train → eval` (stops at the first failure; eval excludes holdout) |
| `nflpm-backup.timer` | daily 04:15 America/Chicago, catches up | `nflpm backup` (rsync `raw/` to `ops.backup_dir`) |

Every service has `OnFailure=nflpm-notify@%n.service`, which runs `nflpm notify --unit <name>`:
it logs to the journal and runs `ops.notify_cmd` from `config.yaml` (best-effort; default `notify-send`).

Data goes to `$NFLPM_DATA_DIR`, defaulting to `~/nflpm-data`.

## Back up raw/

Snapshots and fine-grained Polymarket history can never be re-fetched. `nflpm backup` (daily and at
the start of the weekly run) rsyncs `raw/` to `ops.backup_dir` (`~/nflpm-backup`; `~` is expanded).
That directory is on the same disk as the data, so it guards against deletion, not disk failure:
point it at another disk when one is available. Unset, `nflpm backup` only prints a warning.

## Keep the PC awake

Timers don't wake a suspended machine. Turn off idle suspend in both desktops:

- **KDE**: System Settings → Power Management → On AC Power → untick *Suspend session*
  (or set *When inactive* to *Do nothing*). Screen dimming and turning the screen off are fine.
- **Hyprland**: in `~/.config/hypr/hypridle.conf`, remove or comment out any `listener`
  whose `on-timeout` runs `systemctl suspend`. Locking and DPMS-off are fine. Then
  run `pkill hypridle; hypridle &` (or log out and back in).

## Check on it

```sh
systemctl --user list-timers 'nflpm-*'
journalctl --user -u 'nflpm-*' -n 50
journalctl --user -u nflpm-collect -f
tail ~/nflpm-data/logs/collector/$(date -u +%Y%m%d).jsonl   # one line per venue per run
ls ~/nflpm-data/raw/snapshots/*/$(date -u +%Y%m%d) | tail
```

Test the failure path: `systemctl --user start nflpm-notify@test.service` should pop a notification.

Run the archive by hand with `systemctl --user start nflpm-archive.service`, or with
`uv run nflpm archive [slug ...]` to archive just some games.
