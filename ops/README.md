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
| `nflpm-collect.timer` | every 5 min (`*:0/5`), not caught up after downtime | `scripts/collect.py` |
| `nflpm-archive.timer` | daily 03:30 America/Chicago, catches up after downtime | `scripts/pm_archive.py` |

Data goes to `$NFLPM_DATA_DIR`, defaulting to `~/nflpm-data`.

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

Run the archive by hand with `systemctl --user start nflpm-archive.service`, or with
`uv run scripts/pm_archive.py [slug ...]` to archive just some games.
