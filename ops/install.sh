#!/usr/bin/env bash
# Render ops/systemd/* with absolute paths into ~/.config/systemd/user and enable the timers.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
uv=$(command -v uv)
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
for f in "$repo"/ops/systemd/*; do
  sed -e "s|@REPO@|$repo|g" -e "s|@UV@|$uv|g" "$f" > "$dest/$(basename "$f")"
done
systemctl --user daemon-reload
for t in "$repo"/ops/systemd/*.timer; do
  systemctl --user enable --now "$(basename "$t")"
done
systemctl --user list-timers 'nflpm-*'
