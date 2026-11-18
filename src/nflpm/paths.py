"""Config, data_root, timestamps, and the immutable raw writer."""
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
UTC = timezone.utc


@cache
def config() -> dict:
    return yaml.safe_load((REPO / "config.yaml").read_text())


def data_root() -> Path:
    """env NFLPM_DATA_DIR, else config.yaml paths.data_root; `~` expanded."""
    return Path(os.environ.get("NFLPM_DATA_DIR") or config()["paths"]["data_root"]).expanduser()


def stamp(t: datetime) -> str:
    """UTC, colon-free filename timestamp: 20261117T014400Z."""
    return t.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def write_raw(path: Path, body: bytes) -> None:
    """Write `body` byte-exact (gzipped when `path` ends in .gz); atomic, raises FileExistsError
    instead of overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0) if path.suffix == ".gz" else body)
    try:
        os.link(tmp, path)  # atomic and refuses an existing target, unlike rename
    finally:
        tmp.unlink()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
