"""One shared httpx client: per-host rate cap, backoff on 429/5xx, cache policy per call."""
from __future__ import annotations

import gzip
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

from nflpm.paths import config, write_raw

# immutable: settled markets / completed windows only; a raw file on disk is served with no network call.
# snapshot:  always fetched; the caller writes a new timestamped raw file.
# nocache:   listings, pagination, open markets; always fetched, never served from disk.
Policy = Literal["immutable", "snapshot", "nocache"]

_backoff = wait_exponential(multiplier=1, max=60)  # 1s, 2s, 4s ... 60s


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code == 429 or e.response.status_code >= 500
    return isinstance(e, httpx.TransportError)


def _wait(state: RetryCallState) -> float:
    e = state.outcome.exception() if state.outcome else None
    if isinstance(e, httpx.HTTPStatusError):
        try:
            return min(60.0, float(e.response.headers["Retry-After"]))
        except (KeyError, ValueError):
            pass
    return _backoff(state)


class Http:
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        # Polymarket's CDN 403s some default library User-Agents.
        self.client = httpx.Client(timeout=30, transport=transport,
                                   headers={"User-Agent": "nflpm-research/0.1"})
        self.rps: dict[str, float] = config()["rate_limits_rps"]
        self._last: dict[str, float] = {}

    def _throttle(self, host: str) -> None:
        rate = next((r for venue, r in self.rps.items() if venue in host), 1.0)  # unknown host: 1/s
        wait = self._last.get(host, 0.0) + 1 / rate - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    @retry(stop=stop_after_attempt(6), wait=_wait, retry=retry_if_exception(_retryable), reraise=True)
    def _send(self, method: str, url: str, **kw) -> httpx.Response:
        self._throttle(urlsplit(url).hostname or "")
        return self.client.request(method, url, **kw).raise_for_status()

    def request(self, method: str, url: str, *, policy: Policy, raw_path: Path | None = None,
                **kw) -> httpx.Response:
        """Fetch per `policy`; a fetched body is stored byte-exact at `raw_path` when given."""
        if policy == "immutable":
            if raw_path is None:
                raise ValueError("immutable calls need raw_path")
            if raw_path.exists():
                return httpx.Response(200, content=gzip.decompress(raw_path.read_bytes()),
                                      request=httpx.Request(method, url, params=kw.get("params")))
        r = self._send(method, url, **kw)
        if raw_path is not None:
            write_raw(raw_path, r.content)
        return r
