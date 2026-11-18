"""Phase 1: raw writer, cache policies, retry, archive chunking. No network: httpx.MockTransport."""
import gzip
from datetime import datetime, timedelta

import httpx
import pytest

from nflpm.http import Http
from nflpm.ingest.polymarket import chunks, grains
from nflpm.paths import UTC, write_raw

URL = "https://api.elections.kalshi.com/trade-api/v2/x"


def counting_http(responses: list[httpx.Response]) -> tuple[Http, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return responses[min(len(seen), len(responses)) - 1]

    return Http(transport=httpx.MockTransport(handler)), seen


def test_write_raw_gzips_exact_bytes_and_refuses_overwrite(tmp_path):
    p = tmp_path / "raw/a.json.gz"
    write_raw(p, b'{"x": 1}')
    assert gzip.decompress(p.read_bytes()) == b'{"x": 1}'
    with pytest.raises(FileExistsError):
        write_raw(p, b"other")
    assert gzip.decompress(p.read_bytes()) == b'{"x": 1}'
    assert [f.name for f in p.parent.iterdir()] == ["a.json.gz"]  # no temp left behind


def test_immutable_hit_skips_network(tmp_path):
    http, seen = counting_http([httpx.Response(200, content=b"fresh")])
    p = tmp_path / "c.json.gz"
    assert http.request("GET", URL, policy="immutable", raw_path=p).content == b"fresh"
    assert http.request("GET", URL, policy="immutable", raw_path=p).content == b"fresh"
    assert len(seen) == 1


@pytest.mark.parametrize("policy", ["snapshot", "nocache"])
def test_snapshot_and_nocache_always_hit_network(tmp_path, policy):
    http, seen = counting_http([httpx.Response(200, content=b"x")])
    write_raw(tmp_path / "c.json.gz", b"cached")
    http.request("GET", URL, policy=policy)
    http.request("GET", URL, policy=policy)
    assert len(seen) == 2


def test_retries_429_then_succeeds():
    http, seen = counting_http([httpx.Response(429, headers={"Retry-After": "0"}),
                                httpx.Response(200, content=b"ok")])
    assert http.request("GET", URL, policy="nocache").content == b"ok"
    assert len(seen) == 2


def test_chunks_clip_to_window_and_skip_incomplete_or_precreation():
    start = datetime(2026, 11, 16, 1, 15, tzinfo=UTC)  # MNF, UTC
    created = datetime(2026, 11, 10, 12, tzinfo=UTC)
    now = datetime(2026, 11, 16, 3, 0, tzinfo=UTC)  # game day chunk still open
    got = list(chunks(start, created, now))
    assert got[0] == (datetime(2026, 11, 10, tzinfo=UTC), datetime(2026, 11, 11, tzinfo=UTC))
    assert got[-1][1] == datetime(2026, 11, 16, tzinfo=UTC)  # [16th 00:00, 07:15) not complete yet
    all_done = list(chunks(start, datetime(2026, 1, 1, tzinfo=UTC), start + timedelta(days=1)))
    assert all_done[0][0] == start - timedelta(days=8)  # first chunk clipped to window start
    assert all_done[-1][1] == start + timedelta(hours=6)


def test_grains_drop_fine_data_well_past_floor():
    now = datetime(2026, 11, 18, tzinfo=UTC)
    assert grains(now - timedelta(days=5), now) == [60, 300, 1800, 10800]
    assert grains(now - timedelta(days=22), now) == [300, 1800, 10800]
    assert grains(now - timedelta(days=200), now) == [10800]
