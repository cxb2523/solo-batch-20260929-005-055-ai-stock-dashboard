from datetime import date

import pandas as pd

from dataflow.cache import TimedCache, dataframe_fingerprint
from dataflow.fetch import (
    Fetcher,
    history_cache_key,
    resolve_range,
)
from .helpers import FakeYfClient


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_cache_hit_counts_and_stale_fallback():
    clock = FakeClock()
    cache = TimedCache(ttl=10.0, stale_ttl=100.0, clock=clock)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("boom")
        return f"v{calls}"

    value, source = cache.get_or_compute("k", factory)
    assert value == "v1" and source == "miss" and calls == 1

    clock.now += 5
    value, source = cache.get_or_compute("k", factory)
    assert value == "v1" and source == "hit" and calls == 1

    # 过期后取新失败 -> 陈旧回退
    clock.now += 20
    value, source = cache.get_or_compute("k", factory)
    assert value == "v1" and source == "stale" and calls == 2
    assert (cache.stats.hits, cache.stats.misses, cache.stats.stale) == (1, 1, 1)

    # 超过陈旧窗口后不再回退，异常抛出
    clock.now += 200
    try:
        cache.get_or_compute("k", factory)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected stale window expiry to raise")


def test_force_refresh_bypasses_fresh_cache():
    clock = FakeClock()
    cache = TimedCache(ttl=10.0, clock=clock)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return f"v{calls}"

    assert cache.get_or_compute("k", factory)[1] == "miss"
    assert cache.get_or_compute("k", factory, force_refresh=True)[1] == "bypass"
    assert calls == 2
    assert cache.stats.bypass == 1


def test_history_cache_key_is_ticker_plus_date_range_not_period():
    d1 = date(2026, 9, 1)
    d2 = date(2026, 9, 30)
    s1_1, e1_1 = resolve_range("1y", today=d1)
    s1_2, e1_2 = resolve_range("1y", today=d2)
    key_day1 = history_cache_key("AAPL", s1_1, e1_1)
    key_day2 = history_cache_key("AAPL", s1_2, e1_2)
    assert key_day1 != key_day2  # 同为 1y，日期滚动后键必须不同
    assert history_cache_key("aapl", s1_1, e1_1) == history_cache_key("AAPL", s1_1, e1_1)


def test_fetcher_cache_hit_assertions():
    client = FakeYfClient(rows=80)
    fetcher = Fetcher(client=client)

    r1 = fetcher.fetch("AAPL", "1y")
    r2 = fetcher.fetch("AAPL", "1y")
    assert r1.source == "miss"
    assert r2.source == "hit"
    assert client.history_calls == 1  # 第二次命中缓存，没有再打 yfinance
    stats = fetcher.history_cache.stats
    assert stats.hits == 1 and stats.misses == 1

    r3 = fetcher.fetch("AAPL", "1y", force_refresh=True)
    assert r3.source == "bypass"
    assert client.history_calls == 2

    # 不同 ticker 不共享缓存
    fetcher.fetch("MSFT", "1y")
    assert client.history_calls == 3


def test_fetcher_empty_payload_is_not_cached():
    client = FakeYfClient(empty_tickers=("EMPTY",))
    fetcher = Fetcher(client=client)
    r1 = fetcher.fetch("EMPTY", "1y")
    r2 = fetcher.fetch("EMPTY", "1y")
    assert len(r1.history) == 0 and len(r2.history) == 0
    assert r1.degraded and r2.degraded
    assert client.history_calls == 2  # 空响应不得缓存命中
    assert fetcher.history_cache.stats.hits == 0


def test_fingerprint_stable_for_same_data_changes_when_data_changes():
    frame_a = pd.DataFrame(
        {"Close": [1.0, 2.0], "Volume": [10.0, 20.0]},
        index=pd.date_range("2026-01-01", periods=2, tz="UTC"),
    )
    frame_b = frame_a.copy()
    frame_c = frame_a.copy()
    frame_c.iloc[-1, frame_c.columns.get_loc("Close")] = 2.5
    fp1 = dataframe_fingerprint("x", frame_a)
    fp2 = dataframe_fingerprint("x", frame_b)
    fp3 = dataframe_fingerprint("x", frame_c)
    assert fp1 == fp2
    assert fp1 != fp3
