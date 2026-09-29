import time

import pandas as pd
import pytest

from conftest import make_ohlcv
from dataflow import Config, QuoteCache, quote_key
from dataflow import fetch as fetch_stage
from dataflow.cache import normalize_interval


def test_cache_key_uses_ticker_and_date_interval_not_period():
    # 同一 period 在不同“今天”下必须展开成不同区间键
    now_a = pd.Timestamp("2026-01-01", tz="UTC")
    now_b = pd.Timestamp("2026-06-01", tz="UTC")
    a = normalize_interval("1y", now=now_a)
    b = normalize_interval("1y", now=now_b)
    assert a != b
    # 显式区间稳定，与调用时间无关
    k1 = quote_key("aapl", start="2025-01-01", end="2025-06-01")
    k2 = quote_key("AAPL", start="2025-01-01", end="2025-06-01")
    assert k1 == k2 and k1[0] == "AAPL"


def test_fresh_cache_hit_counts_and_skips_network(fake_fetcher, memory_cache):
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None, allow_demo_fallback=False)
    r1 = fetch_stage.run("AAPL", cfg, period="6mo", cache=memory_cache, fetcher=fetcher)
    r2 = fetch_stage.run("AAPL", cfg, period="6mo", cache=memory_cache, fetcher=fetcher)
    assert counters["calls"] == 1
    assert memory_cache.stats["fresh_hits"] == 1
    assert r1.source == "network" and r2.source == "cache"
    pd.testing.assert_frame_equal(r1.quotes, r2.quotes)


def test_different_tickers_each_fetch_once(fake_fetcher, memory_cache):
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None)
    for symbol in ("AAPL", "MSFT", "GOOGL"):
        fetch_stage.run(symbol, cfg, period="1y", cache=memory_cache, fetcher=fetcher)
    assert counters["calls"] == 3
    # 再来一轮全部命中
    for symbol in ("AAPL", "MSFT", "GOOGL"):
        r = fetch_stage.run(symbol, cfg, period="1y", cache=memory_cache, fetcher=fetcher)
        assert r.source == "cache"
    assert counters["calls"] == 3
    assert memory_cache.stats["fresh_hits"] == 3


def test_expired_entry_refetches_when_network_ok(fake_fetcher, memory_cache):
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None, allow_demo_fallback=False)
    key = quote_key("AAPL", "1y")
    memory_cache.put(key, make_ohlcv(50, seed=1), {}, fetched_at=time.time() - 9999)
    r = fetch_stage.run("AAPL", cfg, period="1y", cache=memory_cache, fetcher=fetcher)
    assert r.source == "network"
    assert counters["calls"] == 1
    assert memory_cache.stats["fresh_hits"] == 0


def test_network_error_falls_back_to_stale_cache(fake_fetcher, memory_cache):
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None, allow_demo_fallback=False)
    key = quote_key("AAPL", "1y")
    old = make_ohlcv(50, seed=1)
    memory_cache.put(key, old, {"longName": "old"}, fetched_at=time.time() - 9999)
    counters["fail_with"] = RuntimeError("rate limited")
    r = fetch_stage.run("AAPL", cfg, period="1y", cache=memory_cache, fetcher=fetcher)
    assert r.source == "stale"
    assert memory_cache.stats["stale_fallbacks"] == 1
    assert any("旧缓存" in w for w in r.warnings)
    pd.testing.assert_frame_equal(r.quotes, old)


def test_network_error_without_cache_and_demo_disabled_returns_empty(fake_fetcher, memory_cache):
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None, allow_demo_fallback=False)
    counters["fail_with"] = ConnectionError("dns down")
    r = fetch_stage.run("ZZZZ", cfg, period="1y", cache=memory_cache, fetcher=fetcher)
    assert r.source == "empty" and r.quotes.empty
    assert r.error and "dns down" in r.error


def test_network_error_without_cache_uses_demo_when_enabled(fake_fetcher, memory_cache):
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None, allow_demo_fallback=True)
    counters["fail_with"] = ConnectionError("offline")
    r = fetch_stage.run("TSLA", cfg, period="1y", cache=memory_cache, fetcher=fetcher)
    assert r.source == "demo"
    assert len(r.quotes) > 200
    assert any("合成行情" in w for w in r.warnings)


def test_empty_response_from_yfinance_is_treated_as_failure(memory_cache):
    cfg = Config(cache_path=None, allow_demo_fallback=False)
    def fetcher(symbol):
        return pd.DataFrame(), {}
    r = fetch_stage.run("BADTICKER", cfg, period="1y",
                        cache=memory_cache, fetcher=fetcher)
    assert r.source == "empty" and r.quotes.empty


def test_blank_ticker_returns_empty_without_network(memory_cache):
    cfg = Config(cache_path=None)
    called = {"n": 0}
    def fetcher(symbol):
        called["n"] += 1
        return make_ohlcv(10), {}
    r = fetch_stage.run("   ", cfg, cache=memory_cache, fetcher=fetcher)
    assert r.source == "empty" and called["n"] == 0


def test_disk_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "q.pkl")
    c1 = QuoteCache(ttl_seconds=3600, path=path)
    key = quote_key("NVDA", start="2025-01-01", end="2025-03-01")
    c1.put(key, make_ohlcv(40, seed=2), {"x": 1})
    c2 = QuoteCache(ttl_seconds=3600, path=path)
    assert c2.stats["disk_loads"] == 1
    hit = c2.get_fresh(key)
    assert hit is not None and len(hit.quotes) == 40


def test_corrupt_disk_cache_does_not_crash(tmp_path):
    path = tmp_path / "broken.pkl"
    path.write_bytes(b"not a pickle")
    cache = QuoteCache(ttl_seconds=60, path=str(path))
    assert cache.stats["disk_errors"] == 1
    assert cache.get_fresh(quote_key("AAPL", "1y")) is None
