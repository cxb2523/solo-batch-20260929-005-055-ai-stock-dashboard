import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow import Config, FingerprintMemo, QuoteCache  # noqa: E402


def make_ohlcv(n: int = 120, *, seed: int = 7, flat: bool = False,
               zero_volume: bool = False, tz: str = "America/New_York"):
    """生成 n 个交易日的合成 OHLCV；flat=True 时价格恒定（触发除零场景）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").date(), periods=n, tz=tz)
    if flat:
        close = np.full(n, 100.0)
    else:
        shocks = rng.normal(0.0005, 0.012, size=n)
        close = 50.0 * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = np.zeros(n) if zero_volume else rng.integers(1e6, 4e7, size=n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": volume, "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=idx)


@pytest.fixture
def fake_fetcher():
    """返回 (fetcher_factory, counters)；每个 ticker 一份确定性行情。"""
    counters = {"calls": 0, "fail_with": None}

    def fetcher(symbol):
        counters["calls"] += 1
        if counters["fail_with"]:
            raise counters["fail_with"]
        seed = sum(ord(c) for c in symbol)
        return make_ohlcv(250, seed=seed), {"longName": symbol, "marketCap": 1e11}

    return fetcher, counters


@pytest.fixture
def memory_cache():
    return QuoteCache(ttl_seconds=3600, path=None)


@pytest.fixture
def memo():
    return FingerprintMemo()


@pytest.fixture
def offline_config():
    return Config(cache_path=None, allow_demo_fallback=False, ai_api_key=None)


@pytest.fixture
def demo_config():
    return Config(cache_path=None, allow_demo_fallback=True, ai_api_key=None)
