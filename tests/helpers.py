"""测试共用的假 yfinance 客户端与合成数据工厂。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataflow.fetch import normalize_history


def synthetic_ohlcv(
    rows: int = 260,
    seed: int = 7,
    start: str = "2024-01-01",
    tz: str = "America/New_York",
    flat: bool = False,
) -> pd.DataFrame:
    if rows == 0:
        frame = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        frame.index = pd.DatetimeIndex([], tz=tz)
        return frame
    index = pd.bdate_range(start=start, periods=rows, tz=tz)
    rng = np.random.default_rng(seed)
    if flat:
        close = np.full(rows, 100.0)
        open_ = high = low = close
        volume = np.full(rows, 1000.0)
    else:
        log_returns = rng.normal(0.0004, 0.012, size=rows)
        close = 100 * np.exp(np.cumsum(log_returns))
        open_ = close * (1 + rng.normal(0, 0.003, size=rows))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.008, rows))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.008, rows))
        volume = rng.integers(1_000_000, 50_000_000, size=rows).astype(float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


class FakeYfClient:
    """按 ticker 返回合成数据，可模拟空数据与网络故障。"""

    def __init__(self, rows: int = 260, tz: str = "America/New_York", empty_tickers=()):
        self.rows = rows
        self.tz = tz
        self.empty_tickers = {t.upper() for t in empty_tickers}
        self.history_calls = 0
        self.quote_calls = 0
        self.fail_history_after: int | None = None
        self.fail_quote = False

    def history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        self.history_calls += 1
        if self.fail_history_after is not None and self.history_calls > self.fail_history_after:
            raise ConnectionError("simulated outage")
        ticker = ticker.upper()
        if ticker in self.empty_tickers:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]).set_index(
                pd.DatetimeIndex([], tz=self.tz)
            )
        seed = sum(ord(ch) for ch in ticker)
        return normalize_history(synthetic_ohlcv(self.rows, seed=seed, tz=self.tz))

    def quote(self, ticker: str) -> dict:
        self.quote_calls += 1
        if self.fail_quote:
            raise ConnectionError("simulated quote outage")
        return {
            "symbol": ticker.upper(),
            "regularMarketPrice": 150.0,
            "marketCap": 2_000_000_000_000.0,
            "longName": f"{ticker.upper()} Inc",
            "sector": "Tech",
        }
