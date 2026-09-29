"""行情缓存与计算结果 memo。

取舍 1 —— 缓存键粒度：
    yfinance 的 ``period="1y"`` 是相对滑动窗口，同一字符串在不同日期
    指向不同区间。因此本项目**不**以 period 做键，而是先把 period 或
    start/end 归一化成 UTC 日期区间，键 = (ticker, start_date, end_date)。

取舍 2 —— 命中缓存与取最新行情：
    * ``quote_ttl_seconds``（默认 60s，交易时段刷新）内命中 -> 直接复用，
      不发网络请求（stats.fresh_hits）；
    * 过期 -> 联网取最新；联网失败 -> 回退同键旧缓存
      （stats.stale_fallbacks，trace 中标注 stale），保证整页不崩。
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

PERIOD_DAYS = {
    "5d": 7,
    "1mo": 30,
    "3mo": 91,
    "6mo": 182,
    "1y": 365,
    "2y": 730,
    "5y": 1826,
    "10y": 3652,
    "max": 3652,
}


def normalize_interval(
    period: str | None = "1y",
    start: str | None = None,
    end: str | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[str, str]:
    """把 period 或 start/end 统一成 (UTC 开始日期, UTC 结束日期) 字符串。"""
    now = now or pd.Timestamp.now(tz="UTC")
    if start is not None:
        start_ts = pd.Timestamp(start)
    else:
        days = PERIOD_DAYS.get(period or "1y", 365)
        start_ts = now - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(end) if end is not None else now
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    return start_ts.date().isoformat(), end_ts.date().isoformat()


def quote_key(
    ticker: str,
    period: str | None = "1y",
    start: str | None = None,
    end: str | None = None,
) -> tuple[str, str, str]:
    start_date, end_date = normalize_interval(period, start, end)
    return ticker.strip().upper(), start_date, end_date


# 旧名，部分调用方按 interval 直接构造
interval_key = quote_key


@dataclass
class CacheEntry:
    ticker: str
    start: str
    end: str
    quotes: pd.DataFrame
    info: dict[str, Any]
    fetched_at: float
    source: str = "network"  # network | cache | stale | demo
    warnings: list[str] = field(default_factory=list)


class QuoteCache:
    """进程内 + 可选磁盘 pickle 的行情缓存，线程安全。"""

    def __init__(self, ttl_seconds: float = 60.0, path: str | None = None):
        self.ttl_seconds = ttl_seconds
        self.path = path
        self._entries: dict[tuple[str, str, str], CacheEntry] = {}
        self._lock = threading.RLock()
        self.stats = {
            "fresh_hits": 0,
            "stale_fallbacks": 0,
            "stores": 0,
            "disk_loads": 0,
            "disk_errors": 0,
        }
        if path:
            self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as fh:
                entries = pickle.load(fh)
            if isinstance(entries, dict):
                self._entries.update(entries)
                self.stats["disk_loads"] += 1
        except Exception:
            # 磁盘缓存损坏不应影响启动
            self.stats["disk_errors"] += 1

    def _persist(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(self._entries, fh)
            os.replace(tmp, self.path)
        except Exception:
            self.stats["disk_errors"] += 1

    def get_fresh(
        self, key: tuple[str, str, str], now: float | None = None
    ) -> CacheEntry | None:
        now = now if now is not None else time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if now - entry.fetched_at <= self.ttl_seconds:
                self.stats["fresh_hits"] += 1
                return entry
            return None

    def get_any(self, key: tuple[str, str, str]) -> CacheEntry | None:
        """stale-if-error：忽略 TTL 取同键最后一份成功结果。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self.stats["stale_fallbacks"] += 1
            return entry

    def put(
        self,
        key: tuple[str, str, str],
        quotes: pd.DataFrame,
        info: dict[str, Any] | None,
        *,
        source: str = "network",
        warnings: list[str] | None = None,
        fetched_at: float | None = None,
    ) -> CacheEntry:
        ticker, start, end = key
        entry = CacheEntry(
            ticker=ticker,
            start=start,
            end=end,
            quotes=quotes,
            info=info or {},
            fetched_at=fetched_at if fetched_at is not None else time.time(),
            source=source,
            warnings=list(warnings or []),
        )
        with self._lock:
            self._entries[key] = entry
            self.stats["stores"] += 1
            self._persist()
        return entry


def content_fingerprint(df: pd.DataFrame) -> str:
    """行情内容指纹：同一份数据刷新页面时指标/预测/AI 可安全复用。"""
    if df is None or len(df) == 0:
        return "empty:" + str(0 if df is None else df.shape[1])
    digest = hashlib.sha256()
    digest.update(repr(list(df.columns)).encode())
    digest.update(repr(df.shape).encode())
    digest.update(str(df.index.min()).encode())
    digest.update(str(df.index.max()).encode())
    close_cols = [c for c in df.columns if c in ("Close", "RSI", "MACD")]
    sample = df[close_cols].tail(5).round(6).to_csv().encode()
    digest.update(sample)
    return digest.hexdigest()[:16]


class FingerprintMemo:
    """按 (阶段, 业务键, 输入指纹) 复用计算结果。

    页面刷新时：只要底层行情指纹不变，指标、ML 训练、AI 文本直接复用，
    不重算；行情变了（指纹不同）才重算。渲染不经过 memo，每次重算。
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], tuple[str, Any]] = {}
        self.stats = {"hits": 0, "misses": 0}

    def get(self, stage: str, key: str, fingerprint: str) -> Any:
        store_key = (stage, key, fingerprint)
        hit = self._store.get(store_key)
        if hit is not None and hit[0] == fingerprint:
            self.stats["hits"] += 1
            return hit[1]
        self.stats["misses"] += 1
        return None

    def set(self, stage: str, key: str, fingerprint: str, value: Any) -> Any:
        self._store[(stage, key, fingerprint)] = (fingerprint, value)
        return value
