"""缓存原语：带新鲜 TTL 与陈旧回退的定时缓存，以及数据指纹。

取舍 1 的实现位置：
- 历史行情缓存键 = ticker + 显式日期区间（见 fetch.py），period 在调用前换算；
- 新鲜 TTL 内命中即复用（视为含最近收盘），TTL 过期或强制刷新才取最新行情；
  取新失败时在 stale_ttl 窗口内回退上一份数据，并单独计数 stale。
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

_MISSING = object()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale: int = 0
    bypass: int = 0

    def reset(self) -> None:
        self.hits = self.misses = self.stale = self.bypass = 0


class TimedCache:
    """新鲜 TTL + 宽限陈旧 TTL 的线程安全缓存。"""

    def __init__(
        self,
        ttl: float,
        stale_ttl: Optional[float] = None,
        name: str = "cache",
        clock: Callable[[], float] = time.time,
    ):
        self.ttl = ttl
        self.stale_ttl = stale_ttl if stale_ttl is not None else ttl
        self.name = name
        self._clock = clock
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def _lookup(self, key: str) -> tuple[Any, Any]:
        """返回 (新鲜值或_MISSING, 陈旧值或_MISSING)。"""
        item = self._store.get(key)
        if item is None:
            return _MISSING, _MISSING
        value, stored_at = item
        age = self._clock() - stored_at
        if age <= self.ttl:
            return value, _MISSING
        if age <= self.stale_ttl:
            return _MISSING, value
        self._store.pop(key, None)
        return _MISSING, _MISSING

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (value, self._clock())

    def get(self, key: str) -> Any:
        with self._lock:
            fresh, _ = self._lookup(key)
            return None if fresh is _MISSING else fresh

    def latest_stale_with_prefix(self, prefix: str) -> Any:
        """取任意以 prefix 开头的键中、仍处于陈旧宽限期内的最新值。

        用于 AI 阶段：新数据指纹尚未缓存但请求失败时，复用同 ticker 的旧分析。
        """
        now = self._clock()
        with self._lock:
            candidates = [
                (stored_at, value)
                for key, (value, stored_at) in self._store.items()
                if key.startswith(prefix) and self.ttl < now - stored_at <= self.stale_ttl
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    def clear(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)

    def get_or_compute(
        self, key: str, factory: Callable[[], Any], force_refresh: bool = False
    ) -> tuple[Any, str]:
        """返回 (value, source)，source ∈ hit/miss/bypass/stale。

        hit    = 新鲜缓存命中，未触发 factory；
        miss   = 冷启动或过期后成功取新；
        bypass = force_refresh 跳过新鲜缓存取新成功；
        stale  = 取新抛错（或强制刷新失败），回退到陈旧缓存。
        """
        with self._lock:
            fresh, stale = self._lookup(key)

        if not force_refresh and fresh is not _MISSING:
            self.stats.hits += 1
            return fresh, "hit"

        if force_refresh:
            self.stats.bypass += 1

        try:
            value = factory()
        except Exception:
            if stale is not _MISSING:
                self.stats.stale += 1
                return stale, "stale"
            raise

        with self._lock:
            self._store[key] = (value, self._clock())
        if not force_refresh:
            self.stats.misses += 1
        return value, "bypass" if force_refresh else "miss"


def dataframe_fingerprint(tag: str, df: Optional[pd.DataFrame], extra: str = "") -> str:
    """给下游阶段做“输入是否变化”的复用判断。

    用行数、首末索引（日期区间）、末根 K 线的收盘价/成交量做指纹，
    同一份数据重复刷新时 indicators/predict/ai 直接复用。
    """
    if df is None or len(df) == 0:
        cols = ",".join(df.columns) if df is not None else ""
        payload = f"{tag}|empty|{cols}|{extra}"
    else:
        close = float(df["Close"].iloc[-1]) if "Close" in df else 0.0
        volume = float(df["Volume"].iloc[-1]) if "Volume" in df else 0.0
        payload = "|".join(
            [
                tag,
                str(len(df)),
                str(df.index[0]),
                str(df.index[-1]),
                f"{close:.10g}",
                f"{volume:.6g}",
                extra,
            ]
        )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
