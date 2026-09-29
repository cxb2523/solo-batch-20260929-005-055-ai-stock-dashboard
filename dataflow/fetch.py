"""fetch 阶段：ticker 输入校验 -> 缓存查询 -> yfinance 拉取 -> 旧缓存/demo 回退。

降级链（任何一环都不让整页崩）：
    新鲜缓存命中 -> 直接复用（不联网）
    -> yfinance 实时拉取
    -> 同键旧缓存（stale-if-error，页面横幅提示）
    -> 内置合成行情（allow_demo_fallback，仅离线演示，显著标注）
    -> 返回空 DataFrame + 错误说明（后续阶段全部占位）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cache import QuoteCache, quote_key
from .config import Config
from .trace import Tracer, get_tracer

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class FetchResult:
    ticker: str
    start: str
    end: str
    quotes: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    source: str = "network"  # cache | network | stale | demo | empty
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


Fetcher = Callable[[str], tuple[pd.DataFrame, dict[str, Any]]]


def empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)


def run(
    ticker: str,
    config: Config | None = None,
    *,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    cache: QuoteCache | None = None,
    fetcher: Fetcher | None = None,
    tracer: Tracer | None = None,
) -> FetchResult:
    config = config or Config()
    cache = cache or QuoteCache(ttl_seconds=config.quote_ttl_seconds, path=None)
    fetcher = fetcher or yfinance_fetch

    raw_ticker = (ticker or "").strip()
    key = quote_key(raw_ticker or "UNKNOWN", period, start, end)
    symbol, start_date, end_date = key

    with get_tracer(tracer, "fetch", inputs={
        "ticker": symbol,
        "period": period,
        "interval": [start_date, end_date],
        "ttl_seconds": cache.ttl_seconds,
    }) as rec:
        if not raw_ticker:
            result = FetchResult(
                ticker=symbol, start=start_date, end=end_date,
                quotes=empty_quotes(), source="empty",
                error="empty ticker",
                warnings=["未提供 ticker，fetch 阶段返回空行情。"],
            )
            rec.status = "degraded"
            _fill(rec, result)
            return result

        fresh = cache.get_fresh(key)
        if fresh is not None:
            result = FetchResult(
                ticker=symbol, start=start_date, end=end_date,
                quotes=fresh.quotes, info=fresh.info, source="cache",
                warnings=list(fresh.warnings),
            )
            rec.note = (
                f"TTL {cache.ttl_seconds:.0f}s 内缓存命中，未联网；"
                f"fresh_hits={cache.stats['fresh_hits']}"
            )
            _fill(rec, result)
            return result

        network_error: str | None = None
        quotes: pd.DataFrame | None = None
        info: dict[str, Any] = {}
        try:
            quotes, info = fetcher(symbol)
            quotes = _normalize_quotes(quotes)
            if quotes is None or quotes.empty:
                network_error = "yfinance 返回空数据（ticker 可能无效或停牌）"
                quotes = None
        except Exception as exc:  # 网络/限流/库异常都走降级
            network_error = f"{type(exc).__name__}: {exc}"

        if quotes is not None and not quotes.empty:
            entry = cache.put(key, quotes, info, source="network")
            result = FetchResult(
                ticker=symbol, start=start_date, end=end_date,
                quotes=entry.quotes, info=entry.info, source="network",
            )
            _fill(rec, result)
            return result

        stale = cache.get_any(key)
        if stale is not None and not stale.quotes.empty:
            result = FetchResult(
                ticker=symbol, start=start_date, end=end_date,
                quotes=stale.quotes, info=stale.info, source="stale",
                warnings=[
                    f"实时拉取失败（{network_error}），已复用 "
                    f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(stale.fetched_at))} "
                    "的旧缓存。",
                ],
                error=network_error,
            )
            rec.status = "degraded"
            rec.note = "stale-if-error：联网失败，回退旧缓存"
            _fill(rec, result)
            return result

        if config.allow_demo_fallback:
            demo = build_demo_quotes(symbol, start_date, end_date)
            cache.put(key, demo, {"longName": f"{symbol} (DEMO 合成数据)"},
                      source="demo",
                      warnings=["真实行情不可用，当前为离线合成数据。"])
            result = FetchResult(
                ticker=symbol, start=start_date, end=end_date,
                quotes=demo,
                info={"longName": f"{symbol} (DEMO 合成数据)"},
                source="demo",
                warnings=[
                    f"实时拉取失败（{network_error}）且无旧缓存，"
                    "已用合成行情占位，数值不代表真实市场。",
                ],
                error=network_error,
            )
            rec.status = "degraded"
            rec.note = "网络与旧缓存均不可用，使用 demo 合成数据"
            _fill(rec, result)
            return result

        result = FetchResult(
            ticker=symbol, start=start_date, end=end_date,
            quotes=empty_quotes(), source="empty",
            warnings=[f"无法获取 {symbol} 行情，且未启用 demo 兜底。"],
            error=network_error,
        )
        rec.status = "degraded"
        rec.note = "无数据可用，fetch 返回空 DataFrame"
        _fill(rec, result)
        return result


def _fill(rec, result: FetchResult) -> None:
    rec.output = {
        "source": result.source,
        "quotes": result.quotes,
        "warnings": result.warnings,
        "error": result.error,
    }
    rec.extras["source"] = result.source


def _normalize_quotes(quotes: pd.DataFrame | None) -> pd.DataFrame:
    if quotes is None:
        return empty_quotes()
    df = quotes.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def yfinance_fetch(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """默认网络 fetcher，惰性导入 yfinance 便于测试替换。"""
    import yfinance as yf

    stock = yf.Ticker(symbol)
    data = stock.history(period="1y", auto_adjust=False)
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    return data, info


def build_demo_quotes(symbol: str, start: str, end: str) -> pd.DataFrame:
    """确定性合成日线（NYSE 交易日），让离线也能跑通整条流水线。"""
    seed = sum(ord(ch) for ch in symbol)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start=start, end=end, tz="America/New_York")
    n = max(len(days), 2)
    drift = 0.0004 + (seed % 7) * 0.0001
    shocks = rng.normal(drift, 0.015, size=n)
    base = 20.0 + (seed % 200)
    close = base * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 50_000_000, size=n).astype(float)
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=days[:n],
    )
    return df
