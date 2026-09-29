"""fetch 阶段：ticker 输入 -> yfinance 拉取 -> 两层缓存。

缓存粒度（取舍 1）：
- 历史 K 线按 ``ticker + start/end 日期区间`` 缓存，而不是按模糊 period，
  这样“1y”不会因为缓存命中而永远停在旧区间；
- 最新行情（last price / volume / market cap）走独立短 TTL 缓存，
  页面普通刷新时与历史缓存分别决定是否取新。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Optional

import pandas as pd

from . import config
from .cache import TimedCache

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

PERIOD_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}


def resolve_range(period: str, today: Optional[date] = None) -> tuple[str, str]:
    """把 period 换算成显式 [start, end) 日期字符串。"""
    end = today or date.today()
    days = PERIOD_DAYS.get(period, 365)
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def history_cache_key(ticker: str, start: str, end: str) -> str:
    return f"H|{ticker.strip().upper()}|{start}|{end}"


def quote_cache_key(ticker: str) -> str:
    return f"Q|{ticker.strip().upper()}"


def empty_ohlcv() -> pd.DataFrame:
    """空数据也要有正确的列与 tz-aware 索引，后续阶段一律据此返回 NaN。"""
    frame = pd.DataFrame(columns=OHLCV_COLUMNS)
    frame.index = pd.DatetimeIndex([], tz="UTC")
    return frame


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    """补齐 OHLCV 列、把索引规范成 tz-aware DatetimeIndex。"""
    if frame is None or len(frame) == 0:
        return empty_ohlcv()
    df = frame.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        # yfinance 正常返回交易所时区；无时区信息时按 UTC 兜底
        df.index = df.index.tz_localize("UTC")
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df.sort_index()


class YFinanceClient:
    """真实 yfinance 适配器；网络与库错误允许直接抛出，由 Fetcher 降级。"""

    def __init__(self) -> None:
        self._tickers: dict[str, Any] = {}

    def _ticker(self, ticker: str) -> Any:
        import yfinance as yf

        if ticker not in self._tickers:
            self._tickers[ticker] = yf.Ticker(ticker)
        return self._tickers[ticker]

    def history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self._ticker(ticker).history(start=start, end=end, auto_adjust=False)

    def quote(self, ticker: str) -> dict:
        """合并 fast_info（快）与 info（全）；任一失败都返回已拿到的部分。"""
        ticker_obj = self._ticker(ticker)
        result: dict[str, Any] = {"symbol": ticker.upper()}
        try:
            fast = ticker_obj.fast_info
            mapping = {
                "last_price": "regularMarketPrice",
                "previous_close": "regularMarketPreviousClose",
                "last_volume": "regularMarketVolume",
                "market_cap": "marketCap",
                "currency": "currency",
                "exchange": "exchange",
                "timezone": "exchangeTimezoneName",
                "year_high": "fiftyTwoWeekHigh",
                "year_low": "fiftyTwoWeekLow",
            }
            for attr, key in mapping.items():
                try:
                    value = getattr(fast, attr)
                    if value is not None and not pd.isna(value):
                        result[key] = value
                except Exception:
                    continue
        except Exception:
            pass
        try:
            info = ticker_obj.info or {}
            for key in (
                "longName",
                "shortName",
                "sector",
                "industry",
                "country",
                "website",
                "fullTimeEmployees",
                "trailingPE",
                "forwardPE",
                "pegRatio",
                "priceToBook",
                "dividendYield",
                "beta",
            ):
                if info.get(key) is not None and not pd.isna(info.get(key)):
                    result[key] = info[key]
        except Exception:
            pass
        return result


@dataclass
class FetchResult:
    ticker: str
    period: str
    start: str
    end: str
    history: pd.DataFrame
    quote: dict = field(default_factory=dict)
    source: str = "miss"
    quote_source: str = "miss"
    degraded: bool = False
    errors: list[str] = field(default_factory=list)


class Fetcher:
    """带缓存与降级的行情抓取器。client 可注入（测试用假实现）。"""

    def __init__(
        self,
        client: Any = None,
        history_ttl: Optional[float] = None,
        history_stale_ttl: Optional[float] = None,
        quote_ttl: Optional[float] = None,
        quote_stale_ttl: Optional[float] = None,
        clock: Callable[[], float] = None,
        today_fn: Callable[[], date] = date.today,
    ):
        self.client = client if client is not None else YFinanceClient()
        clock_kwargs = {"clock": clock} if clock is not None else {}
        self.history_cache = TimedCache(
            history_ttl if history_ttl is not None else config.HISTORY_TTL,
            history_stale_ttl if history_stale_ttl is not None else config.HISTORY_STALE_TTL,
            name="history",
            **clock_kwargs,
        )
        self.quote_cache = TimedCache(
            quote_ttl if quote_ttl is not None else config.QUOTE_TTL,
            quote_stale_ttl if quote_stale_ttl is not None else config.QUOTE_STALE_TTL,
            name="quote",
            **clock_kwargs,
        )
        self._today_fn = today_fn

    def _load_history(self, ticker: str, start: str, end: str, force_refresh: bool):
        key = history_cache_key(ticker, start, end)

        def factory() -> pd.DataFrame:
            frame = normalize_history(self.client.history(ticker, start, end))
            if len(frame) == 0:
                # 空结果不进缓存，避免一次空响应在 TTL 内被反复命中
                raise ValueError(f"empty history for {ticker} [{start}, {end})")
            return frame

        try:
            frame, source = self.history_cache.get_or_compute(
                key, factory, force_refresh=force_refresh
            )
            return frame, source, None
        except Exception as exc:
            return empty_ohlcv(), "miss", str(exc)

    def _load_quote(self, ticker: str, force_refresh: bool):
        key = quote_cache_key(ticker)
        try:
            quote, source = self.quote_cache.get_or_compute(
                key, lambda: self.client.quote(ticker), force_refresh=force_refresh
            )
            return quote or {}, source, None
        except Exception as exc:
            return {}, "miss", str(exc)

    def fetch(
        self, ticker: str, period: str = "1y", force_refresh: bool = False
    ) -> FetchResult:
        ticker = (ticker or "").strip().upper()
        start, end = resolve_range(period, today=self._today_fn())
        errors: list[str] = []

        history, source, hist_error = self._load_history(
            ticker, start, end, force_refresh
        )
        if hist_error:
            errors.append(f"history: {hist_error}")

        quote, quote_source, quote_error = self._load_quote(ticker, force_refresh)
        if quote_error:
            errors.append(f"quote: {quote_error}")

        return FetchResult(
            ticker=ticker,
            period=period,
            start=start,
            end=end,
            history=history,
            quote=quote,
            source=source,
            quote_source=quote_source,
            degraded=bool(errors),
            errors=errors,
        )

    def clear_cache(self) -> None:
        self.history_cache.clear()
        self.quote_cache.clear()
