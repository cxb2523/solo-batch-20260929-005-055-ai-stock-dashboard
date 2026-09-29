"""indicators 阶段：技术指标计算。

取舍 3（后半）：样本不足或除零一律产生 NaN，不抛异常。
* rolling 窗口大于行数 -> 前 N-1 行自然为 NaN；
* 所有比值（RS、量比、Stochastic、涨跌幅）走 :func:`safe_divide`，
  分母为 0/NaN 时结果为 NaN，绝不会出现 inf 或 ZeroDivisionError。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cache import FingerprintMemo, content_fingerprint
from .fetch import OHLCV_COLUMNS
from .trace import Tracer, get_tracer


def safe_divide(numerator: Any, denominator: Any) -> Any:
    """分母为 0 / NaN 时返回 NaN（pandas 标量或 Series 均可）。"""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(num, den, out=np.full_like(den, np.nan, dtype=float),
                          where=(den != 0) & np.isfinite(den))
    scalar = np.isscalar(numerator) and np.isscalar(denominator)
    return ratio.item() if scalar and ratio.shape == () else pd.Series(ratio, index=getattr(denominator, "index", None))


def _ewm_min(series: pd.Series, *, span: int) -> pd.Series:
    """指数移动平均，但预热期（不足 span 个样本）置 NaN。

    与 rolling 指标口径一致：单行/短样本时返回 NaN，而不是 ewm 默认
    给出的"看似有效"的值。
    """
    ewm = series.ewm(span=span, adjust=False, min_periods=span).mean()
    return ewm


def run(
    quotes: pd.DataFrame,
    *,
    tracer: Tracer | None = None,
    memo: FingerprintMemo | None = None,
    memo_key: str = "",
) -> pd.DataFrame:
    """在 OHLCV 行情上叠加技术指标；输入空 DataFrame 时原样返回。"""
    fingerprint = content_fingerprint(quotes)
    if memo is not None and memo_key:
        cached = memo.get("indicators", memo_key, fingerprint)
        if cached is not None:
            with get_tracer(tracer, "indicators",
                            inputs={"quotes": quotes, "memo": "hit"}) as rec:
                rec.output = {"indicators": cached, "memo": "hit"}
                rec.note = "指标按内容指纹复用，未重算"
                return cached

    with get_tracer(tracer, "indicators",
                    inputs={"quotes": quotes, "memo": "miss" if memo else "off"}) as rec:
        required = [c for c in OHLCV_COLUMNS if c not in quotes.columns]
        if required:
            raise ValueError(f"行情缺少必需列: {required}")

        if quotes.empty:
            # 空数据：保留标准列结构直接透传，后续渲染走占位
            result = quotes.copy()
            rec.status = "degraded"
            rec.note = "空行情，指标全部为 NaN（未计算）"
            rec.output = {"indicators": result}
            return result

        df = quotes.copy()
        close = df["Close"]

        df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
        df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
        df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()

        df["EMA_12"] = _ewm_min(close, span=12)
        df["EMA_26"] = _ewm_min(close, span=26)
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_signal"] = _ewm_min(df["MACD"], span=9)
        df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
        rs = safe_divide(gain, loss)
        df["RSI"] = 100 - (100 / (1 + rs))
        # 连续 14 日零跌幅（分母 0）-> RSI=100；双方都 0 -> NaN
        both_zero = (gain == 0) & (loss == 0)
        df.loc[both_zero, "RSI"] = np.nan

        df["BB_middle"] = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std()
        df["BB_upper"] = df["BB_middle"] + bb_std * 2
        df["BB_lower"] = df["BB_middle"] - bb_std * 2

        volume_sma = df["Volume"].rolling(window=20, min_periods=20).mean()
        df["Volume_SMA"] = volume_sma
        df["Volume_ratio"] = safe_divide(df["Volume"], volume_sma)

        df["High_Low_Pct"] = safe_divide(df["High"] - df["Low"], close) * 100
        df["Price_Change"] = close - df["Open"]
        df["Price_Change_Pct"] = safe_divide(close - df["Open"], df["Open"]) * 100

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - close.shift()).abs()
        low_close = (df["Low"] - close.shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["True_Range"] = true_range
        df["ATR"] = true_range.rolling(window=14, min_periods=14).mean()

        low_14 = df["Low"].rolling(window=14, min_periods=14).min()
        high_14 = df["High"].rolling(window=14, min_periods=14).max()
        stoch_k = safe_divide(close - low_14, high_14 - low_14) * 100
        df["Stoch_K"] = stoch_k
        df["Stoch_D"] = stoch_k.rolling(window=3, min_periods=3).mean()

        # 保险：任何残留 inf 都转 NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        rec.output = {"indicators": df}
        if memo is not None and memo_key:
            memo.set("indicators", memo_key, fingerprint, df)
        return df
