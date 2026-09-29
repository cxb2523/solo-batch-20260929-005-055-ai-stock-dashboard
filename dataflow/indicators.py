"""indicators 阶段：RSI / MACD / 布林带 / 随机指标等纯 pandas 计算。

取舍 3 的关键承诺：样本不足或除零时一律产出 NaN，绝不抛异常。
- 空 DataFrame：补齐全部指标列，返回同形状空表；
- 单行：所有 rolling 指标自然为 NaN，除法用 np.where 防 0；
- RSI 在涨跌幅均值均为 0（横盘）时为 NaN，而不是 100/0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INDICATOR_COLUMNS = [
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "RSI",
    "BB_middle",
    "BB_upper",
    "BB_lower",
    "Volume_SMA",
    "Volume_ratio",
    "High_Low_Pct",
    "Price_Change",
    "Price_Change_Pct",
    "ATR",
    "Stoch_K",
    "Stoch_D",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """分母为 0 / NaN 时结果为 NaN，不产生 inf 也不抛异常。"""
    num = numerator.astype("float64")
    den = denominator.astype("float64")
    result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _add_indicator_columns(empty: pd.DataFrame) -> pd.DataFrame:
    for col in INDICATOR_COLUMNS:
        empty[col] = pd.Series(index=empty.index, dtype="float64")
    return empty


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV 数据上追加技术指标列；任何输入形态都不抛异常。"""
    required = ["Open", "High", "Low", "Close", "Volume"]
    if data is None or len(data) == 0:
        base = data.copy() if data is not None else pd.DataFrame()
        for col in required:
            if col not in base.columns:
                base[col] = pd.Series(dtype="float64")
        if not isinstance(base.index, pd.DatetimeIndex) or base.index.tz is None:
            base.index = pd.DatetimeIndex(base.index if len(base) else [], tz="UTC")
        return _add_indicator_columns(base)

    df = data.copy()
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 移动平均（样本不足时前若干行自动为 NaN）
    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # RSI：横盘（gain=loss=0）返回 NaN；loss=0、gain>0 才是 100
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = delta.clip(upper=0).abs().rolling(window=14, min_periods=14).mean()
    positive_gain = (gain > 0) & (loss == 0)
    flat = (gain == 0) & (loss == 0)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~positive_gain, 100.0)
    df["RSI"] = rsi.where(~flat, np.nan)

    # 布林带
    df["BB_middle"] = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std()
    df["BB_upper"] = df["BB_middle"] + bb_std * 2
    df["BB_lower"] = df["BB_middle"] - bb_std * 2

    # 成交量（除零保护）
    df["Volume_SMA"] = df["Volume"].rolling(window=20, min_periods=20).mean()
    df["Volume_ratio"] = safe_ratio(df["Volume"], df["Volume_SMA"])

    # 价格类
    df["High_Low_Pct"] = safe_ratio(high - low, close) * 100
    df["Price_Change"] = close - df["Open"]
    df["Price_Change_Pct"] = safe_ratio(close - df["Open"], df["Open"]) * 100

    # ATR
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14, min_periods=14).mean()

    # 随机指标（区间高低点相等时除零 -> NaN）
    low_14 = low.rolling(window=14, min_periods=14).min()
    high_14 = high.rolling(window=14, min_periods=14).max()
    df["Stoch_K"] = 100 * safe_ratio(close - low_14, high_14 - low_14)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3, min_periods=3).mean()

    return df


def performance_metrics(data: pd.DataFrame) -> dict:
    """总收益 / 年化波动 / 夏普 / 最大回撤；除零与样本不足时为 NaN。"""
    result = {
        "total_return_pct": np.nan,
        "volatility_annual_pct": np.nan,
        "sharpe_ratio": np.nan,
        "max_drawdown_pct": np.nan,
    }
    if data is None or len(data) == 0 or "Close" not in data:
        return result

    close = pd.to_numeric(data["Close"], errors="coerce")
    returns = close.pct_change()

    valid_close = close.dropna()
    if len(valid_close) >= 2:
        total = valid_close.iloc[-1] / valid_close.iloc[0] - 1
        result["total_return_pct"] = total * 100
        std = returns.std()
        result["volatility_annual_pct"] = (
            std * np.sqrt(252) * 100 if std and not np.isnan(std) else np.nan
        )
        mean = returns.mean()
        result["sharpe_ratio"] = (
            mean * 252 / (std * np.sqrt(252))
            if std and not np.isnan(std) and mean is not None
            else np.nan
        )
        running_max = valid_close.cummax()
        drawdown = valid_close / running_max - 1
        result["max_drawdown_pct"] = drawdown.min() * 100
    return result
