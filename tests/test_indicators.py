import numpy as np
import pandas as pd

from dataflow.fetch import empty_ohlcv
from dataflow.indicators import INDICATOR_COLUMNS, add_indicators, performance_metrics, safe_ratio
from .helpers import synthetic_ohlcv


def test_empty_dataframe_has_all_indicator_columns_and_no_raise():
    out = add_indicators(empty_ohlcv())
    assert len(out) == 0
    for col in INDICATOR_COLUMNS:
        assert col in out.columns
    assert str(out.index.tz) == "UTC"
    assert np.issubdtype(out["RSI"].dtype, np.floating)


def test_single_row_all_window_indicators_are_nan():
    one = synthetic_ohlcv(rows=1)
    out = add_indicators(one)
    assert len(out) == 1
    windowed = [
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_signal",
        "BB_upper",
        "BB_lower",
        "ATR",
        "Stoch_K",
        "Volume_SMA",
        "Volume_ratio",
    ]
    for col in windowed:
        assert pd.isna(out[col].iloc[0]), f"{col} 应在单行时为 NaN"


def test_division_by_zero_returns_nan_instead_of_inf_or_raise():
    df = pd.DataFrame(
        {
            "Open": [0.0] * 30,
            "High": [100.0] * 30,
            "Low": [100.0] * 30,
            "Close": [100.0] * 30,
            "Volume": [0.0] * 30,
        },
        index=pd.bdate_range("2026-01-01", periods=30, tz="UTC"),
    )
    out = add_indicators(df)
    # Open=0 的涨跌幅、Volume_SMA=0 的量比、High-Low=0 的随机指标均须 NaN
    assert pd.isna(out["Price_Change_Pct"].dropna()).all() or out["Price_Change_Pct"].isna().all()
    assert out["Volume_ratio"].isna().all()
    assert out["Stoch_K"].isna().all()
    assert not np.isinf(out.select_dtypes(include="number")).any().any()


def test_flat_prices_rsi_is_nan_not_100():
    flat = synthetic_ohlcv(rows=30, flat=True)
    out = add_indicators(flat)
    assert out["RSI"].dropna().empty, "横盘（gain=loss=0）RSI 必须为 NaN"


def test_rsi_is_100_when_only_gains_and_bounded_otherwise():
    closes = [100.0]
    for i in range(1, 40):
        # 持续上涨但幅度变化，避免 rolling 均值出现“零变动”窗口
        closes.append(closes[-1] + (1.0 + (i % 3) * 0.5))
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * 40,
        },
        index=pd.bdate_range("2026-01-01", periods=40, tz="UTC"),
    )
    out = add_indicators(df)
    valid_rsi = out["RSI"].dropna()
    assert not valid_rsi.empty
    assert (valid_rsi == 100).all()


def test_safe_ratio_helper():
    num = pd.Series([1.0, 1.0, np.nan])
    den = pd.Series([0.0, 2.0, 1.0])
    result = safe_ratio(num, den)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 0.5
    assert pd.isna(result.iloc[2])


def test_performance_metrics_safe_for_empty_and_short_data():
    empty_metrics = performance_metrics(empty_ohlcv())
    for value in empty_metrics.values():
        assert pd.isna(value)

    short = synthetic_ohlcv(rows=1)
    one_metrics = performance_metrics(short)
    # 单行无法计算收益/波动，全部 NaN 而不是除零异常
    assert pd.isna(one_metrics["sharpe_ratio"])
    assert pd.isna(one_metrics["volatility_annual_pct"])

    normal = add_indicators(synthetic_ohlcv(rows=60))
    metrics = performance_metrics(normal)
    assert not pd.isna(metrics["total_return_pct"])
