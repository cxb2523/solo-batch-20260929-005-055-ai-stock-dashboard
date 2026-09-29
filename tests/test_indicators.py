import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from dataflow import indicators
from dataflow.fetch import empty_quotes
from dataflow.trace import Tracer


def test_empty_quotes_returns_empty_without_error():
    tracer = Tracer()
    out = indicators.run(empty_quotes(), tracer=tracer)
    assert out.empty
    assert list(out.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
    rec = tracer.get("indicators")
    assert rec.status == "degraded"
    assert rec.inputs["quotes"]["rows"] == 0
    assert rec.elapsed_ms is not None


def test_single_row_only_nan_indicators():
    df = make_ohlcv(1)
    tracer = Tracer()
    out = indicators.run(df, tracer=tracer)
    assert len(out) == 1
    for col in ("SMA_20", "RSI", "MACD_signal", "BB_upper", "ATR", "Stoch_K"):
        assert math.isnan(out[col].iloc[0]), f"{col} 在单行时必须为 NaN"
    rec = tracer.get("indicators")
    assert rec.ok
    assert rec.inputs["quotes"]["rows"] == 1
    assert rec.inputs["quotes"]["timezone"] == "America/New_York"


def test_zero_division_flat_prices_and_zero_volume_never_inf():
    flat = make_ohlcv(40, flat=True, zero_volume=True)
    out = indicators.run(flat)
    ratio_cols = ["RSI", "Volume_ratio", "High_Low_Pct",
                  "Price_Change_Pct", "Stoch_K"]
    for col in ratio_cols:
        values = out[col].to_numpy(dtype=float)
        assert not np.isinf(values).any(), f"{col} 不允许出现 inf"
    # 量比分母全 0 -> 全 NaN
    assert out["Volume_ratio"].isna().all()
    # 前 13 行 RSI 窗口不足 -> NaN；恒定价格段 RSI 也不得是 inf
    assert out["RSI"].iloc[:13].isna().all()


def test_constant_prices_rsi_is_nan_not_100_when_no_movement():
    flat = make_ohlcv(30, flat=True)
    out = indicators.run(flat)
    # gain=loss=0 时 RSI 必须是 NaN（0/0 未定义），不能硬塞 100
    assert out["RSI"].dropna().empty


def test_bollinger_bands_flat_market_width_zero_not_nan_crash():
    flat = make_ohlcv(30, flat=True)
    out = indicators.run(flat)
    valid = out["BB_upper"].dropna()
    assert not valid.empty
    np.testing.assert_allclose(valid, 100.0, atol=1e-8)


def test_memo_hit_skips_recompute():
    from dataflow import FingerprintMemo
    df = make_ohlcv(60)
    memo = FingerprintMemo()
    first = indicators.run(df, memo=memo, memo_key="AAPL|x")
    second = indicators.run(df, memo=memo, memo_key="AAPL|x")
    pd.testing.assert_frame_equal(first, second)
    assert memo.stats["hits"] == 1 and memo.stats["misses"] == 1


def test_missing_required_column_raises_and_trace_records():
    tracer = Tracer()
    bad = pd.DataFrame({"Close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="缺少必需列"):
        indicators.run(bad, tracer=tracer)
    rec = tracer.get("indicators")
    assert rec.status == "error" and "缺少必需列" in rec.error
    assert rec.elapsed_ms is not None


def test_short_data_runs_and_ticker_change_changes_timing_without_crash():
    # 2 行数据：指标大量 NaN，但阶段整体不抛
    tiny = make_ohlcv(2)
    tracer = Tracer()
    out = indicators.run(tiny, tracer=tracer)
    assert len(out) == 2
    assert tracer.get("indicators").ok
