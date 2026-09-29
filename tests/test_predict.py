import math

import pytest

from conftest import make_ohlcv
from dataflow import indicators, predict
from dataflow.fetch import empty_quotes


def test_predict_empty_returns_nan_skipped():
    r = predict.run(empty_quotes(), min_train_rows=60)
    assert math.isnan(r.prediction)
    assert r.reason


def test_predict_single_row_degraded_nan():
    df = indicators.run(make_ohlcv(1))
    r = predict.run(df, min_train_rows=60)
    assert math.isnan(r.prediction)
    assert r.n_train_rows < 60


def test_predict_insufficient_rows_degraded_not_raised():
    df = indicators.run(make_ohlcv(30))
    r = predict.run(df, min_train_rows=60)
    assert math.isnan(r.prediction)
    assert "有效样本" in r.reason


def test_predict_full_data_trains_and_scores():
    df = indicators.run(make_ohlcv(250))
    r = predict.run(df, min_train_rows=60)
    assert not math.isnan(r.prediction)
    assert r.n_train_rows >= 200
    assert not math.isnan(r.test_score)
    assert r.prediction > 0
    assert len(r.feature_importance) >= 5


def test_predict_memo_hit_does_not_retrain():
    from dataflow import FingerprintMemo
    df = indicators.run(make_ohlcv(200, seed=11))
    memo = FingerprintMemo()
    r1 = predict.run(df, memo=memo, memo_key="AAPL", min_train_rows=60)
    r2 = predict.run(df, memo=memo, memo_key="AAPL", min_train_rows=60)
    assert memo.stats["hits"] == 1
    assert r1.prediction == r2.prediction
