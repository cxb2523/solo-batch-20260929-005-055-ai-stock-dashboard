import numpy as np

from dataflow.fetch import empty_ohlcv
from dataflow.indicators import add_indicators
from dataflow.predict import PricePredictor, build_features
from .helpers import synthetic_ohlcv


def test_build_features_empty_and_single_row():
    assert len(build_features(empty_ohlcv())) == 0
    one = build_features(add_indicators(synthetic_ohlcv(rows=1)))
    assert len(one) == 1
    assert np.isnan(one["RSI"].iloc[0])


def test_predictor_insufficient_samples_returns_status_not_raise():
    predictor = PricePredictor(min_samples=60)
    frame = add_indicators(synthetic_ohlcv(rows=10))
    result = predictor.train(frame, "fp-small")
    assert result.status == "insufficient"
    assert np.isnan(result.prediction)
    assert "不足" in result.message


def test_predictor_trains_and_predicts_on_synthetic_data():
    predictor = PricePredictor(min_samples=60)
    frame = add_indicators(synthetic_ohlcv(rows=120, seed=11))
    result = predictor.train(frame, "fp-120")
    assert result.status == "ok"
    assert not np.isnan(result.prediction)
    assert 0.0 <= result.test_score <= 1.0001
    assert result.n_samples >= 60


def test_predictor_reuses_model_for_same_fingerprint():
    predictor = PricePredictor(min_samples=60)
    frame = add_indicators(synthetic_ohlcv(rows=120, seed=11))
    first = predictor.train(frame, "fp-reuse")
    calls_after_first = predictor.train_calls
    second = predictor.train(frame, "fp-reuse")
    assert predictor.train_calls == calls_after_first  # 同指纹未重训
    assert predictor.last_reused is True
    assert first.prediction == second.prediction

    changed = add_indicators(synthetic_ohlcv(rows=121, seed=11))
    predictor.train(changed, "fp-reuse-2")
    assert predictor.train_calls == calls_after_first + 1
    assert predictor.last_reused is False


def test_predictor_empty_data_is_insufficient():
    predictor = PricePredictor(min_samples=60)
    result = predictor.train(empty_ohlcv(), "fp-empty")
    assert result.status == "insufficient"
