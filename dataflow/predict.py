"""predict 阶段：特征工程 + RandomForest 下一日收盘价预测。

样本不足时返回 status='insufficient' 的结果而不是抛异常；
训练/预测复用按输入指纹缓存（取舍 3：数据没变就不重训）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import config

FEATURE_VERSION = "rf-v1"

LAG_COLS = (1, 2, 3, 5, 10)
ROLLING_WINDOWS = (5, 10, 20, 50)


@dataclass
class ModelResult:
    status: str  # ok | insufficient
    prediction: float = np.nan
    current_price: float = np.nan
    predicted_change_pct: float = np.nan
    train_score: float = np.nan
    test_score: float = np.nan
    feature_importance: dict = field(default_factory=dict)
    n_samples: int = 0
    feature_cols: list[str] = field(default_factory=list)
    message: str = ""


def build_features(indicator_frame: pd.DataFrame) -> pd.DataFrame:
    """从带指标的帧构造 ML 特征；缺列自动补 NaN，不抛异常。"""
    if indicator_frame is None or len(indicator_frame) == 0:
        return pd.DataFrame()
    df = indicator_frame.copy()

    df["Returns"] = df["Close"].pct_change() if "Close" in df else np.nan
    df["Returns_5d"] = df["Close"].pct_change(5) if "Close" in df else np.nan
    df["Returns_10d"] = df["Close"].pct_change(10) if "Close" in df else np.nan

    for lag in LAG_COLS:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)
        df[f"Returns_lag_{lag}"] = df["Returns"].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"Close_mean_{window}"] = df["Close"].rolling(window, min_periods=window).mean()
        df[f"Close_std_{window}"] = df["Close"].rolling(window, min_periods=window).std()
        df[f"Volume_mean_{window}"] = df["Volume"].rolling(window, min_periods=window).mean()
        df[f"High_mean_{window}"] = df["High"].rolling(window, min_periods=window).mean()
        df[f"Low_mean_{window}"] = df["Low"].rolling(window, min_periods=window).mean()

    if {"Close", "SMA_20"}.issubset(df.columns):
        sma20 = df["SMA_20"].replace(0, np.nan)
        df["Price_vs_SMA20"] = (df["Close"] - sma20) / sma20 * 100
    else:
        df["Price_vs_SMA20"] = np.nan
    if {"Close", "SMA_50"}.issubset(df.columns):
        sma50 = df["SMA_50"].replace(0, np.nan)
        df["Price_vs_SMA50"] = (df["Close"] - sma50) / sma50 * 100
    else:
        df["Price_vs_SMA50"] = np.nan

    df["Price_volatility_10d"] = df["Returns"].rolling(10, min_periods=10).std()
    df["Price_volatility_20d"] = df["Returns"].rolling(20, min_periods=20).std()
    return df


def feature_columns(features: pd.DataFrame) -> list[str]:
    """显式特征清单，避免原代码里 'Returns in col' 误伤 Returns_lag 的问题。"""
    cols: list[str] = []
    for lag in LAG_COLS:
        cols += [
            f"Close_lag_{lag}",
            f"Volume_lag_{lag}",
            f"Returns_lag_{lag}",
        ]
    for window in ROLLING_WINDOWS:
        cols += [
            f"Close_mean_{window}",
            f"Close_std_{window}",
            f"Volume_mean_{window}",
            f"High_mean_{window}",
            f"Low_mean_{window}",
        ]
    cols += [
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_histogram",
        "ATR",
        "BB_upper",
        "BB_lower",
        "Stoch_K",
        "Price_vs_SMA20",
        "Price_vs_SMA50",
        "Price_volatility_10d",
        "Price_volatility_20d",
    ]
    return [c for c in cols if c in features.columns]


class PricePredictor:
    """封装 scaler/model；按输入指纹跳过未变化数据的重复训练。"""

    def __init__(self, min_samples: Optional[int] = None, random_state: int = 42):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        self.min_samples = min_samples or config.MIN_TRAIN_SAMPLES
        self._model = RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self._scaler = StandardScaler()
        self._cached_fingerprint: Optional[str] = None
        self._cached_result: Optional[ModelResult] = None
        self.train_calls = 0
        self.last_reused = False

    def train(
        self,
        indicator_frame: pd.DataFrame,
        fingerprint: str,
        force_refresh: bool = False,
    ) -> ModelResult:
        if (
            not force_refresh
            and fingerprint == self._cached_fingerprint
            and self._cached_result is not None
        ):
            self.last_reused = True
            return self._cached_result
        self.train_calls += 1
        self.last_reused = False

        features = build_features(indicator_frame)
        if len(features) == 0:
            result = ModelResult(
                status="insufficient",
                message="无行情数据，跳过 ML 预测",
            )
            self._remember(fingerprint, result)
            return result

        cols = feature_columns(features)
        x_all = features[cols].ffill().bfill()
        y_all = features["Close"].shift(-1)

        x_all = x_all.iloc[:-1]
        y_all = y_all.iloc[:-1]
        mask = ~(x_all.isna().any(axis=1) | y_all.isna())
        x_all, y_all = x_all[mask], y_all[mask]

        if len(x_all) < self.min_samples or len(cols) < 5:
            result = ModelResult(
                status="insufficient",
                current_price=float(features["Close"].dropna().iloc[-1])
                if features["Close"].notna().any()
                else np.nan,
                n_samples=int(len(x_all)),
                feature_cols=cols,
                message=f"有效样本 {len(x_all)} 行，不足 {self.min_samples} 行，跳过 ML 预测",
            )
            self._remember(fingerprint, result)
            return result

        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            x_all, y_all, test_size=0.2, random_state=42
        )
        x_train_scaled = self._scaler.fit_transform(x_train)
        x_test_scaled = self._scaler.transform(x_test)
        self._model.fit(x_train_scaled, y_train)

        last_row = features[cols].ffill().bfill().iloc[[-1]]
        prediction = float(self._model.predict(self._scaler.transform(last_row))[0])
        current_price = float(features["Close"].iloc[-1])

        result = ModelResult(
            status="ok",
            prediction=prediction,
            current_price=current_price,
            predicted_change_pct=(prediction - current_price) / current_price * 100
            if current_price
            else np.nan,
            train_score=float(self._model.score(x_train_scaled, y_train)),
            test_score=float(self._model.score(x_test_scaled, y_test)),
            feature_importance={
                col: float(imp)
                for col, imp in sorted(
                    zip(cols, self._model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            n_samples=int(len(x_all)),
            feature_cols=cols,
        )
        self._remember(fingerprint, result)
        return result

    def _remember(self, fingerprint: str, result: ModelResult) -> None:
        self._cached_fingerprint = fingerprint
        self._cached_result = result
