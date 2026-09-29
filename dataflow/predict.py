"""predict 阶段：特征工程 + 随机森林预测下一交易日收盘价。

降级规则（不抛异常）：
* 空数据 / 少于 ``min_train_rows`` 行有效样本 -> prediction=NaN，status=degraded；
* 特征不足 5 个 / sklearn 训练失败 -> prediction=NaN，记录 error；
* 同一份输入内容指纹 -> 直接 memo 复用模型结果，刷新不重训。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cache import FingerprintMemo, content_fingerprint
from .trace import Tracer, get_tracer

RAW_EXCLUDE = {
    "Open", "High", "Low", "Close", "Volume",
    "Dividends", "Stock Splits",
}
FEATURE_WHITELIFT = ("lag", "mean", "std", "Price_vs_", "volatility", "Returns")


@dataclass
class PredictResult:
    prediction: float = math.nan
    current_price: float = math.nan
    predicted_change_pct: float = math.nan
    train_score: float = math.nan
    test_score: float = math.nan
    n_train_rows: int = 0
    feature_cols: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": _maybe_nan(self.prediction),
            "current_price": _maybe_nan(self.current_price),
            "predicted_change_pct": _maybe_nan(self.predicted_change_pct),
            "train_score": _maybe_nan(self.train_score),
            "test_score": _maybe_nan(self.test_score),
            "n_train_rows": self.n_train_rows,
            "feature_cols": self.feature_cols,
            "feature_importance": self.feature_importance,
            "reason": self.reason,
        }


def _maybe_nan(value: float) -> float | None:
    try:
        return None if (value is None or math.isnan(float(value))) else float(value)
    except (TypeError, ValueError):
        return None


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """纯函数式特征工程，不依赖 sklearn，空/单行输入安全。"""
    df = data.copy()
    if df.empty:
        return df

    df["Returns"] = df["Close"].pct_change()
    df["Returns_5d"] = df["Close"].pct_change(5)
    df["Returns_10d"] = df["Close"].pct_change(10)

    for lag in (1, 2, 3, 5, 10):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)
        df[f"Returns_lag_{lag}"] = df["Returns"].shift(lag)

    for window in (5, 10, 20, 50):
        df[f"Close_mean_{window}"] = df["Close"].rolling(window, min_periods=window).mean()
        df[f"Close_std_{window}"] = df["Close"].rolling(window, min_periods=window).std()
        df[f"Volume_mean_{window}"] = df["Volume"].rolling(window, min_periods=window).mean()

    for col in ("SMA_20", "SMA_50"):
        if col in df.columns:
            ratio = (df["Close"] - df[col]) / df[col] * 100
            df[f"Price_vs_{col}"] = ratio.replace([np.inf, -np.inf], np.nan)

    df["Price_volatility_10d"] = df["Returns"].rolling(10, min_periods=10).std()
    df["Price_volatility_20d"] = df["Returns"].rolling(20, min_periods=20).std()
    return df


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for col in df.columns:
        if col in RAW_EXCLUDE:
            continue
        if (
            "lag" in col
            or "mean" in col
            or "std" in col
            or col.startswith("Price_vs_")
            or "volatility" in col
            or col in ("RSI", "MACD", "ATR", "Returns")
        ):
            selected.append(col)
    return selected


def run(
    indicators_df: pd.DataFrame,
    *,
    min_train_rows: int = 60,
    tracer: Tracer | None = None,
    memo: FingerprintMemo | None = None,
    memo_key: str = "",
) -> PredictResult:
    fingerprint = content_fingerprint(indicators_df)
    if memo is not None and memo_key:
        cached = memo.get("predict", memo_key, fingerprint)
        if cached is not None:
            with get_tracer(tracer, "predict",
                            inputs={"indicators": indicators_df, "memo": "hit"}) as rec:
                rec.output = cached.to_dict()
                rec.note = "模型结果按内容指纹复用，未重训"
                return cached

    with get_tracer(tracer, "predict",
                    inputs={"indicators": indicators_df,
                            "min_train_rows": min_train_rows,
                            "memo": "miss" if memo else "off"}) as rec:
        result = PredictResult()

        def finish(status: str, note: str | None = None) -> PredictResult:
            rec.status = status
            rec.note = note
            rec.output = result.to_dict()
            if memo is not None and memo_key and status == "ok":
                memo.set("predict", memo_key, fingerprint, result)
            return result

        if indicators_df is None or indicators_df.empty:
            result.reason = "无行情数据，跳过 ML 预测"
            return finish("skipped", result.reason)

        result.current_price = float(indicators_df["Close"].dropna().iloc[-1]) \
            if indicators_df["Close"].notna().any() else math.nan

        features = add_features(indicators_df)
        feature_cols = select_feature_cols(features)
        result.feature_cols = feature_cols

        if len(feature_cols) < 5:
            result.reason = f"可用特征仅 {len(feature_cols)} 个（需 >=5），跳过预测"
            return finish("degraded", result.reason)

        target = features["Close"].shift(-1)
        x_all = features[feature_cols].ffill().bfill()
        x_all = x_all.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        x_all = x_all.iloc[:-1]
        y_all = target.iloc[:-1]
        mask = x_all.notna().all(axis=1) & y_all.notna()
        x_all = x_all[mask]
        y_all = y_all[mask]
        result.n_train_rows = int(len(x_all))

        if result.n_train_rows < min_train_rows:
            result.reason = (
                f"有效样本 {result.n_train_rows} 行 < 最小 {min_train_rows} 行，"
                "预测返回 NaN"
            )
            return finish("degraded", result.reason)

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            x_train, x_test, y_train, y_test = train_test_split(
                x_all, y_all, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            model.fit(x_train_s, y_train)
            result.train_score = float(model.score(x_train_s, y_train))
            result.test_score = float(model.score(x_test_s, y_test))

            latest = features[feature_cols].ffill().bfill().iloc[-1:].fillna(0.0)
            result.prediction = float(model.predict(scaler.transform(latest))[0])
            if result.current_price and not math.isnan(result.current_price):
                change = (result.prediction - result.current_price) / result.current_price * 100
                result.predicted_change_pct = float(change)
            importance = model.feature_importances_
            result.feature_importance = {
                col: float(score)
                for col, score in sorted(
                    zip(feature_cols, importance), key=lambda kv: kv[1], reverse=True
                )
            }
            return finish("ok", f"训练样本 {result.n_train_rows} 行，模型训练成功")
        except Exception as exc:
            result.reason = f"ML 训练异常: {type(exc).__name__}: {exc}"
            return finish("error", result.reason)
