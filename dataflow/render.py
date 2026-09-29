"""render 阶段：把各阶段结果渲染成多标签页视图。

取舍 3（渲染侧）：渲染**每次刷新都重算**（行情/指标/预测/AI 已在上游
按内容指纹复用，重算渲染成本极低，且能立即反映上游变化）。

整页不崩的最后一道防线：每个标签页独立 try/except，单个标签构建失败
时该标签显示占位与错误原因，其余标签照常渲染。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .trace import Tracer, get_tracer

RAW_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INDICATOR_COLUMNS = [
    "Close", "SMA_20", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_signal",
    "BB_upper", "BB_middle", "BB_lower", "ATR", "Stoch_K", "Stoch_D",
]


@dataclass
class RenderContext:
    ticker: str
    quotes: pd.DataFrame
    indicators: pd.DataFrame | None
    info: dict[str, Any]
    prediction: dict[str, Any]
    ai: dict[str, Any]


@dataclass
class Tab:
    id: str
    title: str
    status: str = "ok"  # ok | degraded | error
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "status": self.status,
                "payload": self.payload, "error": self.error}


@dataclass
class RenderResult:
    tabs: list[Tab]

    def to_dict(self) -> dict[str, Any]:
        return {"tabs": [tab.to_dict() for tab in self.tabs]}


def run(context: RenderContext, *, tracer: Tracer | None = None) -> RenderResult:
    indicators = context.indicators if context.indicators is not None else context.quotes
    with get_tracer(tracer, "render", inputs={
        "ticker": context.ticker,
        "quotes": context.quotes,
        "indicators": indicators,
        "has_prediction": context.prediction is not None,
        "has_ai": context.ai is not None,
    }) as rec:
        builders = [
            ("overview", "概览指标", _overview_tab),
            ("chart", "K线与指标", _chart_tab),
            ("performance", "绩效", _performance_tab),
            ("prediction", "ML 预测", _prediction_tab),
            ("ai", "AI 解读", _ai_tab),
            ("data", "原始/指标数据", _data_tab),
        ]
        tabs: list[Tab] = []
        for tab_id, title, builder in builders:
            try:
                tabs.append(builder(context, indicators))
            except Exception as exc:  # 单标签失败隔离
                tabs.append(Tab(
                    id=tab_id, title=title, status="error",
                    payload={"placeholder": True},
                    error=f"{type(exc).__name__}: {exc}",
                ))
        rec.output = {"tabs": [
            {"id": t.id, "status": t.status, "error": t.error} for t in tabs
        ]}
        rec.note = f"{sum(t.status == 'ok' for t in tabs)}/{len(tabs)} 个标签正常渲染"
        return RenderResult(tabs=tabs)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _fmt_index(series: pd.Series) -> list[str]:
    if isinstance(series.index, pd.DatetimeIndex):
        return [ts.isoformat() for ts in series.index]
    return [str(v) for v in series.index]


def _overview_tab(ctx: RenderContext, df: pd.DataFrame) -> Tab:
    if df.empty:
        return Tab("overview", "概览指标", status="degraded",
                   payload={"metrics": {}, "placeholder": "无行情数据"})
    latest = df.iloc[-1]
    close = _num(latest.get("Close"))
    prev_close = _num(df["Close"].iloc[-2]) if len(df) > 1 else None
    change = change_pct = None
    if close is not None and prev_close:
        change = round(close - prev_close, 4)
        change_pct = round((close - prev_close) / prev_close * 100, 3)
    volume = _num(latest.get("Volume"))
    volume_sma = _num(latest.get("Volume_SMA"))
    volume_vs_avg = None
    if volume is not None and volume_sma:
        volume_vs_avg = round((volume - volume_sma) / volume_sma * 100, 2)
    market_cap = _num(ctx.info.get("marketCap"))
    metrics = {
        "latest_close": close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "volume_vs_avg_pct": volume_vs_avg,
        "rsi": _num(latest.get("RSI")),
        "sma_20": _num(latest.get("SMA_20")),
        "macd": _num(latest.get("MACD")),
        "market_cap": market_cap,
        "company": ctx.info.get("longName") or ctx.info.get("shortName"),
        "currency": ctx.info.get("currency"),
    }
    return Tab("overview", "概览指标", payload={"metrics": metrics})


def _chart_tab(ctx: RenderContext, df: pd.DataFrame) -> Tab:
    if df.empty:
        return Tab("chart", "K线与指标", status="degraded",
                   payload={"placeholder": "无行情数据，图表占位"})
    data = df.tail(260)
    dates = _fmt_index(data)

    candle = {
        "type": "candlestick",
        "x": dates,
        "open": [_num(v) for v in data["Open"]],
        "high": [_num(v) for v in data["High"]],
        "low": [_num(v) for v in data["Low"]],
        "close": [_num(v) for v in data["Close"]],
        "name": "K线",
    }
    traces = [candle]
    for col, name in (("SMA_20", "SMA20"), ("SMA_50", "SMA50"), ("SMA_200", "SMA200")):
        if col in data.columns and data[col].notna().any():
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": [_num(v) for v in data[col]], "name": name,
            })
    for col, name in (("BB_upper", "BB上轨"), ("BB_lower", "BB下轨")):
        if col in data.columns and data[col].notna().any():
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": [_num(v) for v in data[col]],
                "name": name, "line": {"dash": "dot"},
            })

    subplots = []
    if "MACD" in data.columns:
        subplots.append({
            "title": "MACD",
            "traces": [
                {"type": "scatter", "mode": "lines", "x": dates,
                 "y": [_num(v) for v in data["MACD"]], "name": "MACD"},
                {"type": "scatter", "mode": "lines", "x": dates,
                 "y": [_num(v) for v in data["MACD_signal"]], "name": "Signal"},
            ],
        })
    if "RSI" in data.columns:
        subplots.append({
            "title": "RSI / Stochastic",
            "traces": [
                {"type": "scatter", "mode": "lines", "x": dates,
                 "y": [_num(v) for v in data["RSI"]], "name": "RSI"},
                {"type": "scatter", "mode": "lines", "x": dates,
                 "y": [_num(v) for v in data["Stoch_K"]], "name": "Stoch %K"},
            ],
            "hlines": [{"y": 70}, {"y": 30}],
        })
    if "Volume" in data.columns:
        subplots.append({
            "title": "Volume",
            "traces": [{
                "type": "bar", "x": dates,
                "y": [_num(v) for v in data["Volume"]], "name": "Volume",
            }],
        })

    return Tab("chart", "K线与指标", payload={
        "title": f"{ctx.ticker} K线与技术指标",
        "traces": traces,
        "subplots": subplots,
        "dates": dates,
    })


def _performance_tab(ctx: RenderContext, df: pd.DataFrame) -> Tab:
    if df.empty:
        return Tab("performance", "绩效", status="degraded",
                   payload={"placeholder": "无行情数据，绩效指标占位"})
    close = df["Close"].dropna()
    daily = close.pct_change().dropna()
    cumulative = (1 + daily).cumprod() - 1
    total_return = _num(cumulative.iloc[-1] * 100) if len(cumulative) else None
    std = daily.std()
    volatility = annual_sharpe = None
    if std and not math.isnan(std) and std > 0:
        volatility = round(float(std) * math.sqrt(252) * 100, 3)
        annual_sharpe = round(float(daily.mean()) * 252 / (float(std) * math.sqrt(252)), 3)
    running_max = close.cummax()
    drawdown = (close / running_max) - 1
    max_drawdown = round(float(drawdown.min()) * 100, 3) if len(drawdown) else None
    dates = _fmt_index(cumulative)
    return Tab("performance", "绩效", payload={
        "metrics": {
            "total_return_pct": total_return,
            "volatility_annual_pct": volatility,
            "sharpe_annual": annual_sharpe,
            "max_drawdown_pct": max_drawdown,
            "observations": int(len(close)),
        },
        "cumulative_chart": {
            "type": "scatter", "mode": "lines+markers",
            "x": dates,
            "y": [_num(v * 100) for v in cumulative],
            "name": "累计收益 %",
        },
    })


def _prediction_tab(ctx: RenderContext, df: pd.DataFrame) -> Tab:
    pred = ctx.prediction or {}
    value = pred.get("prediction")
    if value is None:
        return Tab("prediction", "ML 预测", status="degraded", payload={
            "placeholder": pred.get("reason") or "样本不足，ML 预测占位（NaN）",
            "n_train_rows": pred.get("n_train_rows", 0),
        })
    importance = pred.get("feature_importance", {}) or {}
    top = list(importance.items())[:10]
    return Tab("prediction", "ML 预测", payload={
        "prediction": round(float(value), 3),
        "current_price": pred.get("current_price"),
        "predicted_change_pct": pred.get("predicted_change_pct"),
        "train_score": pred.get("train_score"),
        "test_score": pred.get("test_score"),
        "n_train_rows": pred.get("n_train_rows"),
        "importance_chart": {
            "type": "bar", "orientation": "h",
            "x": [round(float(score), 5) for _, score in top],
            "y": [name for name, _ in top],
        },
    })


def _ai_tab(ctx: RenderContext, df: pd.DataFrame) -> Tab:
    ai = ctx.ai or {}
    source = ai.get("source", "placeholder")
    if not ai:
        return Tab("ai", "AI 解读", status="degraded",
                   payload={"placeholder": "AI 不可用，标签页保留为占位"})
    status = "ok" if source == "remote" else "degraded"
    return Tab("ai", "AI 解读", status=status, payload={
        "headline": ai.get("headline", ""),
        "bullets": ai.get("bullets", []),
        "source": source,
        "model": ai.get("model"),
        "warnings": ai.get("warnings", []),
    })


def _data_tab(ctx: RenderContext, df: pd.DataFrame) -> Tab:
    if df.empty:
        return Tab("data", "原始/指标数据", status="degraded",
                   payload={"placeholder": "无数据"})
    raw_cols = [c for c in RAW_COLUMNS if c in df.columns]
    ind_cols = [c for c in INDICATOR_COLUMNS if c in df.columns]
    tail = df.tail(30)

    def table(columns: list[str]) -> dict[str, Any]:
        sub = tail[columns]
        return {
            "columns": columns,
            "rows": [
                [_idx_label(sub.index[i])] + [_num(v) for v in sub.iloc[i]]
                for i in range(len(sub))
            ],
        }

    return Tab("data", "原始/指标数据", payload={
        "raw": table(raw_cols),
        "indicators": table(ind_cols),
        "n_rows": int(len(df)),
    })


def _idx_label(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
