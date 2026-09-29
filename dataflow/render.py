"""render 阶段：把五段结果组装成 Flask 页面可直接渲染的视图模型。

每个标签页独立 try/except：单标签内部出错只产生该标签的占位，
`render_dashboard` 本身不抛异常（取舍 2：整页不能崩）。
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def is_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.floating, np.integer)):
        return not (isinstance(value, float) and math.isnan(float(value)))
    return False


def fmt_price(value: Any, digits: int = 2) -> str:
    return f"${value:.{digits}f}" if is_number(value) else "N/A"


def fmt_num(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if is_number(value) else "N/A"


def fmt_int(value: Any) -> str:
    return f"{int(value):,}" if is_number(value) else "N/A"


def fmt_pct(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}%" if is_number(value) else "N/A"


def fmt_market_cap(value: Any) -> str:
    if not is_number(value):
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:,.0f}"


@dataclass
class MetricView:
    label: str
    value: str
    delta: str = ""
    tone: str = "neutral"  # positive | negative | neutral


@dataclass
class TabView:
    key: str
    title: str
    payload: dict = field(default_factory=dict)
    placeholder: str = ""

    @property
    def is_placeholder(self) -> bool:
        return bool(self.placeholder)


def sparkline_svg(
    values: pd.Series,
    width: int = 320,
    height: int = 60,
    color: str = "#4f8cff",
) -> str:
    """用纯 SVG 画迷你走势线，避免 render 阶段依赖 plotly；NaN 自动跳过。"""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return '<div class="placeholder-small">样本不足，无法绘图</div>'
    lo, hi = float(clean.min()), float(clean.max())
    span = hi - lo
    step_x = width / max(len(clean) - 1, 1)

    points = []
    for i, value in enumerate(clean.tolist()):
        y = height - 4 if span == 0 else height - 4 - (value - lo) / span * (height - 8)
        points.append(f"{i * step_x:.2f},{y:.2f}")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/></svg>'
    )


def _header_metrics(history: pd.DataFrame, indicator_frame: pd.DataFrame, quote: dict):
    metrics: list[MetricView] = []
    if history is not None and len(history) >= 1:
        latest = history.iloc[-1]
        close = latest.get("Close", np.nan)
        try:
            close = float(close)
        except (TypeError, ValueError):
            close = np.nan
        prev_close = float(history["Close"].iloc[-2]) if len(history) >= 2 else np.nan

        delta_text, tone = "", "neutral"
        if is_number(close) and is_number(prev_close) and prev_close != 0:
            change_pct = (close - prev_close) / prev_close * 100
            delta_text = f"{change_pct:+.2f}%"
            tone = "positive" if change_pct >= 0 else "negative"
        metrics.append(MetricView("最新价", fmt_price(close), delta_text, tone))

        volume = latest.get("Volume", np.nan)
        if len(history) >= 20:
            avg_volume = history["Volume"].tail(20).mean()
            if is_number(volume) and is_number(avg_volume) and avg_volume != 0:
                vol_delta = (float(volume) - avg_volume) / avg_volume * 100
                metrics.append(
                    MetricView(
                        "成交量",
                        fmt_int(volume),
                        f"{vol_delta:+.1f}% vs 20日均量",
                    )
                )
            else:
                metrics.append(MetricView("成交量", fmt_int(volume)))
        else:
            metrics.append(MetricView("成交量", fmt_int(volume)))

    if indicator_frame is not None and len(indicator_frame):
        rsi = indicator_frame["RSI"].iloc[-1] if "RSI" in indicator_frame else np.nan
        if is_number(rsi):
            status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            metrics.append(MetricView("RSI(14)", fmt_num(rsi, 1), status))
        else:
            metrics.append(MetricView("RSI(14)", "N/A"))

        sma20 = indicator_frame["SMA_20"].iloc[-1] if "SMA_20" in indicator_frame else np.nan
        close_now = (
            indicator_frame["Close"].iloc[-1]
            if "Close" in indicator_frame
            else np.nan
        )
        if is_number(sma20) and is_number(close_now) and sma20 != 0:
            distance = (close_now - sma20) / sma20 * 100
            metrics.append(
                MetricView(
                    "vs SMA20",
                    fmt_pct(distance),
                    "上方" if distance >= 0 else "下方",
                )
            )
        else:
            metrics.append(MetricView("vs SMA20", "N/A"))

    metrics.append(
        MetricView(
            "市值",
            fmt_market_cap((quote or {}).get("marketCap")),
        )
    )
    return metrics


def _price_tab(indicator_frame: pd.DataFrame) -> TabView:
    try:
        if indicator_frame is None or len(indicator_frame) == 0:
            return TabView("price", "价格走势", placeholder="暂无行情数据")
        tail = indicator_frame.tail(120)
        rows = []
        for idx, row in tail.tail(20).iloc[::-1].iterrows():
            rows.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": fmt_price(row.get("Open")),
                    "high": fmt_price(row.get("High")),
                    "low": fmt_price(row.get("Low")),
                    "close": fmt_price(row.get("Close")),
                    "volume": fmt_int(row.get("Volume")),
                }
            )
        return TabView(
            "price",
            "价格走势",
            {
                "chart": sparkline_svg(tail["Close"], color="#00c48c"),
                "rows": rows,
            },
        )
    except Exception as exc:  # noqa: BLE001 - 单标签降级
        return TabView("price", "价格走势", placeholder=f"价格标签页渲染失败：{html.escape(str(exc))}")


def _technical_tab(indicator_frame: pd.DataFrame, perf: dict) -> TabView:
    try:
        if indicator_frame is None or len(indicator_frame) == 0:
            return TabView("technical", "技术指标", placeholder="暂无技术指标数据")
        cols = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "MACD_signal", "BB_upper", "BB_lower", "ATR"]
        available = [c for c in cols if c in indicator_frame.columns]
        rows = []
        for idx, row in indicator_frame[available].tail(10).iloc[::-1].iterrows():
            rows.append({"date": idx.strftime("%Y-%m-%d"), **{c: fmt_num(row.get(c), 3) for c in available}})
        return TabView(
            "technical",
            "技术指标",
            {
                "columns": available,
                "rows": rows,
                "rsi_chart": sparkline_svg(
                    indicator_frame["RSI"].tail(120), color="#af52de"
                ),
                "performance": {
                    "total_return": fmt_pct(perf.get("total_return_pct")),
                    "volatility": fmt_pct(perf.get("volatility_annual_pct")),
                    "sharpe": fmt_num(perf.get("sharpe_ratio")),
                    "max_drawdown": fmt_pct(perf.get("max_drawdown_pct")),
                },
            },
        )
    except Exception as exc:  # noqa: BLE001
        return TabView(
            "technical",
            "技术指标",
            placeholder=f"技术标签页渲染失败：{html.escape(str(exc))}",
        )


def _prediction_tab(model_result: Any) -> TabView:
    try:
        if model_result is None or model_result.status != "ok":
            message = getattr(model_result, "message", "暂无预测结果")
            return TabView("prediction", "ML 预测", placeholder=html.escape(message))
        top_features = list(model_result.feature_importance.items())[:10]
        return TabView(
            "prediction",
            "ML 预测",
            {
                "prediction": fmt_price(model_result.prediction),
                "change": fmt_pct(model_result.predicted_change_pct),
                "train_score": fmt_pct(model_result.train_score * 100),
                "test_score": fmt_pct(model_result.test_score * 100),
                "n_samples": model_result.n_samples,
                "features": top_features,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return TabView(
            "prediction",
            "ML 预测",
            placeholder=f"ML 标签页渲染失败：{html.escape(str(exc))}",
        )


def _ai_tab(ai_result: Any) -> TabView:
    try:
        if ai_result is None:
            return TabView("ai", "AI 分析", placeholder="暂无 AI 分析")
        source_label = {"llm": "云端大模型", "cache": "复用缓存", "local": "本地规则"}.get(
            ai_result.source, ai_result.source
        )
        return TabView(
            "ai",
            "AI 分析",
            {
                "insights": [html.escape(str(line)) for line in ai_result.insights],
                "source": source_label,
                "degraded": ai_result.degraded,
                "message": html.escape(ai_result.message),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return TabView(
            "ai", "AI 分析", placeholder=f"AI 标签页渲染失败：{html.escape(str(exc))}"
        )


def _company_tab(quote: dict) -> TabView:
    try:
        if not quote:
            return TabView("company", "公司信息", placeholder="公司信息暂不可用（行情接口未返回）")
        fields = [
            ("公司名称", quote.get("longName") or quote.get("shortName")),
            ("行业板块", quote.get("sector")),
            ("细分行业", quote.get("industry")),
            ("国家/地区", quote.get("country")),
            ("官网", quote.get("website")),
            ("交易所", quote.get("exchange")),
            ("币种", quote.get("currency")),
            ("市盈率(TTM)", fmt_num(quote.get("trailingPE"))),
            ("远期市盈率", fmt_num(quote.get("forwardPE"))),
            ("市净率", fmt_num(quote.get("priceToBook"))),
            ("Beta", fmt_num(quote.get("beta"))),
            ("52周最高", fmt_price(quote.get("year_high"))),
            ("52周最低", fmt_price(quote.get("year_low"))),
        ]
        rows = [(label, html.escape(str(value))) if value not in (None, "") else (label, "N/A") for label, value in fields]
        return TabView("company", "公司信息", {"rows": rows})
    except Exception as exc:  # noqa: BLE001
        return TabView(
            "company",
            "公司信息",
            placeholder=f"公司标签页渲染失败：{html.escape(str(exc))}",
        )


@dataclass
class DashboardView:
    ticker: str
    period: str
    metrics: list[MetricView]
    tabs: list[TabView]
    data_range: dict


def render_dashboard(
    ticker: str,
    period: str,
    history: pd.DataFrame,
    indicator_frame: pd.DataFrame,
    quote: dict,
    model_result: Any,
    ai_result: Any,
    performance: dict,
) -> DashboardView:
    """始终返回完整 DashboardView；标签内部异常全部被收敛为占位。"""
    metrics = _header_metrics(history, indicator_frame, quote)
    tabs = [
        _price_tab(indicator_frame),
        _technical_tab(indicator_frame, performance),
        _prediction_tab(model_result),
        _ai_tab(ai_result),
        _company_tab(quote),
    ]

    start_label = history.index[0].strftime("%Y-%m-%d") if history is not None and len(history) else "N/A"
    end_label = history.index[-1].strftime("%Y-%m-%d") if history is not None and len(history) else "N/A"
    tz_label = str(history.index.tz) if history is not None and getattr(history.index, "tz", None) else "N/A"
    return DashboardView(
        ticker=ticker,
        period=period,
        metrics=metrics,
        tabs=tabs,
        data_range={
            "rows": 0 if history is None else int(len(history)),
            "start": start_label,
            "end": end_label,
            "timezone": tz_label,
        },
    )
