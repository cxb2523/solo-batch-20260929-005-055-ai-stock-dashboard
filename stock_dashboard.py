"""Streamlit 入口：五段式逻辑全部来自 dataflow 包，这里只做页面组装。

运行：streamlit run stock_dashboard.py
数据流追踪面板（Flask）见 trace_app.py。
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dataflow.pipeline import Pipeline
from dataflow.render import fmt_market_cap, fmt_num, fmt_pct, fmt_price

st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline()


def create_advanced_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """K 线 + 均线 + 布林带 + 成交量 + MACD + RSI；数据为空时返回占位图。"""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"{symbol} Price & Moving Averages",
            "Volume",
            "MACD",
            "RSI & Stochastic",
        ),
        row_heights=[0.5, 0.15, 0.2, 0.15],
    )
    if data is None or len(data) == 0:
        fig.update_layout(
            title=f"{symbol} - 暂无可用行情数据",
            template="plotly_dark",
            height=600,
        )
        return fig

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ),
        row=1,
        col=1,
    )

    for column, name, color in (
        ("SMA_20", "SMA 20", "#ff9500"),
        ("SMA_50", "SMA 50", "#007aff"),
        ("SMA_200", "SMA 200", "#5856d6"),
    ):
        if column in data.columns and data[column].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[column],
                    line=dict(color=color, width=1.5), name=name,
                ),
                row=1, col=1,
            )

    if {"BB_upper", "BB_lower"}.issubset(data.columns):
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_upper"], name="BB Upper",
                       line=dict(color="rgba(128,128,128,0.5)", width=1),
                       showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_lower"], name="BB Lower",
                       line=dict(color="rgba(128,128,128,0.5)", width=1),
                       fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
                       showlegend=False),
            row=1, col=1,
        )

    if len(data):
        volume_colors = [
            "#00ff88" if row["Close"] >= row["Open"] else "#ff4444"
            for _, row in data.iterrows()
        ]
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], marker_color=volume_colors,
                   name="Volume", opacity=0.7),
            row=2, col=1,
        )

    if {"MACD", "MACD_signal", "MACD_histogram"}.issubset(data.columns):
        fig.add_trace(
            go.Scatter(x=data.index, y=data["MACD"], name="MACD",
                       line=dict(color="#007aff", width=2)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal",
                       line=dict(color="#ff9500", width=2)),
            row=3, col=1,
        )
        histogram_colors = [
            "#00ff88" if value >= 0 else "#ff4444"
            for value in data["MACD_histogram"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(x=data.index, y=data["MACD_histogram"], name="Histogram",
                   marker_color=histogram_colors, opacity=0.6),
            row=3, col=1,
        )

    if "RSI" in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                       line=dict(color="#af52de", width=2)),
            row=4, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
    if {"Stoch_K", "Stoch_D"}.issubset(data.columns):
        fig.add_trace(
            go.Scatter(x=data.index, y=data["Stoch_K"], name="Stoch %K",
                       line=dict(color="#ffcc00", width=1.5)),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["Stoch_D"], name="Stoch %D",
                       line=dict(color="#ff6600", width=1.5)),
            row=4, col=1,
        )

    fig.update_layout(
        title=f"{symbol} - Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template="plotly_dark",
        font=dict(size=10),
    )
    for row_index in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=row_index, col=1)
    return fig


def main() -> None:
    st.title("🚀 Professional AI Stock Market Dashboard")
    st.caption("五段式数据流 dataflow：fetch → indicators → predict → ai → render")

    popular = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
    }
    with st.sidebar:
        st.header("📊 Controls")
        names = list(popular) + ["Custom"]
        choice = st.selectbox("Stock", names)
        symbol = (
            st.text_input("Symbol", value="AAPL", max_chars=10).upper()
            if choice == "Custom"
            else popular[choice]
        )
        period = st.selectbox(
            "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
        )
        force_refresh = st.button("🔄 强制刷新（绕过缓存）", type="primary")

    pipeline = get_pipeline()
    result = pipeline.run(symbol, period, force_refresh=force_refresh)
    run = result.run

    if result.degraded:
        for note in result.errors:
            st.warning(note)

    # trace 条
    with st.expander(
        f"🔎 阶段 trace（总耗时 {run.duration_ms} ms，点击展开输入列/行数/异常）"
    ):
        rows = [
            {
                "stage": s.stage,
                "status": s.status + (" (degraded)" if s.degraded else ""),
                "duration_ms": s.duration_ms,
                "reused": s.reused,
                "input_rows": s.input_rows,
                "input_tz": s.input_timezone,
                "input_columns": ", ".join(s.input_columns),
                "output_rows": s.output_rows,
                "output_tz": s.output_timezone,
                "message": s.message,
                "exception": s.exception,
            }
            for s in run.stages
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    view = result.view
    history = result.fetch_result.history

    if history is None or len(history) == 0:
        st.error(f"❌ {symbol} 无可用行情（已尝试缓存回退）。请检查代码或稍后重试。")
        return

    metric_cols = st.columns(len(view.metrics))
    for col, metric in zip(metric_cols, view.metrics):
        col.metric(metric.label, metric.value, metric.delta or None)

    tab_chart, tab_perf, tab_predict, tab_ai, tab_info, tab_raw = st.tabs(
        [
            "📈 Technical",
            "📊 Performance",
            "🔮 ML",
            "🧠 AI",
            "📋 Company",
            "🔧 Raw",
        ]
    )

    with tab_chart:
        st.plotly_chart(
            create_advanced_chart(result.indicator_frame, symbol),
            use_container_width=True,
        )

    with tab_perf:
        from dataflow.indicators import performance_metrics

        perf = performance_metrics(result.indicator_frame)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", fmt_pct(perf["total_return_pct"], 1))
        c2.metric("Volatility (Ann.)", fmt_pct(perf["volatility_annual_pct"], 1))
        c3.metric("Sharpe", fmt_num(perf["sharpe_ratio"]))
        c4.metric("Max Drawdown", fmt_pct(perf["max_drawdown_pct"], 1))

    with tab_predict:
        model = result.model_result
        if model is not None and model.status == "ok":
            c1, c2 = st.columns(2)
            c1.metric("Next Day Prediction", fmt_price(model.prediction),
                      fmt_pct(model.predicted_change_pct))
            c2.metric("Test R²", fmt_pct(model.test_score * 100, 1))
            st.caption(f"Train R² {model.train_score:.1%} · 样本 {model.n_samples} 行")
            importance = pd.DataFrame(
                list(model.feature_importance.items())[:10],
                columns=["feature", "importance"],
            )
            st.bar_chart(importance.set_index("feature"))
        else:
            st.warning(model.message if model else "暂无 ML 结果")

    with tab_ai:
        ai = result.ai_result
        source_label = {"llm": "云端大模型", "cache": "复用缓存", "local": "本地规则"}
        st.caption(f"来源：{source_label.get(ai.source, ai.source)}")
        if ai.degraded:
            st.info(ai.message)
        for insight in ai.insights:
            st.write(insight)

    with tab_info:
        quote = result.fetch_result.quote or {}
        if quote:
            st.write(
                {
                    "Name": quote.get("longName"),
                    "Sector": quote.get("sector"),
                    "Industry": quote.get("industry"),
                    "Market Cap": fmt_market_cap(quote.get("marketCap")),
                    "Exchange": quote.get("exchange"),
                }
            )
        else:
            st.warning("公司信息暂不可用")

    with tab_raw:
        raw = result.indicator_frame
        tech_columns = [
            c for c in [
                "Close", "SMA_20", "SMA_50", "RSI", "MACD",
                "MACD_signal", "BB_upper", "BB_lower", "ATR",
            ] if c in raw.columns
        ]
        st.dataframe(raw[tech_columns].tail(10).round(3), use_container_width=True)


if __name__ == "__main__":
    main()
