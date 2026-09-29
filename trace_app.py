"""按 ticker 展示五段数据流实际数据与耗时的 Flask 应用。

运行：python trace_app.py  （默认 http://127.0.0.1:5000）
设置 DATAFLOW_DEMO=1 可在无网络环境用内置合成数据演示降级与耗时。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from dataflow.ai import AIAnalyzer
from dataflow.fetch import Fetcher, normalize_history
from dataflow.pipeline import Pipeline
from dataflow.predict import PricePredictor

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]


class DemoClient:
    """离线演示用假 yfinance 客户端：合成可复现的 K 线与报价。"""

    def __init__(self) -> None:
        self.history_calls = 0
        self.quote_calls = 0

    def history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        self.history_calls += 1
        symbol = (ticker or "AAPL").upper()
        index = pd.bdate_range(start=start, end=end, tz="UTC")
        if symbol == "EMPTY":
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            ).set_index(pd.DatetimeIndex([], tz="UTC"))
        if symbol == "BROKEN":
            raise ConnectionError("simulated network outage")
        seed = sum(ord(ch) for ch in symbol)
        rng = np.random.default_rng(seed)
        drift = 0.0003 + (seed % 7) * 0.0002
        log_returns = rng.normal(drift, 0.012, size=len(index))
        close = 100 * np.exp(np.cumsum(log_returns))
        open_ = close * (1 + rng.normal(0, 0.004, size=len(index)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, len(index)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, len(index)))
        volume = rng.integers(5_000_000, 60_000_000, size=len(index))
        frame = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume.astype(float),
            },
            index=index,
        )
        return normalize_history(frame)

    def quote(self, ticker: str) -> dict:
        self.quote_calls += 1
        symbol = ticker.upper()
        if symbol == "BROKEN":
            raise ConnectionError("simulated quote outage")
        seed = sum(ord(ch) for ch in symbol)
        return {
            "symbol": symbol,
            "regularMarketPrice": 100 + seed % 150,
            "regularMarketPreviousClose": 99 + seed % 150,
            "regularMarketVolume": 20_000_000 + seed * 1000,
            "marketCap": (50 + seed) * 1e9,
            "currency": "USD",
            "exchange": "DEMO",
            "exchangeTimezoneName": "UTC",
            "fiftyTwoWeekHigh": 200 + seed % 50,
            "fiftyTwoWeekLow": 60 + seed % 20,
            "longName": f"{symbol} Demo Corp",
            "sector": "Technology",
            "industry": "Software",
            "country": "United States",
            "website": f"https://example.invalid/{symbol.lower()}",
        }


class BrokenAiClient:
    """演示 AI 请求失败时的降级路径。"""

    available = True
    model = "demo-model"

    def analyze(self, prompt: str) -> str:
        raise ConnectionError("simulated AI outage")


def build_pipeline() -> Pipeline:
    if os.getenv("DATAFLOW_DEMO") == "1":
        client = DemoClient()
        fetcher = Fetcher(client=client)
        ai_client = BrokenAiClient() if os.getenv("DATAFLOW_DEMO_AI_FAIL") == "1" else None
        analyzer = AIAnalyzer(client=ai_client) if ai_client is not None else AIAnalyzer()
        return Pipeline(
            fetcher=fetcher,
            predictor=PricePredictor(),
            analyzer=analyzer,
        )
    return Pipeline(
        fetcher=Fetcher(),
        predictor=PricePredictor(),
        analyzer=AIAnalyzer(),
    )


pipeline = build_pipeline()


def create_app(pipeline_obj: Pipeline = None) -> Flask:
    app = Flask(__name__)
    pipe = pipeline_obj or pipeline

    def _run(ticker: str, period: str, force_refresh: bool):
        ticker = (ticker or "AAPL").strip().upper()[:12]
        period = period if period in PERIODS else "1y"
        return pipe.run(ticker, period, force_refresh=force_refresh)

    @app.route("/")
    def index():
        ticker = request.args.get("ticker", "AAPL")
        period = request.args.get("period", "1y")
        force_refresh = request.args.get("refresh") == "1"
        result = _run(ticker, period, force_refresh)
        history_runs = pipe.store.history(result.ticker, limit=8)
        return render_template(
            "dashboard.html",
            result=result,
            view=result.view,
            run=result.run,
            stages=result.run.stages if result.run else [],
            history=history_runs,
            periods=PERIODS,
        )

    @app.route("/trace/<ticker>")
    def trace_json(ticker: str):
        runs = pipe.store.history(ticker, limit=int(request.args.get("limit", 10)))
        return jsonify(
            {
                "ticker": ticker.upper(),
                "runs": [run.to_dict() for run in runs],
            }
        )

    @app.route("/healthz")
    def healthz():
        return {"status": "ok", "tickers": pipe.store.tickers()}

    @app.route("/reset", methods=["POST"])
    def reset():
        pipe.fetcher.clear_cache()
        pipe.store.clear()
        return redirect(url_for("index"))

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
