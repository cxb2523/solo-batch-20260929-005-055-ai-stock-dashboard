"""Flask trace 应用：按 ticker 展示 dataflow 各阶段实际数据与耗时。

运行：python trace_app.py  -> http://127.0.0.1:5050

页面刷新语义（取舍 3）：
* 60s TTL 内同 ticker+日期区间 -> fetch 命中缓存不联网；
* 指标/预测/AI 按输入内容指纹 memo（行情没变就复用，trace 标注 hit）；
* render 每次重算；换 ticker -> 全部重算，阶段耗时与降级提示随之变化。
"""

from __future__ import annotations

import threading

from flask import Flask, jsonify, render_template, request

from dataflow import Config, FingerprintMemo, QuoteCache, run_pipeline
from dataflow.fetch import Fetcher
from dataflow.ai import AIProvider

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"]


def create_app(
    config: Config | None = None,
    *,
    cache: QuoteCache | None = None,
    memo: FingerprintMemo | None = None,
    fetcher: Fetcher | None = None,
    ai_provider: AIProvider | None = None,
) -> Flask:
    config = config or Config.from_env()
    # 共享缓存/memo：刷新页面时缓存命中次数、内容指纹复用才可见
    cache = cache or QuoteCache(ttl_seconds=config.quote_ttl_seconds, path=config.cache_path)
    memo = memo or FingerprintMemo()
    lock = threading.Lock()

    app = Flask(__name__)
    app.config["DATAFLOW_CONFIG"] = config

    @app.get("/")
    def index():
        ticker = (request.args.get("ticker") or "AAPL").strip().upper()
        period = request.args.get("period") or "1y"
        if period not in PERIODS:
            period = "1y"
        return render_template(
            "trace.html",
            ticker=ticker, period=period,
            periods=PERIODS, tickers=DEFAULT_TICKERS,
        )

    def _run(ticker: str, period: str):
        with lock:
            return run_pipeline(
                ticker, period=period, config=config,
                cache=cache, memo=memo,
                fetcher=fetcher, ai_provider=ai_provider,
            )

    @app.get("/api/run")
    def api_run():
        ticker = (request.args.get("ticker") or "").strip()
        period = request.args.get("period") or "1y"
        if period not in PERIODS:
            period = "1y"
        result = _run(ticker, period)
        payload = result.to_payload()
        payload["cache_stats"] = cache.stats
        payload["memo_stats"] = memo.stats
        return jsonify(payload)

    @app.get("/api/trace")
    def api_trace():
        ticker = (request.args.get("ticker") or "").strip().upper()
        period = request.args.get("period") or "1y"
        if period not in PERIODS:
            period = "1y"
        result = _run(ticker, period)
        return jsonify({
            "ticker": result.ticker,
            "quote_source": result.fetch.source,
            "banners": result.banners,
            "stages": result.trace_list(),
            "cache_stats": cache.stats,
            "memo_stats": memo.stats,
        })

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
