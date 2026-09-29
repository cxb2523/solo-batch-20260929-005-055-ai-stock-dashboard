"""五阶段流水线编排：fetch -> indicators -> predict -> ai -> render。"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from . import ai as ai_stage
from . import fetch as fetch_stage
from . import indicators as indicators_stage
from . import predict as predict_stage
from . import render as render_stage
from .cache import FingerprintMemo, QuoteCache
from .config import Config
from .trace import Tracer


@dataclass
class PipelineResult:
    ticker: str
    period: str
    fetch: fetch_stage.FetchResult
    indicators_df: Any
    prediction: predict_stage.PredictResult
    ai: ai_stage.AIResult
    render: render_stage.RenderResult
    tracer: Tracer
    banners: list[str] = field(default_factory=list)

    def trace_list(self) -> list[dict[str, Any]]:
        return self.tracer.to_list()

    def to_payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "banners": self.banners,
            "stages": self.trace_list(),
            "tabs": self.render.to_dict()["tabs"],
            "quote_source": self.fetch.source,
        }


def run_pipeline(
    ticker: str,
    *,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    config: Config | None = None,
    cache: QuoteCache | None = None,
    memo: FingerprintMemo | None = None,
    fetcher: fetch_stage.Fetcher | None = None,
    ai_provider: ai_stage.AIProvider | None = None,
    tracer: Tracer | None = None,
) -> PipelineResult:
    config = config or Config()
    memo = memo or FingerprintMemo()
    tracer = tracer or Tracer(label=ticker)

    cache = cache or QuoteCache(ttl_seconds=config.quote_ttl_seconds, path=config.cache_path)

    fetch_result = _safe(
        tracer, "fetch",
        lambda: fetch_stage.run(
            ticker, config, period=period, start=start, end=end,
            cache=cache, fetcher=fetcher, tracer=tracer,
        ),
        default=fetch_stage.FetchResult(
            ticker=(ticker or "").strip().upper() or "UNKNOWN",
            start="", end="",
            quotes=fetch_stage.empty_quotes(), source="empty",
            warnings=["fetch 阶段异常，已降级为空行情。"],
        ),
    )
    ticker_norm = fetch_result.ticker
    memo_key = f"{fetch_result.start}:{fetch_result.end}"

    indicators_df = _safe(
        tracer, "indicators",
        lambda: indicators_stage.run(
            fetch_result.quotes, tracer=tracer, memo=memo,
            memo_key=ticker_norm + "|" + memo_key,
        ),
        default=fetch_stage.empty_quotes(),
    )
    prediction = _safe(
        tracer, "predict",
        lambda: predict_stage.run(
            indicators_df, min_train_rows=config.min_train_rows,
            tracer=tracer, memo=memo, memo_key=ticker_norm + "|" + memo_key,
        ),
        default=predict_stage.PredictResult(reason="predict 阶段异常，已跳过预测"),
    )
    ai_result = _safe(
        tracer, "ai",
        lambda: ai_stage.run(
            indicators_df, fetch_result.info,
            ticker=ticker_norm, prediction=prediction.to_dict(),
            config=config, tracer=tracer, memo=memo,
            memo_key=ticker_norm + "|" + memo_key, provider=ai_provider,
        ),
        default=ai_stage.AIResult(
            headline=f"{ticker_norm} AI 解读暂不可用",
            bullets=["AI 阶段发生异常，标签页保留为占位。"],
            source="placeholder",
            warnings=["AI 阶段异常，已降级为占位内容。"],
        ),
    )
    context = render_stage.RenderContext(
        ticker=ticker_norm,
        quotes=fetch_result.quotes,
        indicators=indicators_df,
        info=fetch_result.info,
        prediction=prediction.to_dict(),
        ai=ai_result.to_dict(),
    )
    render_result = _safe(
        tracer, "render",
        lambda: render_stage.run(context, tracer=tracer),
        default=render_stage.RenderResult(tabs=[
            render_stage.Tab(id="overview", title="概览指标", status="error",
                             payload={"placeholder": True}, error="render 阶段异常"),
        ]),
    )

    banners = _collect_banners(fetch_result, prediction, ai_result, render_result)
    return PipelineResult(
        ticker=ticker_norm, period=period, fetch=fetch_result,
        indicators_df=indicators_df, prediction=prediction, ai=ai_result,
        render=render_result, tracer=tracer, banners=banners,
    )


def _safe(tracer: Tracer, stage: str, fn, *, default):
    """调用阶段；若阶段内部抛出（trace 已由其钩子记录），用 default 兜底。"""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - 整页不崩是硬要求
        rec = tracer.get(stage)
        if rec is None:
            # 阶段在进入 with 之前就炸了：补一条 trace
            with tracer.step(stage, inputs={"fatal_before_hook": True}) as record:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"
        return default


def _collect_banners(fetch_result, prediction, ai_result, render_result) -> list[str]:
    banners = list(fetch_result.warnings)
    if fetch_result.source == "cache":
        banners.insert(0, f"{fetch_result.ticker}：命中 TTL 新鲜缓存，本次未联网取最新行情。")
    elif fetch_result.source == "stale":
        banners.insert(0, f"{fetch_result.ticker}：实时拉取失败，页面展示的是旧缓存行情（stale）。")
    elif fetch_result.source == "demo":
        banners.insert(0, f"{fetch_result.ticker}：真实行情不可用，当前为 DEMO 合成数据，请勿据此判断市场。")
    if prediction.reason:
        banners.append(f"ML 预测降级：{prediction.reason}")
    banners.extend(ai_result.warnings)
    failed_tabs = [tab.title for tab in render_result.tabs if tab.status == "error"]
    if failed_tabs:
        banners.append(f"以下标签页渲染失败，已隔离为占位：{'、'.join(failed_tabs)}")
    return banners
