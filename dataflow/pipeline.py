"""五段式编排：fetch -> indicators -> predict -> ai -> render。

刷新策略（取舍 3）：
- fetch：force_refresh 绕过新鲜缓存取新；普通刷新由各自 TTL 决定；
- indicators / predict / ai：输入指纹不变时直接复用上次结果（render 每次重算）；
- 任何阶段异常都被捕获并写入 trace，后续阶段以空数据/占位继续，整页不崩。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .ai import AIAnalyzer
from .cache import dataframe_fingerprint
from .fetch import Fetcher
from .indicators import add_indicators, performance_metrics
from .predict import FEATURE_VERSION, PricePredictor
from .render import DashboardView, render_dashboard
from .trace import RunTrace, TraceStore, trace_store


@dataclass
class PipelineResult:
    ticker: str
    period: str
    view: Optional[DashboardView] = None
    run: Optional[RunTrace] = None
    fetch_result: Any = None
    indicator_frame: Optional[pd.DataFrame] = None
    model_result: Any = None
    ai_result: Any = None
    degraded: bool = False
    errors: list[str] = field(default_factory=list)


class Pipeline:
    """持有跨刷新复用的 fetcher / predictor / analyzer（进程内单例语义）。"""

    def __init__(
        self,
        fetcher: Optional[Fetcher] = None,
        predictor: Optional[PricePredictor] = None,
        analyzer: Optional[AIAnalyzer] = None,
        store: Optional[TraceStore] = None,
    ):
        self.fetcher = fetcher or Fetcher()
        self.predictor = predictor or PricePredictor()
        self.analyzer = analyzer or AIAnalyzer()
        self.store = store or trace_store
        self._indicator_fingerprint: Optional[str] = None
        self._indicator_frame: Optional[pd.DataFrame] = None

    def run(self, ticker: str, period: str = "1y", force_refresh: bool = False) -> PipelineResult:
        ticker = (ticker or "").strip().upper()
        run = self.store.start_run(ticker, period, force_refresh)
        result = PipelineResult(ticker=ticker, period=period, run=run)

        # ---- fetch ----
        with run.span("fetch", input_data=None) as stage:
            try:
                fetch_result = self.fetcher.fetch(ticker, period, force_refresh=force_refresh)
                result.fetch_result = fetch_result
                stage.set_output(fetch_result.history)
                stage.extra["history_source"] = fetch_result.source
                stage.extra["quote_source"] = fetch_result.quote_source
                stage.extra["range"] = [fetch_result.start, fetch_result.end]
                if fetch_result.source == "stale":
                    stage.degraded = True
                    stage.message = "最新行情拉取失败，复用陈旧历史缓存"
                    run.note(stage.message)
                if fetch_result.errors:
                    stage.degraded = True
                    stage.extra["errors"] = fetch_result.errors
                    if fetch_result.source != "stale":
                        summary_text = "；".join(fetch_result.errors[:2])
                        stage.message = "行情拉取失败，已返回空数据占位：" + summary_text
                        run.note(stage.message)
            except Exception as exc:  # noqa: BLE001
                stage.degraded = True
                stage.message = f"fetch 阶段异常：{exc}"
                run.note(f"fetch 失败：{exc}")

        fetch_result = result.fetch_result
        history = fetch_result.history if fetch_result is not None else None
        quote = fetch_result.quote if fetch_result is not None else {}
        if history is None:
            from .fetch import empty_ohlcv

            history = empty_ohlcv()

        # ---- indicators（指纹复用）----
        ind_fingerprint = dataframe_fingerprint("ind", history, period)
        with run.span("indicators", input_data=history) as stage:
            if not force_refresh and ind_fingerprint == self._indicator_fingerprint:
                indicator_frame = self._indicator_frame.copy()
                stage.reused = True
                stage.message = "输入指纹未变，复用上一次指标结果"
            else:
                indicator_frame = add_indicators(history)
                self._indicator_fingerprint = ind_fingerprint
                self._indicator_frame = indicator_frame.copy()
            result.indicator_frame = indicator_frame
            stage.set_output(indicator_frame)

        perf = performance_metrics(indicator_frame)

        # ---- predict（预测器内部按指纹复用模型）----
        model_fingerprint = f"{FEATURE_VERSION}|{ind_fingerprint}"
        with run.span("predict", input_data=indicator_frame) as stage:
            try:
                model_result = self.predictor.train(
                    indicator_frame, model_fingerprint, force_refresh=force_refresh
                )
                result.model_result = model_result
                if model_result.status != "ok":
                    stage.degraded = True
                    stage.message = model_result.message
                    run.note(f"ML 降级：{model_result.message}")
                stage.extra["n_samples"] = model_result.n_samples
                stage.reused = self.predictor.last_reused
            except Exception as exc:  # noqa: BLE001
                stage.degraded = True
                stage.message = f"ML 阶段异常，已跳过：{exc}"
                run.note(stage.message)

        # ---- ai（键含指纹，内容未变命中缓存；失败走本地/陈旧降级）----
        ai_fingerprint = dataframe_fingerprint("ai", indicator_frame, period)
        with run.span("ai", input_data=indicator_frame) as stage:
            try:
                ai_result = self.analyzer.analyze(
                    ticker,
                    indicator_frame,
                    quote,
                    ai_fingerprint,
                    force_refresh=force_refresh,
                )
                result.ai_result = ai_result
                stage.extra["source"] = ai_result.source
                stage.reused = ai_result.source == "cache"
                stage.degraded = ai_result.degraded
                stage.message = ai_result.message
                if ai_result.degraded:
                    run.note(f"AI 降级：{ai_result.message}")
            except Exception as exc:  # noqa: BLE001
                stage.degraded = True
                stage.message = f"AI 阶段异常，已占位：{exc}"
                run.note(stage.message)

        # ---- render（每次刷新都重算；内部单标签异常不影响整页）----
        with run.span("render", input_data=indicator_frame) as stage:
            try:
                view = render_dashboard(
                    ticker=ticker,
                    period=period,
                    history=history,
                    indicator_frame=indicator_frame,
                    quote=quote,
                    model_result=result.model_result,
                    ai_result=result.ai_result,
                    performance=perf,
                )
                result.view = view
                placeholder_tabs = [tab.title for tab in view.tabs if tab.is_placeholder]
                if placeholder_tabs:
                    stage.degraded = True
                    stage.message = "占位标签页：" + "、".join(placeholder_tabs)
                stage.extra["tabs"] = [tab.key for tab in view.tabs]
            except Exception as exc:  # noqa: BLE001
                stage.degraded = True
                stage.message = f"渲染异常：{exc}"
                run.note(stage.message)

        result.degraded = any(s.degraded for s in run.stages)
        result.errors = list(run.notes)
        run.finish("degraded" if result.degraded else "ok")
        self.store.save(run)
        return result
