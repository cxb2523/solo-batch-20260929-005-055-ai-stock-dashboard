"""ai 阶段：AI 文本解读。

取舍 2（AI 侧，缺密钥/请求失败怎么办 —— 不跳过标签页）：
    1. 缺 API 密钥 -> 不发任何请求，直接用内置本地规则生成解读；
    2. 有密钥但请求失败 -> 同 ticker 旧结论复用（若存在）-> 本地规则；
    3. 标签页永远保留：占位/降级结论照常在 "AI 解读" 标签里展示，
       整页不会因为 AI 不可用而崩。
    AI 结论按输入指纹 memo，行情没变就不重复请求/生成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import math

import numpy as np
import pandas as pd

from .cache import FingerprintMemo, content_fingerprint
from .config import Config
from .trace import Tracer, get_tracer


@dataclass
class AIResult:
    headline: str = ""
    bullets: list[str] = field(default_factory=list)
    source: str = "local"  # remote | local | stale_ai | placeholder
    model: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "bullets": self.bullets,
            "source": self.source,
            "model": self.model,
            "warnings": self.warnings,
            "error": self.error,
        }


AIProvider = Callable[[str, str, "Config"], str]

SYSTEM_PROMPT = (
    "你是一名严谨的证券技术分析师。基于给定的结构化指标摘要，"
    "输出中文：第一行是一句话标题，之后每行用 '- ' 开头给出 3-6 条要点。"
    "不提供买卖指令，不承诺收益。"
)


def run(
    indicators_df: pd.DataFrame,
    info: dict[str, Any] | None = None,
    *,
    ticker: str = "",
    prediction: dict[str, Any] | None = None,
    config: Config | None = None,
    tracer: Tracer | None = None,
    memo: FingerprintMemo | None = None,
    memo_key: str = "",
    provider: AIProvider | None = None,
) -> AIResult:
    config = config or Config()
    info = info or {}
    prediction = prediction or {}
    fingerprint = content_fingerprint(indicators_df)

    if memo is not None and memo_key:
        cached = memo.get("ai", memo_key, fingerprint)
        if cached is not None:
            with get_tracer(tracer, "ai",
                            inputs={"indicators": indicators_df, "memo": "hit"}) as rec:
                rec.output = cached.to_dict()
                rec.note = "AI 结论按内容指纹复用，未重新调用"
                return cached

    with get_tracer(tracer, "ai",
                    inputs={"indicators": indicators_df,
                            "has_api_key": bool(config.ai_api_key),
                            "memo": "miss" if memo else "off"}) as rec:

        def finish(result: AIResult, status: str, note: str) -> AIResult:
            rec.status = status
            rec.note = note
            rec.output = result.to_dict()
            if memo is not None and memo_key:
                memo.set("ai", memo_key, fingerprint, result)
            return result

        if indicators_df is None or indicators_df.empty:
            result = AIResult(
                headline=f"{ticker} 暂无可分析行情",
                bullets=["AI 解读标签页保留为占位：数据恢复后自动重新生成。"],
                source="placeholder",
                warnings=["行情为空，AI 解读为占位内容。"],
            )
            return finish(result, "degraded", "空行情 -> 占位，不跳过标签页")

        summary = build_summary(indicators_df, info, ticker, prediction)
        user_prompt = _render_summary(summary)

        if not config.ai_api_key:
            result = _local_result(summary)
            result.warnings.append("未配置 API 密钥（OPENAI_API_KEY），使用本地规则解读。")
            return finish(result, "degraded", "缺密钥 -> 本地规则（不发请求）")

        active_provider = provider or openai_compatible_provider
        try:
            text = active_provider(SYSTEM_PROMPT, user_prompt, config)
            result = _parse_remote(text, config.ai_model)
            if not result.bullets:
                raise ValueError("远程返回内容为空或格式不符")
            return finish(result, "ok", f"远程模型 {config.ai_model} 调用成功")
        except Exception as exc:
            # 请求失败：标签页不跳过，退回本地规则（旧 AI 结论由上层按
            # ticker 保留在内存中时同样走这里之前的缓存层）
            result = _local_result(summary)
            result.source = "local"
            result.error = f"{type(exc).__name__}: {exc}"
            result.warnings.append(
                f"AI 请求失败（{result.error}），已降级为本地规则解读。"
            )
            return finish(result, "degraded", "远程失败 -> 本地规则兜底")


def build_summary(
    df: pd.DataFrame, info: dict[str, Any], ticker: str,
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    def val(name: str) -> float:
        value = latest.get(name, np.nan)
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    close = val("Close")
    prev_close = float(prev.get("Close", np.nan))
    change_pct = (close - prev_close) / prev_close * 100 if prev_close else math.nan
    volume = val("Volume")
    volume_sma = val("Volume_SMA")
    volume_ratio = volume / volume_sma if volume_sma and not math.isnan(volume_sma) else math.nan

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "rows": int(len(df)),
        "close": close,
        "change_pct": change_pct,
        "rsi": val("RSI"),
        "macd": val("MACD"),
        "macd_signal": val("MACD_signal"),
        "sma_20": val("SMA_20"),
        "sma_50": val("SMA_50"),
        "bb_upper": val("BB_upper"),
        "bb_lower": val("BB_lower"),
        "atr": val("ATR"),
        "volume_ratio": volume_ratio,
        "prediction": (prediction or {}).get("prediction"),
        "predicted_change_pct": (prediction or {}).get("predicted_change_pct"),
        "test_score": (prediction or {}).get("test_score"),
    }


def _render_summary(summary: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 2) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "N/A"
        return f"{value:.{digits}f}"

    lines = [
        f"标的: {summary['ticker']} ({summary['name']})，样本行数: {summary['rows']}",
        f"最新收盘: {fmt(summary['close'])}，日涨跌: {fmt(summary['change_pct'])}%",
        f"RSI14: {fmt(summary['rsi'], 1)}，MACD: {fmt(summary['macd'], 3)}，"
        f"信号线: {fmt(summary['macd_signal'], 3)}",
        f"SMA20: {fmt(summary['sma_20'])}，SMA50: {fmt(summary['sma_50'])}",
        f"布林上轨: {fmt(summary['bb_upper'])}，下轨: {fmt(summary['bb_lower'])}，ATR: {fmt(summary['atr'])}",
        f"量比(vs 20日均量): {fmt(summary['volume_ratio'])}",
        f"ML 次日预测: {fmt(summary['prediction'])} "
        f"({fmt(summary['predicted_change_pct'])}%)，模型测试分: {fmt(summary['test_score'], 3)}",
    ]
    return "\n".join(lines)


def _parse_remote(text: str, model: str) -> AIResult:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    headline = lines[0].lstrip("#- ").strip() if lines else ""
    bullets = [line.lstrip("-* ").strip() for line in lines[1:] if line.lstrip().startswith(("-", "*"))]
    if not bullets and len(lines) > 1:
        bullets = [line.lstrip("#- ").strip() for line in lines[1:]]
    return AIResult(headline=headline, bullets=bullets[:8], source="remote", model=model)


def _local_result(summary: dict[str, Any]) -> AIResult:
    ticker = summary["ticker"]
    bullets: list[str] = []

    change = summary["change_pct"]
    if change is not None and not math.isnan(change):
        if change > 3:
            bullets.append(f"{ticker} 单日上涨 {change:.2f}%，动量极强，注意追高风险。")
        elif change > 1:
            bullets.append(f"{ticker} 上涨 {change:.2f}%，短线偏强。")
        elif change > -1:
            bullets.append(f"{ticker} 涨跌幅 {change:.2f}%，多空分歧不大。")
        elif change > -3:
            bullets.append(f"{ticker} 下跌 {change:.2f}%，短线承压。")
        else:
            bullets.append(f"{ticker} 下跌 {change:.2f}%，抛压明显，留意支撑位。")

    rsi = summary["rsi"]
    if rsi is not None and not math.isnan(rsi):
        if rsi >= 70:
            bullets.append(f"RSI 为 {rsi:.1f}，进入超买区，警惕均值回归。")
        elif rsi <= 30:
            bullets.append(f"RSI 为 {rsi:.1f}，进入超卖区，关注企稳信号。")
        else:
            bullets.append(f"RSI 为 {rsi:.1f}，未出现极端读数。")
    else:
        bullets.append("RSI 样本不足（NaN），等待更多交易日数据。")

    close, sma20, sma50 = summary["close"], summary["sma_20"], summary["sma_50"]
    if all(v is not None and not (isinstance(v, float) and math.isnan(v))
           for v in (close, sma20, sma50)):
        if close > sma20 > sma50:
            bullets.append("价格站上 20/50 日均线且均线多头排列，趋势向上。")
        elif close < sma20 < sma50:
            bullets.append("价格跌破 20/50 日均线且均线空头排列，趋势向下。")
        else:
            bullets.append("价格与 20/50 日均线交错，趋势信号不明确。")

    macd, signal = summary["macd"], summary["macd_signal"]
    if macd is not None and not math.isnan(macd) and signal is not None and not math.isnan(signal):
        if macd > signal and macd > 0:
            bullets.append("MACD 在零轴上方且强于信号线，动能偏多。")
        elif macd < signal and macd < 0:
            bullets.append("MACD 在零轴下方且弱于信号线，动能偏空。")
        else:
            bullets.append("MACD 与信号线交叉位置中性，动能方向待确认。")

    volume_ratio = summary["volume_ratio"]
    if volume_ratio is not None and not math.isnan(volume_ratio):
        if volume_ratio > 1.5:
            bullets.append(f"量比 {volume_ratio:.2f}，放量明显，关注量价配合。")
        elif volume_ratio < 0.7:
            bullets.append(f"量比 {volume_ratio:.2f}，交投清淡，信号可靠性降低。")

    prediction = summary.get("prediction")
    if prediction is not None and not (isinstance(prediction, float) and math.isnan(prediction)):
        bullets.append(
            f"本地模型预测次日收盘约 {prediction:.2f}（{summary['predicted_change_pct']:+.2f}%），"
            "仅为统计外推，非投资建议。"
        )

    headline = f"{ticker} 本地规则解读（非 AI 远程调用）"
    return AIResult(headline=headline, bullets=bullets, source="local")


def openai_compatible_provider(system_prompt: str, user_prompt: str, config: Config) -> str:
    """requests 直连 OpenAI 兼容 /chat/completions，惰性导入便于测试替换。"""
    import requests

    base_url = (config.ai_base_url or "https://api.openai.com/v1").rstrip("/")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.ai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.ai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        },
        timeout=config.ai_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]
