"""ai 阶段：可选的 LLM 市场分析 + 永不缺席的本地规则降级。

取舍 2 的落点：
- 没有 API 密钥（OPENAI_API_KEY / DATAFLOW_AI_API_KEY）：不请求网络，
  标签页显示“本地规则分析（占位）”，不跳过、整页正常；
- 有密钥但请求失败：优先返回 7 天内的陈旧 AI 缓存；再不行退回本地占位；
- 本地规则对 NaN 指标全部安全，空数据/单行也只产出一条提示。
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config
from .cache import TimedCache


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not (
        value is None or (isinstance(value, float) and math.isnan(value))
    )


@dataclass
class AIResult:
    source: str  # llm | local | cache
    insights: list[str] = field(default_factory=list)
    degraded: bool = False
    message: str = ""
    model: str = ""


def _get(row: pd.Series, key: str, default: float = float("nan")) -> float:
    try:
        value = row[key]
    except (KeyError, TypeError):
        return default
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def local_market_analysis(symbol: str, data: pd.DataFrame, info: dict) -> list[str]:
    """原始 generate_market_analysis 的 NaN-safe 版本。"""
    insights: list[str] = []
    info = info or {}
    if data is None or len(data) == 0:
        return [f"📭 {symbol} 暂无可分析的行情数据"]

    latest = data.iloc[-1]
    close = _get(latest, "Close")

    if len(data) >= 2 and _is_number(close):
        prev_close = _get(data.iloc[-2], "Close")
        if _is_number(prev_close) and prev_close != 0:
            change_pct = (close - prev_close) / prev_close * 100
            if change_pct > 3:
                insights.append(
                    f"🚀 {symbol} 出现强劲上涨动能，单日涨幅 {change_pct:.2f}%"
                )
            elif change_pct > 1:
                insights.append(f"🟢 {symbol} 明显上行（+{change_pct:.2f}%）")
            elif change_pct > 0:
                insights.append(f"🟡 {symbol} 小幅收涨（+{change_pct:.2f}%）")
            elif change_pct > -1:
                insights.append(f"🟡 {symbol} 小幅回落（{change_pct:.2f}%）")
            elif change_pct > -3:
                insights.append(f"🔴 {symbol} 承压下跌（{change_pct:.2f}%）")
            else:
                insights.append(f"🔻 {symbol} 遭遇明显抛售（{change_pct:.2f}%）")

    rsi = _get(latest, "RSI")
    if _is_number(rsi):
        if rsi > 80:
            insights.append(f"🚨 RSI {rsi:.1f} 严重超买，警惕回落")
        elif rsi > 70:
            insights.append(f"⚠️ RSI {rsi:.1f} 进入超买区，注意风险")
        elif rsi < 20:
            insights.append(f"🛒 RSI {rsi:.1f} 严重超卖，或存反弹机会")
        elif rsi < 30:
            insights.append(f"💡 RSI {rsi:.1f} 处于超卖区，关注企稳")
        elif 40 <= rsi <= 60:
            insights.append(f"⚖️ RSI {rsi:.1f}，多空动能均衡")
        else:
            bias = "偏多" if rsi > 50 else "偏空"
            insights.append(f"📊 RSI {rsi:.1f}，动能{bias}")

    sma20 = _get(latest, "SMA_20")
    sma50 = _get(latest, "SMA_50")
    if _is_number(close) and _is_number(sma20) and _is_number(sma50):
        if close > sma20 > sma50:
            insights.append("📈 多头排列：价格位于 20/50 日均线上方")
        elif close < sma20 < sma50:
            insights.append("📉 空头排列：价格位于关键均线下方")
        else:
            insights.append("➡️ 均线方向不一致，或处于震荡阶段")

    bb_upper = _get(latest, "BB_upper")
    bb_lower = _get(latest, "BB_lower")
    if _is_number(close) and _is_number(bb_upper) and close > bb_upper:
        insights.append("📊 价格突破布林上轨，短线或过热")
    if _is_number(close) and _is_number(bb_lower) and close < bb_lower:
        insights.append("📊 价格跌破布林下轨，或存超卖反弹")

    macd = _get(latest, "MACD")
    macd_signal = _get(latest, "MACD_signal")
    if _is_number(macd) and _is_number(macd_signal):
        if macd > macd_signal and macd > 0:
            insights.append("⚡ MACD 多头动能较强")
        elif macd < macd_signal and macd < 0:
            insights.append("⚡ MACD 呈空头动能")
        elif macd > macd_signal:
            insights.append("⚡ MACD 金叉，动能改善")
        else:
            insights.append("⚡ MACD 死叉，动能走弱")

    if "Volume" in data.columns and len(data) >= 20:
        avg_volume = data["Volume"].tail(20).mean()
        volume = _get(latest, "Volume")
        if _is_number(avg_volume) and avg_volume > 0 and _is_number(volume):
            ratio = volume / avg_volume
            if ratio > 2:
                insights.append("🔥 成交量异常放大，参与度高")
            elif ratio > 1.5:
                insights.append("📊 放量配合价格变化")
            elif ratio < 0.5:
                insights.append("📊 成交明显缩量，共识不强")

    market_cap = info.get("marketCap")
    if _is_number(market_cap):
        if market_cap > 200e9:
            insights.append("🏢 大盘蓝筹，波动通常较低")
        elif market_cap > 10e9:
            insights.append("🏢 中盘股，兼顾成长与稳定")
        else:
            insights.append("🏢 小盘股，成长与波动均较高")

    if not insights:
        insights.append(f"📊 {symbol} 指标样本不足，暂无有效信号")
    return insights


class OpenAICompatibleClient:
    """OpenAI Chat Completions 兼容客户端（urllib，无第三方依赖）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key or os.getenv("DATAFLOW_AI_API_KEY") or os.getenv(
            "OPENAI_API_KEY"
        )
        self.base_url = (base_url or config.AI_BASE_URL).rstrip("/")
        self.model = model or config.AI_MODEL
        self.timeout = timeout if timeout is not None else config.AI_HTTP_TIMEOUT

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def analyze(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("missing AI API key")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是一名技术分析助手。基于给定指标给出 4-7 条中文要点，"
                            "每条一行，以 emoji 开头，不要输出多余解释。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()


def build_prompt(symbol: str, data: pd.DataFrame, info: dict) -> str:
    """把最新指标压成短文本上下文；任何字段缺失都用 N/A。"""
    if data is None or len(data) == 0:
        return f"股票 {symbol} 当前没有可用行情数据。"
    latest = data.iloc[-1]
    fields = ["Close", "RSI", "MACD", "MACD_signal", "SMA_20", "SMA_50", "BB_upper", "BB_lower", "ATR"]
    lines = [f"股票代码：{symbol}，样本行数：{len(data)}"]
    for name in fields:
        value = _get(latest, name)
        lines.append(f"{name}: {value:.4f}" if _is_number(value) else f"{name}: N/A")
    if info:
        lines.append(
            "公司信息: "
            + ", ".join(
                f"{k}={info[k]}"
                for k in ("longName", "sector", "industry", "marketCap")
                if info.get(k) is not None
            )
        )
    lines.append("请给出市场走势、超买超卖、均线、量能方面的分析要点。")
    return "\n".join(lines)


class AIAnalyzer:
    """带新鲜/陈旧缓存与本地降级的 AI 阶段。"""

    def __init__(
        self,
        client: Any = None,
        ttl: Optional[float] = None,
        stale_ttl: Optional[float] = None,
        clock=None,
    ):
        self.client = client if client is not None else OpenAICompatibleClient()
        kwargs = {"clock": clock} if clock is not None else {}
        self._cache = TimedCache(
            ttl if ttl is not None else config.AI_TTL,
            stale_ttl if stale_ttl is not None else config.AI_STALE_TTL,
            name="ai",
            **kwargs,
        )

    def analyze(
        self,
        symbol: str,
        data: pd.DataFrame,
        info: dict,
        fingerprint: str,
        force_refresh: bool = False,
    ) -> AIResult:
        key = f"AI|{symbol.upper()}|{fingerprint}"
        cached = self._cache.get(key)
        if cached is not None and not force_refresh:
            cached.source = "cache"
            return cached

        local_insights = local_market_analysis(symbol, data, info)

        if not getattr(self.client, "available", False):
            result = AIResult(
                source="local",
                insights=local_insights,
                degraded=True,
                message="未配置 AI API 密钥，展示本地规则分析（占位）",
            )
            self._cache.put(key, result)
            return result

        try:
            content = self.client.analyze(build_prompt(symbol, data, info))
            insights = [line.strip() for line in content.splitlines() if line.strip()]
            result = AIResult(
                source="llm",
                insights=insights or local_insights,
                model=getattr(self.client, "model", ""),
            )
            self._cache.put(key, result)
            return result
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            stale = self._cache.latest_stale_with_prefix(f"AI|{symbol.upper()}|")
            if stale is not None and isinstance(stale, AIResult):
                return AIResult(
                    source="cache",
                    insights=stale.insights,
                    degraded=True,
                    message=f"AI 请求失败（{exc}），已复用旧缓存分析",
                    model=stale.model,
                )
            return AIResult(
                source="local",
                insights=local_insights,
                degraded=True,
                message=f"AI 请求失败（{exc}），展示本地规则分析（占位）",
            )
