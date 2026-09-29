"""全局配置。

取舍说明（任务要求落进代码的三个取舍，见各模块内注释）：

1. 缓存键粒度 —— 以 ``ticker + 归一化后的 UTC 日期区间`` 为粒度
   （``dataflow/cache.py`` 的 ``quote_key``）。不使用 yfinance 的
   ``period`` 关键字，因为 "1y" 是滑动窗口，今天和明天指向不同区间，
   直接拿 period 做键会把不同日期区间的行情当成一份缓存。
2. 命中 vs 最新 —— 新鲜 TTL 内命中即复用；过期后先取新，失败再回退
   旧缓存（stale-if-error），并在 trace/页面上明确标注。
3. 刷新时复用 —— 行情按 TTL 复用；指标/预测/AI 按输入内容指纹 memo
   复用（同一份数据刷新不重训模型、不重调 AI）；渲染每次重算。
   指标样本不足或除零一律返回 NaN，绝不抛异常。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # 行情缓存：交易时段内 N 秒内的命中视为新鲜，直接复用不联网
    quote_ttl_seconds: float = 60.0
    # 磁盘缓存路径，None 表示只用进程内缓存（测试用）
    cache_path: str | None = ".cache/quotes.pkl"
    # 真实拉取失败、且连旧缓存都没有时，是否使用内置合成行情
    # （仅用于离线演示 trace_app，页面会显著标注 demo 降级）
    allow_demo_fallback: bool = True

    # AI（OpenAI 兼容协议）。缺密钥时根本不发请求，走本地规则分析
    ai_base_url: str | None = None
    ai_api_key: str | None = None
    ai_model: str = "gpt-4o-mini"
    ai_timeout_seconds: float = 8.0

    # ML 最少样本数；不足时 predict 阶段返回 NaN/占位，不训练
    min_train_rows: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        cache_path = os.environ.get("STOCK_CACHE_PATH", ".cache/quotes.pkl")
        return cls(
            quote_ttl_seconds=float(os.environ.get("STOCK_QUOTE_TTL", "60")),
            cache_path=cache_path or None,
            allow_demo_fallback=os.environ.get("STOCK_DEMO_FALLBACK", "1") != "0",
            ai_base_url=os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("AI_BASE_URL"),
            ai_api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_API_KEY"),
            ai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            ai_timeout_seconds=float(os.environ.get("AI_TIMEOUT", "8")),
            min_train_rows=int(os.environ.get("ML_MIN_TRAIN_ROWS", "60")),
        )
