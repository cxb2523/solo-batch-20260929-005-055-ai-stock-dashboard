"""运行期可调参数，全部支持环境变量覆盖，便于测试。"""
from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# 日线历史行情：新鲜缓存 5 分钟（命中即视为含最新行情），陈旧缓存最多保留 24 小时
HISTORY_TTL = _float("DATAFLOW_HISTORY_TTL", 300.0)
HISTORY_STALE_TTL = _float("DATAFLOW_HISTORY_STALE_TTL", 86400.0)

# 最新报价独立缓存 60 秒（不与历史区间共享 TTL）
QUOTE_TTL = _float("DATAFLOW_QUOTE_TTL", 60.0)
QUOTE_STALE_TTL = _float("DATAFLOW_QUOTE_STALE_TTL", 3600.0)

# AI 分析：新鲜缓存 30 分钟，失败时陈旧缓存最多保留 7 天
AI_TTL = _float("DATAFLOW_AI_TTL", 1800.0)
AI_STALE_TTL = _float("DATAFLOW_AI_STALE_TTL", 7 * 86400.0)
AI_HTTP_TIMEOUT = _float("DATAFLOW_AI_HTTP_TIMEOUT", 15.0)
AI_BASE_URL = os.getenv("DATAFLOW_AI_BASE_URL", "https://api.openai.com/v1")
AI_MODEL = os.getenv("DATAFLOW_AI_MODEL", "gpt-4o-mini")

# ML 训练所需的最少有效样本数（dropna 之后）
MIN_TRAIN_SAMPLES = int(_float("DATAFLOW_MIN_TRAIN_SAMPLES", 60))
