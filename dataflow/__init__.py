"""dataflow —— 股票仪表盘的五段式数据流。

阶段划分（每段一个 trace 钩子）::

    fetch -> indicators -> predict -> ai -> render

三个工程取舍在代码中的落点：

1. 缓存粒度（fetch.py / cache.py）：缓存键以 **ticker + 显式日期区间** 为粒度，
   period(1y/...) 在请求时换算成 start/end；另有独立的短 TTL 最新报价缓存。
   新鲜 TTL 内直接命中缓存（含最新收盘价），过期后尝试拉新、失败再回退陈旧缓存。
2. 降级策略（ai.py / pipeline.py / render.py）：缺 API 密钥时 AI 标签页显示
   本地规则占位分析；请求失败时优先复用旧缓存，失败再占位。任何单段异常都被
   pipeline 捕获并转成标签页占位，整页不崩。
3. 刷新复用（pipeline.py）：刷新时 fetch 按 TTL/force_refresh 决定是否重取，
   indicators / predict / ai 按输入数据指纹复用，render 永远重算；
   指标在样本不足或除零时一律返回 NaN，不抛异常。
"""

from .trace import StageTrace, RunTrace, TraceStore, trace_store

__all__ = ["StageTrace", "RunTrace", "TraceStore", "trace_store"]
