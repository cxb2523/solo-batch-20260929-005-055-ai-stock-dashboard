"""Dataflow package: fetch -> indicators -> predict -> ai -> render.

每个阶段暴露一个 ``run`` 入口，接收一个可选的 ``tracer``，
通过 ``with tracer.step(name, inputs=...)`` 留下统一的 trace 钩子，
记录输入列名、时区、行数、耗时与异常。
"""

from .config import Config
from .trace import StageTrace, Tracer, describe_value
from .cache import QuoteCache, FingerprintMemo, interval_key, quote_key, content_fingerprint
from .pipeline import PipelineResult, run_pipeline

__all__ = [
    "Config",
    "StageTrace",
    "Tracer",
    "describe_value",
    "QuoteCache",
    "FingerprintMemo",
    "interval_key",
    "quote_key",
    "content_fingerprint",
    "PipelineResult",
    "run_pipeline",
]
