"""统一 trace 钩子。

每个阶段（fetch / indicators / predict / ai / render）用
``with tracer.step("fetch", inputs={"data": df, ...})`` 包裹，
自动记录：输入列名、时区、行数、耗时、异常，以及输出摘要。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def describe_value(value: Any, *, max_columns: int = 60) -> Any:
    """把任意输入转成可 JSON 序列化、体积可控的描述。"""
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return _describe_frame(value, max_columns=max_columns)
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "name": None if value.name is None else str(value.name),
            "rows": int(len(value)),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): describe_value(v, max_columns=max_columns) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe_value(v, max_columns=max_columns) for v in value]
    return repr(value)[:200]


def _describe_frame(df: pd.DataFrame, *, max_columns: int) -> dict[str, Any]:
    tz = None
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        tz = str(df.index.tz)
    index_range = None
    if len(df) and isinstance(df.index, pd.DatetimeIndex):
        index_range = [
            df.index.min().isoformat(),
            df.index.max().isoformat(),
        ]
    columns = [str(c) for c in df.columns][:max_columns]
    return {
        "type": "DataFrame",
        "rows": int(len(df)),
        "columns": columns,
        "n_columns": int(df.shape[1]),
        "index_dtype": str(df.index.dtype),
        "timezone": tz,
        "index_range": index_range,
    }


@dataclass
class StageTrace:
    stage: str
    started_at: float
    elapsed_ms: float | None = None
    inputs: Any = None
    output: Any = None
    ok: bool = False
    status: str = "ok"  # ok | degraded | error | skipped
    error: str | None = None
    note: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_ms": self.elapsed_ms,
            "inputs": _jsonable(describe_value(self.inputs)),
            "output": _jsonable(describe_value(self.output)),
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "note": self.note,
            "extras": _jsonable(describe_value(self.extras)),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


class _Step:
    """with 上下文：一个阶段一个 trace 记录。"""

    def __init__(self, tracer: "Tracer", stage: str, inputs: Any):
        self.tracer = tracer
        self.record = StageTrace(
            stage=stage, started_at=time.time(), inputs=describe_value(inputs)
        )

    def __enter__(self) -> StageTrace:
        return self.record

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record.elapsed_ms = round((time.time() - self.record.started_at) * 1000, 3)
        if exc is not None:
            # 阶段内部异常是数据流水线的一等公民：记录在 trace 上，
            # 是否吞掉由阶段/pipeline 自己决定（这里继续向上抛）。
            self.record.ok = False
            self.record.status = "error"
            self.record.error = f"{exc_type.__name__}: {exc}"
        else:
            self.record.ok = self.record.status != "error"
        self.tracer.records.append(self.record)
        return False


class _NullStep:
    def __enter__(self) -> StageTrace:  # pragma: no cover - 纯透传
        return StageTrace(stage="noop", started_at=time.time())

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover
        return False


class Tracer:
    def __init__(self, label: str = ""):
        self.label = label
        self.records: list[StageTrace] = []

    def step(self, stage: str, inputs: Any = None) -> _Step:
        return _Step(self, stage, describe_value(inputs))

    def get(self, stage: str) -> StageTrace | None:
        for record in reversed(self.records):
            if record.stage == stage:
                return record
        return None

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]


_NULL_STEP = _NullStep()


def get_tracer(tracer: "Tracer | None", stage: str, inputs: Any = None):
    """没有 tracer 时返回透传占位，阶段代码无需判空。"""
    if tracer is None:
        return _NULL_STEP
    return tracer.step(stage, inputs)
