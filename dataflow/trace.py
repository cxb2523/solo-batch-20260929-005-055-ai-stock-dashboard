"""阶段追踪钩子。

每个阶段用 ``with run.span("fetch", input_data=df)`` 包裹，
自动记录输入列名 / 时区 / 行数、输出概况、耗时与异常；TraceStore 按 ticker
保留最近若干次运行，供 trace_app.py 展示。
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import pandas as pd


def describe_frame(value: Any) -> dict:
    """返回 DataFrame 的列名、索引时区、行数；非 DataFrame 返回空骨架。"""
    if isinstance(value, pd.DataFrame):
        tz_name = None
        index = value.index
        if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
            tz_name = str(index.tz)
        return {
            "columns": [str(c) for c in value.columns],
            "rows": int(len(value)),
            "timezone": tz_name,
        }
    if isinstance(value, pd.Series):
        return {"columns": [str(value.name)], "rows": int(len(value)), "timezone": None}
    return {"columns": [], "rows": 0, "timezone": None}


@dataclass
class StageTrace:
    stage: str
    started_at: float
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error | skipped
    input_columns: list[str] = field(default_factory=list)
    input_timezone: Optional[str] = None
    input_rows: int = 0
    output_columns: list[str] = field(default_factory=list)
    output_timezone: Optional[str] = None
    output_rows: int = 0
    reused: bool = False
    degraded: bool = False
    message: str = ""
    exception: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def set_output(self, value: Any) -> None:
        info = describe_frame(value)
        self.output_columns = info["columns"]
        self.output_timezone = info["timezone"]
        self.output_rows = info["rows"]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunTrace:
    ticker: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    duration_ms: float = 0.0
    period: str = ""
    force_refresh: bool = False
    status: str = "running"  # running | ok | degraded | error
    stages: list[StageTrace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def span(self, stage: str, input_data: Any = None) -> "_StageSpan":
        return _StageSpan(self, stage, input_data)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def finish(self, status: str = "ok") -> None:
        self.finished_at = time.time()
        self.duration_ms = round((self.finished_at - self.started_at) * 1000, 3)
        self.status = status

    def to_dict(self) -> dict:
        return asdict(self)


class _StageSpan:
    """trace 钩子：进入时记输入，退出时记耗时；异常被记录但不吞掉。"""

    def __init__(self, run: RunTrace, stage: str, input_data: Any):
        info = describe_frame(input_data)
        self.trace = StageTrace(
            stage=stage,
            started_at=time.time(),
            input_columns=info["columns"],
            input_timezone=info["timezone"],
            input_rows=info["rows"],
        )
        self._perf_start = time.perf_counter()
        self._run = run

    def __enter__(self) -> StageTrace:
        self._run.stages.append(self.trace)
        return self.trace

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.trace.duration_ms = round((time.perf_counter() - self._perf_start) * 1000, 3)
        if exc is not None:
            self.trace.status = "error"
            self.trace.exception = "".join(
                traceback.format_exception_only(exc_type, exc)
            ).strip()
        return False  # 不吞异常，交给 pipeline 统一降级


class TraceStore:
    """按 ticker 保存最近 N 次运行的环形缓冲，线程安全。"""

    def __init__(self, max_per_ticker: int = 20):
        self._runs: dict[str, deque[RunTrace]] = defaultdict(
            lambda: deque(maxlen=max_per_ticker)
        )
        self._lock = threading.RLock()

    def start_run(self, ticker: str, period: str, force_refresh: bool) -> RunTrace:
        run = RunTrace(ticker=ticker, period=period, force_refresh=force_refresh)
        with self._lock:
            self._runs[ticker.upper()].append(run)
        return run

    def save(self, run: RunTrace) -> None:
        # start_run 已把同一可变对象放入 deque，无需再次入队（否则历史翻倍）
        """保留给外部显式落库语义；当前 deque 持有的是同一引用，无需操作。"""

    def latest(self, ticker: str) -> Optional[RunTrace]:
        with self._lock:
            runs = self._runs.get(ticker.upper())
            return runs[-1] if runs else None

    def history(self, ticker: str, limit: int = 10) -> list[RunTrace]:
        with self._lock:
            runs = self._runs.get(ticker.upper())
            if not runs:
                return []
            return list(runs)[-limit:][::-1]

    def tickers(self) -> list[str]:
        with self._lock:
            return sorted(self._runs.keys())

    def clear(self, ticker: str | None = None) -> None:
        with self._lock:
            if ticker is None:
                self._runs.clear()
            else:
                self._runs.pop(ticker.upper(), None)


trace_store = TraceStore()
