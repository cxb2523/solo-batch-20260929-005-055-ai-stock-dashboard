from dataflow.ai import AIAnalyzer, local_market_analysis
from dataflow.fetch import empty_ohlcv
from dataflow.indicators import add_indicators
from .helpers import synthetic_ohlcv


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class MissingKeyClient:
    available = False
    model = "none"

    def analyze(self, prompt):
        raise AssertionError("没有密钥时不应发起请求")


class BrokenClient:
    available = True
    model = "gpt-test"

    def __init__(self):
        self.calls = 0

    def analyze(self, prompt):
        self.calls += 1
        raise ConnectionError("boom")


def test_local_analysis_handles_empty_and_single_row():
    empty_insights = local_market_analysis("AAPL", empty_ohlcv(), {})
    assert len(empty_insights) == 1 and "暂无可分析" in empty_insights[0]

    one = add_indicators(synthetic_ohlcv(rows=1))
    insights = local_market_analysis("AAPL", one, {})
    assert insights  # 不抛异常且至少一条


def test_missing_api_key_uses_local_placeholder():
    analyzer = AIAnalyzer(client=MissingKeyClient())
    frame = add_indicators(synthetic_ohlcv(rows=40))
    result = analyzer.analyze("AAPL", frame, {}, "fp1")
    assert result.source == "local"
    assert result.degraded is True
    assert "API 密钥" in result.message
    assert result.insights


def test_request_failure_falls_back_to_local_then_stale_cache():
    clock = Clock()
    client = BrokenClient()
    analyzer = AIAnalyzer(
        client=client,
        ttl=10.0,
        stale_ttl=100.0,
        clock=clock,
    )
    frame = add_indicators(synthetic_ohlcv(rows=40))

    # 第一次失败：没有旧缓存 -> 本地占位
    first = analyzer.analyze("AAPL", frame, {}, "fp-new")
    assert first.source == "local"
    assert first.degraded

    # 先制造一份“旧指纹缓存”
    from dataflow.ai import AIResult

    analyzer._cache.put(
        "AI|AAPL|fp-old",
        AIResult(source="llm", insights=["旧的云端结论"], model="gpt-test"),
    )
    clock.now += 50  # 进入陈旧窗口但未超宽限

    second = analyzer.analyze("AAPL", frame, {}, "fp-new2", force_refresh=True)
    assert second.source == "cache"
    assert second.degraded is True
    assert "复用旧缓存" in second.message
    assert second.insights == ["旧的云端结论"]

    # 超过陈旧宽限后再失败 -> 重新退化为本地占位
    clock.now += 200
    third = analyzer.analyze("AAPL", frame, {}, "fp-new3", force_refresh=True)
    assert third.source == "local"
    assert client.calls >= 2
