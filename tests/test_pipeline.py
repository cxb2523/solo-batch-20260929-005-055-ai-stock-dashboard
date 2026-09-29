from dataflow.ai import AIAnalyzer
from dataflow.fetch import Fetcher
from dataflow.pipeline import Pipeline
from dataflow.predict import PricePredictor
from dataflow.trace import TraceStore
from .helpers import FakeYfClient


def _pipeline(rows=260, empty_tickers=(), fail_quote=False, min_samples=60):
    client = FakeYfClient(rows=rows, empty_tickers=empty_tickers)
    client.fail_quote = fail_quote
    return Pipeline(
        fetcher=Fetcher(client=client),
        predictor=PricePredictor(min_samples=min_samples),
        analyzer=AIAnalyzer(),  # 无密钥 -> 本地占位，测试不触网
        store=TraceStore(),
    )


def test_full_pipeline_renders_and_records_all_stage_traces():
    pipe = _pipeline(rows=120)
    result = pipe.run("AAPL", "1y")
    assert result.view is not None
    stages = [s.stage for s in result.run.stages]
    assert stages == ["fetch", "indicators", "predict", "ai", "render"]
    for stage in result.run.stages:
        assert stage.duration_ms >= 0
        assert stage.status == "ok"
    # 输入列/行数被 trace 钩子记录
    fetch_trace = result.run.stages[0]
    assert fetch_trace.output_rows == 120
    assert "Close" in fetch_trace.output_columns
    assert fetch_trace.output_timezone == "America/New_York"
    # AI 无密钥 -> 明确降级提示
    ai_stage = result.run.stages[3]
    assert ai_stage.degraded
    assert "API 密钥" in ai_stage.message
    assert result.view.tabs[3].is_placeholder is False


def test_refresh_reuses_indicators_and_predictions():
    pipe = _pipeline(rows=120)
    pipe.run("AAPL", "1y")
    train_calls = pipe.predictor.train_calls

    second = pipe.run("AAPL", "1y")
    ind_stage = second.run.stages[1]
    pred_stage = second.run.stages[2]
    assert ind_stage.reused is True
    assert pred_stage.reused is True
    assert pipe.predictor.train_calls == train_calls
    # fetch 走缓存命中
    assert second.fetch_result.source == "hit"

    # render 每次都重算，从不标复用
    assert second.run.stages[4].reused is False


def test_force_refresh_recomputes_indicators_and_model():
    pipe = _pipeline(rows=120)
    pipe.run("AAPL", "1y")
    result = pipe.run("AAPL", "1y", force_refresh=True)
    assert result.fetch_result.source == "bypass"
    assert result.run.stages[1].reused is False
    assert result.run.stages[2].reused is False


def test_empty_ticker_degrades_without_page_crash():
    pipe = _pipeline(rows=120, empty_tickers=("GHOST",))
    result = pipe.run("GHOST", "1y")
    # 整页依然有 view，五个标签都在
    assert result.view is not None
    assert len(result.view.tabs) == 5
    # 价格/技术/ML 标签退化为占位
    placeholders = {tab.key for tab in result.view.tabs if tab.is_placeholder}
    assert {"price", "technical", "prediction"}.issubset(placeholders)
    # AI 标签本地降级但仍有内容
    ai_tab = next(tab for tab in result.view.tabs if tab.key == "ai")
    assert not ai_tab.is_placeholder and ai_tab.payload["degraded"] is True
    assert result.degraded is True
    assert result.run.status == "degraded"


def test_single_row_history_does_not_crash():
    pipe = _pipeline(rows=1)
    result = pipe.run("TINY", "1y")
    assert result.view is not None
    assert result.view.data_range["rows"] == 1
    assert result.run.status == "degraded"  # ML 样本不足导致降级
    assert result.model_result.status == "insufficient"


def test_ticker_switch_changes_trace_history_and_data():
    pipe = _pipeline(rows=120)
    pipe.run("AAPL", "1y")
    pipe.run("MSFT", "1y")
    aapl_runs = pipe.store.history("AAPL")
    msft_runs = pipe.store.history("MSFT")
    assert len(aapl_runs) == 1 and len(msft_runs) == 1
    assert aapl_runs[0].ticker != msft_runs[0].ticker


def test_quote_failure_still_renders_with_company_placeholder():
    pipe = _pipeline(rows=120, fail_quote=True)
    result = pipe.run("AAPL", "1y")
    company_tab = next(tab for tab in result.view.tabs if tab.key == "company")
    assert company_tab.is_placeholder
    assert "公司信息" in company_tab.placeholder
    # 价格/技术标签不受 quote 失败影响
    price_tab = next(tab for tab in result.view.tabs if tab.key == "price")
    assert not price_tab.is_placeholder
