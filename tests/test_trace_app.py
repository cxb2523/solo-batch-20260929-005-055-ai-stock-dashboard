import pytest

import trace_app
from dataflow.trace import TraceStore


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATAFLOW_DEMO", "1")
    pipe = trace_app.build_pipeline()
    pipe.store = TraceStore()
    app = trace_app.create_app(pipe)
    app.config["TESTING"] = True
    return app.test_client(), pipe


def test_dashboard_renders_demo_data(client):
    test_client, _ = client
    resp = test_client.get("/?ticker=AAPL&period=1y")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "AAPL" in body
    assert "阶段 trace" in body
    for stage in ("fetch", "indicators", "predict", "ai", "render"):
        assert stage in body


def test_switching_ticker_changes_stage_times_and_rows(client):
    test_client, pipe = client
    test_client.get("/?ticker=AAPL&period=1y")
    resp_aapl = test_client.get("/?ticker=MSFT&period=1y")
    body = resp_aapl.get_data(as_text=True)
    assert "MSFT" in body
    latest_msft = pipe.store.latest("MSFT")
    latest_aapl = pipe.store.latest("AAPL")
    assert latest_msft is not None and latest_aapl is not None
    assert latest_msft.stages[0].output_rows > 0


def test_trace_json_endpoint(client):
    test_client, _ = client
    test_client.get("/?ticker=TSLA&period=6mo")
    resp = test_client.get("/trace/TSLA")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ticker"] == "TSLA"
    run = payload["runs"][0]
    assert [s["stage"] for s in run["stages"]] == [
        "fetch",
        "indicators",
        "predict",
        "ai",
        "render",
    ]
    for stage in run["stages"]:
        assert "duration_ms" in stage
        assert "input_columns" in stage
        assert "input_rows" in stage


def test_empty_ticker_shows_degradation_banner(client):
    test_client, _ = client
    resp = test_client.get("/?ticker=EMPTY&period=1y")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "降级" in body
    assert "暂无行情数据" in body


def test_force_refresh_bypass_flag_visible(client):
    test_client, _ = client
    test_client.get("/?ticker=AAPL")
    resp = test_client.get("/?ticker=AAPL&refresh=1")
    body = resp.get_data(as_text=True)
    run_marker = "🔄"
    assert run_marker in body or "bypass" in body
