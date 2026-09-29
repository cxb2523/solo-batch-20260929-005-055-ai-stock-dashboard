import pytest

from conftest import make_ohlcv
from dataflow import Config, FingerprintMemo, QuoteCache, run_pipeline


def test_pipeline_full_end_to_end_records_five_stage_traces(fake_fetcher):
    fetcher, _ = fake_fetcher
    cfg = Config(cache_path=None, ai_api_key=None)
    result = run_pipeline("AAPL", period="1y", config=cfg, fetcher=fetcher)
    stages = [r.stage for r in result.tracer.records]
    assert stages == ["fetch", "indicators", "predict", "ai", "render"]
    for rec in result.tracer.records:
        assert rec.elapsed_ms is not None and rec.elapsed_ms >= 0
    assert result.fetch.source == "network"
    assert len(result.indicators_df) == 250
    assert result.render.tabs and len(result.render.tabs) == 6
    fetch_rec = result.tracer.get("fetch")
    assert fetch_rec.output["source"] == "network"


def test_refresh_reuses_quote_and_memo_but_render_always_runs(fake_fetcher):
    fetcher, counters = fake_fetcher
    cache = QuoteCache(ttl_seconds=3600, path=None)
    memo = FingerprintMemo()
    cfg = Config(cache_path=None, ai_api_key=None)
    r1 = run_pipeline("AAPL", config=cfg, cache=cache, memo=memo, fetcher=fetcher)
    r2 = run_pipeline("AAPL", config=cfg, cache=cache, memo=memo, fetcher=fetcher)
    assert counters["calls"] == 1
    assert cache.stats["fresh_hits"] == 1
    # 指标/预测/AI 三次 memo 命中
    assert memo.stats["hits"] == 3
    # render 每次重算：两条 trace 都存在
    assert r1.tracer.get("render").ok and r2.tracer.get("render").ok


def test_switching_ticker_changes_data_and_misses_cache(fake_fetcher):
    fetcher, counters = fake_fetcher
    cache = QuoteCache(ttl_seconds=3600, path=None)
    memo = FingerprintMemo()
    cfg = Config(cache_path=None, ai_api_key=None)
    ra = run_pipeline("AAPL", config=cfg, cache=cache, memo=memo, fetcher=fetcher)
    rb = run_pipeline("MSFT", config=cfg, cache=cache, memo=memo, fetcher=fetcher)
    assert counters["calls"] == 2
    assert ra.ticker != rb.ticker
    assert ra.indicators_df["Close"].iloc[-1] != rb.indicators_df["Close"].iloc[-1]
    assert cache.stats["fresh_hits"] == 0


def test_network_failure_whole_page_does_not_crash(fake_fetcher):
    fetcher, counters = fake_fetcher
    counters["fail_with"] = RuntimeError("network down")
    cfg = Config(cache_path=None, allow_demo_fallback=False, ai_api_key=None)
    result = run_pipeline("AAPL", config=cfg, fetcher=fetcher)
    assert result.fetch.source == "empty"
    assert any("无法获取" in b or "异常" in b for b in result.banners)
    # 五个阶段都有 trace，且全部记录耗时
    assert len(result.tracer.records) == 5
    assert all(r.elapsed_ms is not None for r in result.tracer.records)
    # 标签页仍被渲染（降级/错误态）
    assert len(result.render.tabs) == 6


def test_ai_remote_failure_does_not_crash_page(fake_fetcher):
    fetcher, _ = fake_fetcher
    cfg = Config(cache_path=None, ai_api_key="sk-test")
    def bad_provider(system, user, config):
        raise RuntimeError("500 server")
    result = run_pipeline("AAPL", config=cfg, fetcher=fetcher, ai_provider=bad_provider)
    ai_rec = result.tracer.get("ai")
    assert ai_rec.status == "degraded"
    assert result.ai.source == "local"
    assert any("AI 请求失败" in b for b in result.banners)


def test_trace_payload_is_json_serializable(fake_fetcher):
    import json
    fetcher, _ = fake_fetcher
    cfg = Config(cache_path=None, allow_demo_fallback=True, ai_api_key=None)
    counters = {"n": 0}
    def flaky(symbol):
        counters["n"] += 1
        if counters["n"] == 1:
            raise ConnectionError("offline first")
        return fetcher(symbol)
    # 第一次走 demo
    r1 = run_pipeline("TSLA", config=cfg, fetcher=flaky)
    assert r1.fetch.source == "demo"
    payload = r1.to_payload()
    json.dumps(payload)  # 不含 NaN/不可序列化对象


def test_flask_app_switches_tickers_and_reports_trace(fake_fetcher):
    from trace_app import create_app
    fetcher, counters = fake_fetcher
    cfg = Config(cache_path=None, ai_api_key=None)
    app = create_app(cfg, fetcher=fetcher)
    client = app.test_client()

    resp_a = client.get("/api/trace?ticker=AAPL&period=1y")
    assert resp_a.status_code == 200
    data_a = resp_a.get_json()
    assert data_a["ticker"] == "AAPL"
    assert [s["stage"] for s in data_a["stages"]] == [
        "fetch", "indicators", "predict", "ai", "render"]
    assert all(s["elapsed_ms"] is not None for s in data_a["stages"])

    resp_b = client.get("/api/trace?ticker=MSFT&period=1y")
    data_b = resp_b.get_json()
    assert data_b["ticker"] == "MSFT"
    assert counters["calls"] == 2
    # 两次阶段耗时各自存在（不强制严格不等，但输入行数/来源反映切换）
    assert data_a["stages"][0]["output"]["source"] == "network"
    assert data_b["stages"][0]["output"]["source"] == "network"

    # 再刷 AAPL：新鲜缓存命中
    resp_a2 = client.get("/api/trace?ticker=AAPL&period=1y")
    data_a2 = resp_a2.get_json()
    assert data_a2["quote_source"] == "cache"
    assert data_a2["cache_stats"]["fresh_hits"] == 1


def test_flask_index_renders_form():
    from trace_app import create_app
    cfg = Config(cache_path=None)
    app = create_app(cfg, fetcher=lambda s: (make_ohlcv(100), {}))
    page = app.test_client().get("/?ticker=NVDA&period=6mo")
    assert page.status_code == 200
    assert b"NVDA" in page.data and b"Dataflow Trace" in page.data


def test_flask_app_blank_ticker_demo_mode_still_200():
    from trace_app import create_app
    cfg = Config(cache_path=None, allow_demo_fallback=True, ai_api_key=None)
    app = create_app(cfg, fetcher=lambda s: (_ for _ in ()).throw(ConnectionError("x")))
    resp = app.test_client().get("/api/run?ticker=&period=1y")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["quote_source"] == "empty"
    assert len(data["tabs"]) == 6
