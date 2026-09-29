import pytest

from conftest import make_ohlcv
from dataflow import Config, ai, indicators
from dataflow.fetch import empty_quotes


def test_missing_api_key_uses_local_rules_without_request():
    cfg = Config(ai_api_key=None)
    called = {"n": 0}
    def provider(system, user, config):
        called["n"] += 1
        return "标题\n- 要点1\n- 要点2"
    df = indicators.run(make_ohlcv(100))
    r = ai.run(df, {}, ticker="AAPL", config=cfg, provider=provider)
    assert called["n"] == 0
    assert r.source == "local" and r.bullets
    assert any("API 密钥" in w for w in r.warnings)


def test_empty_data_keeps_tab_placeholder_not_skip():
    cfg = Config(ai_api_key=None)
    r = ai.run(empty_quotes(), {}, ticker="NOPE", config=cfg)
    assert r.source == "placeholder"
    assert "占位" in r.bullets[0]


def test_remote_failure_falls_back_to_local_and_records_error():
    cfg = Config(ai_api_key="sk-test", ai_model="m")
    def provider(system, user, config):
        raise TimeoutError("upstream timeout")
    df = indicators.run(make_ohlcv(80))
    r = ai.run(df, {}, ticker="AAPL", config=cfg, provider=provider)
    assert r.source == "local"
    assert r.error and "timeout" in r.error
    assert any("降级" in w for w in r.warnings)
    assert r.bullets  # 标签页仍有内容


def test_remote_success_parses_headline_and_bullets():
    cfg = Config(ai_api_key="sk-test", ai_model="m")
    def provider(system, user, config):
        return "AAPL 短线偏强\n- RSI 中性\n- 均线多头排列"
    df = indicators.run(make_ohlcv(100))
    r = ai.run(df, {}, ticker="AAPL", config=cfg, provider=provider)
    assert r.source == "remote" and r.model == "m"
    assert r.headline == "AAPL 短线偏强" and len(r.bullets) == 2


def test_remote_empty_body_triggers_local_fallback():
    cfg = Config(ai_api_key="sk-test")
    df = indicators.run(make_ohlcv(80))
    r = ai.run(df, {}, ticker="AAPL", config=cfg, provider=lambda *a: "   ")
    assert r.source == "local"
    assert r.warnings


def test_ai_memo_hit_on_identical_input():
    from dataflow import FingerprintMemo
    cfg = Config(ai_api_key="sk-test")
    memo = FingerprintMemo()
    df = indicators.run(make_ohlcv(80, seed=5))
    calls = {"n": 0}
    def provider(system, user, config):
        calls["n"] += 1
        return "标题\n- 要点"
    ai.run(df, {}, ticker="AAPL", config=cfg, memo=memo,
           memo_key="AAPL", provider=provider)
    ai.run(df, {}, ticker="AAPL", config=cfg, memo=memo,
           memo_key="AAPL", provider=provider)
    assert calls["n"] == 1 and memo.stats["hits"] == 1
