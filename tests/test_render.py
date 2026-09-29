from conftest import make_ohlcv
from dataflow import indicators, render
from dataflow.fetch import empty_quotes


def _ctx(df, info=None, prediction=None, ai_result=None):
    return render.RenderContext(
        ticker="AAPL",
        quotes=df,
        indicators=indicators.run(df) if not df.empty else df,
        info=info or {},
        prediction=prediction or {},
        ai=ai_result or {},
    )


def test_render_six_tabs_ok_on_full_data():
    ctx = _ctx(
        make_ohlcv(200),
        info={"longName": "Apple", "marketCap": 3e12},
        prediction={"prediction": 55.0, "current_price": 54.0,
                    "predicted_change_pct": 1.85,
                    "train_score": 0.9, "test_score": 0.8,
                    "n_train_rows": 180, "feature_importance": {"f1": 1.0}},
        ai_result={"headline": "h", "bullets": ["a"], "source": "remote"},
    )
    result = render.run(ctx)
    assert [t.id for t in result.tabs] == [
        "overview", "chart", "performance", "prediction", "ai", "data"]
    assert all(t.status == "ok" for t in result.tabs)
    chart = next(t for t in result.tabs if t.id == "chart")
    assert len(chart.payload["traces"]) >= 4


def test_render_empty_data_degrades_tabs_without_raising():
    ctx = _ctx(
        empty_quotes(),
        prediction={"reason": "样本不足"},
        ai_result={"source": "placeholder", "headline": "",
                   "bullets": ["占位"], "warnings": []},
    )
    result = render.run(ctx)
    statuses = {t.id: t.status for t in result.tabs}
    assert statuses["overview"] == "degraded"
    assert statuses["chart"] == "degraded"
    assert statuses["prediction"] == "degraded"
    assert statuses["ai"] == "degraded"


def test_render_isolates_single_tab_failure():
    ctx = _ctx(make_ohlcv(100))
    # 用一个会炸的 payload 触发 prediction 标签内部异常
    class Boom:
        def get(self, key, default=None):
            raise RuntimeError("boom in tab")
    ctx.prediction = Boom()
    result = render.run(ctx)
    pred_tab = next(t for t in result.tabs if t.id == "prediction")
    assert pred_tab.status == "error" and "boom" in pred_tab.error
    others_ok = [t for t in result.tabs if t.id != "prediction"]
    assert all(t.status in ("ok", "degraded") for t in others_ok)
