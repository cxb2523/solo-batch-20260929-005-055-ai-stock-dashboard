# 🚀 AI-Powered Stock Market Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/erikthiart/ai-stock-dashboard)](https://github.com/erikthiart/ai-stock-dashboard)

> **Professional-grade stock analysis with machine learning predictions and real-time technical indicators**

A comprehensive, AI-powered stock market dashboard that combines advanced technical analysis, machine learning price predictions, and intelligent market insights in a beautiful, interactive interface.

![Main Dashboard](screenshots/main_dashboard.jpg)

## ✨ Features

### 🤖 **Artificial Intelligence**
- **Machine Learning Price Prediction** - Random Forest model with 30+ technical features
- **AI Market Analysis** - Natural language insights based on technical indicators
- **Feature Importance Analysis** - Understand what drives price movements
- **Model Performance Metrics** - Train/test accuracy with confidence levels

### 📈 **Advanced Technical Analysis**
- **Professional Charts** - Multi-panel candlestick charts with technical overlays
- **20+ Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages, Stochastic
- **Volume Analysis** - Volume trends and confirmation signals
- **Performance Metrics** - Sharpe ratio, volatility, maximum drawdown

### 🎯 **Real-Time Data**
- **Live Stock Data** - Real-time prices from Yahoo Finance
- **Multiple Timeframes** - 1M to 5Y analysis periods
- **Popular Stock Presets** - Quick access to FAANG+ stocks
- **Custom Symbol Input** - Analyze any publicly traded stock

### 🎨 **Professional Interface**
- **Dark Theme** - Easy on the eyes for extended analysis
- **Responsive Design** - Works perfectly on desktop and mobile
- **Interactive Charts** - Zoom, pan, and explore data
- **Organized Tabs** - Clean separation of different analysis types

![Technical Analysis](screenshots/technical_analysis.jpg)

## 🧩 数据流架构与追踪（dataflow）

原本糊在 `stock_dashboard.py` 里的逻辑被拆成 `dataflow/` 包，五个阶段各留一个 trace 钩子，
记录**输入列名、时区、行数、耗时与异常**：

```text
ticker 输入 -> fetch -> indicators -> predict -> ai -> render
             (dataflow/fetch.py) (indicators.py) (predict.py) (ai.py) (render.py)
```

- `dataflow/trace.py`：`RunTrace` / `StageTrace` / `TraceStore`（按 ticker 的环形缓冲）
- `dataflow/cache.py`：新鲜 TTL + 陈旧回退缓存、数据指纹
- `dataflow/pipeline.py`：五段编排，单段异常被捕获并转成标签页占位，整页不崩

### 三个工程取舍（已落进代码）

1. **缓存粒度**：历史 K 线缓存键 = `ticker + start/end 日期区间`（`period` 在请求时换算，
   见 `fetch.resolve_range` / `history_cache_key`），避免“1y”命中旧缓存后区间永不滚动；
   最新行情走独立的短 TTL 缓存。新鲜 TTL 内命中即复用（视为含最新收盘），
   过期/强制刷新才取新；取新失败时在陈旧宽限 TTL 内回退上一份并单独计数 `stale`。
   TTL 可用 `DATAFLOW_HISTORY_TTL`、`DATAFLOW_QUOTE_TTL` 等环境变量覆盖。
2. **AI 降级**：缺 `OPENAI_API_KEY` / `DATAFLOW_AI_API_KEY` 时不请求网络，
   AI 标签页显示本地规则分析（占位）；有密钥但请求失败时优先复用同 ticker 的旧 AI 缓存，
   再退化为本地占位。任意单段异常都不会让整页崩掉。
3. **刷新复用**：普通刷新时 `fetch` 由 TTL 决定是否取新；`indicators` / `predict` / `ai`
   按输入数据指纹复用（trace 中带“复用”徽标），`render` 永远重算；
   “强制刷新”绕过所有缓存。指标在样本不足或除零时一律返回 **NaN**，不抛异常。

### 追踪面板与测试

```bash
# Flask 追踪面板：按 ticker 展示各阶段实际数据、行数、时区、耗时与降级提示
python trace_app.py                  # http://127.0.0.1:5000
DATAFLOW_DEMO=1 python trace_app.py  # 无网络时用内置合成数据演示（EMPTY 可看空数据降级）

# 测试（含空数据、单行、除零、缓存命中次数断言）
python -m pytest -q

# 原 Streamlit 入口保留，现在只是 dataflow 的薄封装
streamlit run stock_dashboard.py
```

JSON trace：`GET /trace/<TICKER>`，健康检查：`GET /healthz`。

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/erikthiart/ai-stock-dashboard.git
cd ai-stock-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run stock_dashboard.py
```

4. **Open your browser**
```
Navigate to http://localhost:8501
```

![ML Predictions](screenshots/ml_predictions.jpg)

## 📦 Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
```

## 🎮 How to Use

### 1. **Select Your Stock**
- Choose from popular presets (Apple, Tesla, Google, etc.)
- Or enter any stock symbol manually
- Select your preferred analysis timeframe

### 2. **Explore the Analysis**
- **Main Dashboard**: Key metrics and price changes
- **Technical Charts**: Advanced multi-panel analysis
- **Performance**: Risk metrics and cumulative returns
- **AI Predictions**: Machine learning price forecasts
- **Market Analysis**: AI-generated insights

### 3. **Understand the Insights**
- 🟢 **Green indicators**: Bullish signals
- 🔴 **Red indicators**: Bearish signals  
- 🟡 **Yellow indicators**: Neutral/mixed signals
- ⚠️ **Warning indicators**: Overbought/oversold conditions

![Performance Metrics](screenshots/performance_metrics.jpg)

## 🧠 Machine Learning Model

Our AI uses a **Random Forest Regressor** trained on 30+ features including:

- **Price-based features**: Returns, volatility, price changes
- **Technical indicators**: RSI, MACD, moving averages
- **Volume features**: Volume ratios and trends  
- **Lag features**: Historical price and volume data
- **Statistical features**: Rolling means and standard deviations

**Model Performance:**
- Real-time training on historical data
- Cross-validation with train/test splits
- Feature importance analysis
- Confidence metrics displayed

![AI Analysis](screenshots/ai_analysis.jpg)

## 📊 Technical Indicators

| Indicator | Purpose | Interpretation |
|-----------|---------|----------------|
| **RSI** | Momentum | >70 Overbought, <30 Oversold |
| **MACD** | Trend | Signal line crossovers |
| **Bollinger Bands** | Volatility | Price vs. bands position |
| **Moving Averages** | Trend | Price vs. MA relationships |
| **Stochastic** | Momentum | %K and %D oscillator |
| **Volume** | Confirmation | Volume vs. average ratios |

## 🎯 Use Cases

### 📈 **For Traders**
- Quick technical analysis of any stock
- AI-powered price predictions for next trading day
- Volume confirmation signals
- Multiple timeframe analysis

### 💼 **For Investors**
- Long-term performance metrics
- Risk assessment (volatility, drawdown)
- Company fundamental information
- Market trend analysis

### 🎓 **For Learning**
- Understanding technical indicators
- Machine learning in finance
- Market behavior patterns
- Professional chart analysis

![Company Info](screenshots/company_info.jpg)

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- Not financial advice or investment recommendations
- Past performance doesn't guarantee future results
- Always do your own research before investing
- Consider consulting with financial professionals
- Markets involve risk and potential loss of capital

## 🛠️ Technical Architecture

```
├── stock_dashboard.py      # Main application
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── screenshots/           # UI screenshots
    ├── main_dashboard.jpg
    ├── technical_analysis.jpg
    ├── ml_predictions.jpg
    ├── performance_metrics.jpg
    ├── ai_analysis.jpg
    └── company_info.jpg
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **Yahoo Finance** for providing free stock data
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning capabilities

## 📞 Support

If you find this project helpful, please give it a ⭐ on GitHub!

For questions or issues:
- Open an [Issue](https://github.com/erikthiart/ai-stock-dashboard/issues)

---

<div align="center">

**Built with ❤️ and Python**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/erikthiart)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/erikthiart)

</div>
