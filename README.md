<div align="center">
  
# 🚀 Quantitative Momentum Scanner (XGBoost)
**An algorithmic momentum identification engine for Indian Equities (NSE).**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

</div>

<br>

> **The Problem:** Traditional momentum screeners rely on rigid, static percentage targets (e.g., "sell at +2%"). Market volatility is dynamic, meaning static targets consistently lead to premature exits or missed take-profits.
> 
> **The Solution:** This system abandons static targets. It calculates the Average True Range (ATR) of an asset and trains an eXtreme Gradient Boosting (XGBoost) classification engine to predict the probability of a volatility-adjusted expansion over a rolling forward window.

---

## 🧠 System Architecture & Workflow

The algorithm processes data through a strict quantitative pipeline:

1. **Data Ingestion:** Fetches 10-year historical OHLCV data for user-defined NSE tickers and the benchmark index (`^NSEI`).
2. **Feature Engineering:** Calculates 9 custom momentum, volume, and volatility indicators.
3. **Dynamic Target Labelling:** Computes a custom multiplier against the ATR to establish a moving "Take Profit" zone. If the future price hits this zone, the historical data row is labelled as a `1` (Win).
4. **Machine Learning Optimization:** Runs a `RandomizedSearchCV` across an XGBoost Classifier, utilizing `TimeSeriesSplit` to prevent look-ahead bias and `scale_pos_weight` to manage class imbalance.
5. **Inference & Ranking:** Evaluates today's closing data through the winning engine and outputs a live probability matrix.

---

## ⚙️ Predictive Indicators Engine

The model engineers multiple features from raw `yfinance` data to feed the XGBoost trees. 

| Indicator | Mechanism | Purpose |
| :--- | :--- | :--- |
| **Relative Strength** | Spread vs. `^NSEI` | Measures if the asset is independently outperforming the broader Indian market. |
| **SMA Distance** | `%` spread from SMA | Identifies mean-reversion pullbacks or extreme over-extensions. |
| **Volume Surge** | Z-Score normalization | Flags sudden institutional accumulation against historical baselines. |
| **Volatility** | Rolling Standard Dev. | Adapts the model to the asset's current daily expected moves. |
| **BB Squeeze** | Bollinger Band Width | Identifies extreme price compression preceding explosive breakouts. |
| **MACD Histogram** | Derivative of MACD | Pinpoints the exact moment downward momentum dies and buying accelerates. |
| **Rate of Change** | 3-Day `%` Velocity | Measures pure, unfiltered, short-term price aggression. |
| **VWAP Deviation** | Spread from VWAP | Identifies pullbacks to the institutional "fair value" baseline. |
| **RSI Metric** | Win/Loss Ratio | Traditional overbought/oversold oscillator. |

---

## 📊 Terminal Output & Ranking System

The script outputs a ranked summary of all queried assets, sorting them by Live Signal Strength, Historical Win Rate, and Model Confidence.

![Terminal Output Summary](XGBOOST_FINAL_edit.png)

### Signal Classifications:
* 🔥 **STRONG BUY:** Model confidence is significantly above the user-defined threshold (+15%).
* 🟢 **BUY:** Model confidence meets the baseline threshold requirement.
* 🟡 **WEAK BUY:** Model confidence is nearing the threshold (-10%), alerting the user to monitor charts for a setup.
* 🔴 **WAIT:** Setup is statistically invalid based on current indicator data.

---

## ⚠️ Proprietary Notice (Scrubbed Alpha)

**Important:** The hyperparameters, ATR multipliers, lookback windows, and aggression factors contained in this public repository have been reset to generic textbook defaults. 

This has been done to protect proprietary alpha. The defaults included in this script *will* run and execute properly, but users deploying this code in live environments must perform their own `RandomizedSearchCV` optimizations based on their specific asset class, timeframe, and risk tolerance to achieve institutional win rates.

---

## 🚀 Installation & Usage

**1. Clone the repository & install dependencies:**
```bash
git clone https://github.com/Anirudh-AI1/Quant_XGBoost_Momentum_Model.git
cd Quant_XGBoost_Momentum_Model
pip install pandas numpy matplotlib scikit-learn xgboost yfinance tabulate

python XGBoost_target_predictor_github.py

That will render cleanly so people can just click the "copy" button on GitHub and paste it straight into their terminals. 
