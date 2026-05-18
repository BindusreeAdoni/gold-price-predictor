# 🥇 Gold Price Predictor — CNN + Bidirectional LSTM

> Deep learning model trained on 26 years of gold price data to predict prices and directional market movement.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen?style=flat-square)

**🔴 Live App → [gold-price-predictor-cnnbilstm.streamlit.app](https://gold-price-predictor-cnnbilstm.streamlit.app)**

---

## Overview

This project uses a CNN + Bidirectional LSTM neural network to learn from 26 years of gold price history (2000–present) and predict future prices. Data is fetched live via yfinance and the model is deployed as an interactive web app on Streamlit Cloud.

---

## Features

- 📡 **Live data feed** — gold prices auto-refresh every hour via yfinance
- 📅 **Today's movement** — Open, High, Low, Close and intraday range for the latest trading day
- 🎯 **Predicted vs Actual** — model predictions overlaid on real test data
- 📊 **5 evaluation metrics** — RMSE, MAE, R², MAPE, Directional Accuracy with plain-English explanations
- 🔮 **Future forecast** — 7 / 14 / 30 day projection with ±2% confidence band
- 📈 **Interactive price history** — 1M / 6M / 1Y / 5Y / All range selector

---

## Model Architecture

```
Input → 60 days × 7 features (Open, High, Low, Close, Volume, MA7, MA30)
    ↓
Conv1D (64 filters) → MaxPooling → Dropout
    ↓
Bidirectional LSTM (100 units) → Dropout
    ↓
Bidirectional LSTM (50 units) → Dropout
    ↓
Dense (25) → Dense (1) → Predicted Close Price
```

**Why this architecture?**
- **CNN** — extracts local short-term price patterns across the 60-day window
- **Bi-LSTM** — reads the sequence both forward and backward, capturing long-range trend dependencies
- **60-day window** — covers ~3 months of trading, enough context for trend and volatility

---

## Dataset

| | |
|---|---|
| Source | Historical Gold/USD (GC=F via yfinance) |
| Range | August 2000 – present |
| Size | ~6,400 trading days |
| Features | Open, High, Low, Close, Volume, MA7, MA30 |

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| **RMSE** | Penalises large errors heavily — sensitive to volatility spikes |
| **MAE** | Plain average daily prediction error |
| **R²** | How much of the price variation the model explains |
| **MAPE** | Error as % of actual price — intuitive for non-technical audiences |
| **Directional Accuracy** | How often the model correctly predicted up vs down movement |

---

## Project Structure

```
gold-price-predictor/
├── app.py                   # Streamlit web app
├── cnn_bilstm_gold.keras    # Trained model
├── scaler_X.pkl             # Feature scaler
├── scaler_y.pkl             # Target scaler
├── GoldUSD_processed.csv    # Fallback dataset
├── requirements.txt         # Dependencies
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/BindusreeAdoni/gold-price-predictor.git
cd gold-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations & Future Scope

- Model uses OHLCV data only — cannot anticipate geopolitical events or macroeconomic shifts
- Recursive multi-day forecasting compounds errors over time
- **Planned:** macroeconomic features (DXY, Oil, S&P500), news sentiment via FinBERT, walk-forward validation

---

## Tech Stack

Python · TensorFlow · Keras · Streamlit · Plotly · yfinance · scikit-learn · NumPy · pandas

---

## Author

**Adoni Bindu Sree**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/bindusreeadoni/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/BindusreeAdoni)
