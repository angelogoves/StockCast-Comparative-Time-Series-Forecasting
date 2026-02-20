# 📈 StockCast: Comparative Forecasting of AAPL, TSLA, and MSFT

A comparative study of three popular time series forecasting models — ARIMA, Facebook Prophet, and LSTM — applied to stock prices of Apple (AAPL), Tesla (TSLA), and Microsoft (MSFT). The project benchmarks each model using MAE and RMSE, assessing their performance in stable and volatile markets, and provides insights for financial forecasting.

---

## 🚀 Features
- End-to-end pipeline for stock price forecasting  
- Data sourced from [Yahoo Finance](https://finance.yahoo.com/) (via Kaggle dataset)  
- Models implemented:  
  - **ARIMA** – classic statistical baseline  
  - **Prophet** – interpretable business forecasting tool  
  - **LSTM** – deep learning for nonlinear patterns  
- Performance evaluation using **MAE** and **RMSE**  
- Visualisations of forecasts, loss curves, and residuals  
- Comparative insights into model suitability across market conditions  

---

## 📊 Results Summary
- **ARIMA** → Best on AAPL & MSFT, efficient & interpretable, weaker on volatile TSLA  
- **Prophet** → Smooth forecasts, highly interpretable, but struggled with volatility  
- **LSTM** → Strongest on TSLA, competitive with ARIMA, but “black box” and costly to train  

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- Libraries: `pandas`, `numpy`, `statsmodels`, `fbprophet` / `prophet`, `tensorflow/keras`, `matplotlib`, `scikit-learn`

---
