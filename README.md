# 📈 StockCast: Comparative Forecasting of AAPL, TSLA, and MSFT

A comprehensive comparative study of three popular time series forecasting models — ARIMA, Facebook Prophet, and LSTM — applied to stock prices of Apple (AAPL), Tesla (TSLA), and Microsoft (MSFT). The project benchmarks each model using MAE and RMSE, assessing their performance in stable and volatile markets, and provides insights for financial forecasting.

## 🚀 Features

### Core Functionality
- **End-to-end pipeline** for stock price forecasting  
- **Real-time data** sourced from [Yahoo Finance](https://finance.yahoo.com/)  
- **Multi-model approach** with comprehensive benchmarking
- **Interactive EDA** with advanced visualizations

### Models Implemented
- **ARIMA** – classic statistical baseline with parameter optimization
- **Prophet** – interpretable business forecasting tool with hyperparameter tuning
- **LSTM** – deep learning for nonlinear patterns with automated hyperparameter search
- **Hybrid Ensemble** – combines multiple models for improved accuracy

### Advanced Features
- **Hyperparameter optimization** using Keras Tuner for LSTM models
- **Automated model selection** with Auto ARIMA
- **Interactive web application** built with Streamlit
- **Real-time forecasting** with customizable prediction horizons
- **Comprehensive evaluation** using MAE, RMSE, and residual analysis
- **Professional visualizations** with Plotly and Matplotlib

---

## 📊 Results Summary

### Model Performance Insights
- **ARIMA** → Best on AAPL & MSFT, efficient & interpretable, weaker on volatile TSLA  
- **Prophet** → Smooth forecasts, highly interpretable, but struggled with volatility  
- **LSTM** → Strongest on TSLA, competitive with ARIMA, but "black box" and costly to train
- **Hybrid Models** → Best overall performance by combining strengths of multiple approaches

### Key Findings
- LSTM models excel in capturing nonlinear patterns in volatile stocks like TSLA
- ARIMA remains robust for stable, predictable stocks like AAPL and MSFT
- Prophet provides excellent interpretability but may oversimplify complex market dynamics
- Ensemble approaches offer the most reliable predictions across different market conditions

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11+**  
- **Streamlit** - Interactive web application framework
- **Plotly** - Advanced interactive visualizations
- **YFinance** - Real-time financial data

### Machine Learning Libraries
- `pandas`, `numpy` - Data manipulation and analysis
- `statsmodels`, `pmdarima` - ARIMA modeling and analysis
- `prophet` - Facebook's Prophet forecasting tool
- `tensorflow/keras` - Deep learning with LSTM networks
- `keras-tuner` - Automated hyperparameter optimization
- `scikit-learn` - Machine learning utilities

### Visualization & Analysis
- `matplotlib`, `seaborn` - Statistical plotting
- `plotly` - Interactive charts and dashboards

---

## 🎯 Project Structure

```
stock_market_analysis/
├── app_streamlit.py          # Interactive web application
├── main.ipynb               # Complete analysis notebook
├── plots.py                 # Visualization utilities
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── notebooks/              # Individual analysis notebooks
│   ├── 1-Data_collection&EDA.ipynb
│   ├── 2-ARIMA.ipynb
│   ├── 3-Prophet.ipynb
│   ├── 4-LSTM.ipynb
│   └── 5-Evaluation.ipynb
└── output/                 # Generated data and models
    ├── Dataset/           # Stock price data
    ├── models/           # Trained model files
    │   ├── lstm/        # LSTM models and scalers
    │   └── lstm_tuner/  # Hyperparameter tuning results
    └── *.joblib         # Serialized model objects
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/angelogoves/StockCast-Comparative-Time-Series-Forecasting.git
   cd StockCast-Comparative-Time-Series-Forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the interactive application:**
   ```bash
   streamlit run app_streamlit.py
   ```

### Using the Web Application

The Streamlit application provides an intuitive interface for:
- **Stock Selection** - Choose from NASDAQ-listed stocks
- **Parameter Configuration** - Customize model parameters
- **Forecast Generation** - Generate predictions with different models
- **Interactive Visualization** - Explore results with dynamic charts

### Running the Analysis

Execute the main notebook for the complete analysis:
```bash
jupyter notebook main.ipynb
```

This will run through:
1. Data collection and preprocessing
2. Exploratory data analysis
3. Model training and optimization
4. Performance evaluation and comparison
5. Results visualization and interpretation

### View Live Application

The interactive web application is deployed on Streamlit and can be accessed at:
**[https://stockcast-comparative-time-series-forecasting.streamlit.app/](https://stockcast-comparative-time-series-forecasting.streamlit.app/)**

The live application provides the same functionality as running locally, with real-time data and interactive visualizations.

---

## 📈 Models Overview

### ARIMA (AutoRegressive Integrated Moving Average)
- **Strengths**: Interpretable, efficient for stable time series
- **Use Case**: Best for stocks with consistent patterns (AAPL, MSFT)
- **Features**: Automated parameter selection with Auto ARIMA

### Prophet (Facebook's Forecasting Tool)
- **Strengths**: Handles seasonality, holidays, and changepoints automatically
- **Use Case**: Business forecasting with interpretable components
- **Features**: Hyperparameter tuning for optimal performance

### LSTM (Long Short-Term Memory)
- **Strengths**: Captures complex nonlinear patterns and long-term dependencies
- **Use Case**: Volatile stocks with complex dynamics (TSLA)
- **Features**: Automated hyperparameter optimization with Keras Tuner

### Hybrid Ensemble
- **Approach**: Combines predictions from multiple models
- **Benefit**: Reduces individual model biases and improves robustness
- **Implementation**: Weighted averaging of ARIMA, Prophet, and LSTM predictions

---

## 📊 Evaluation Metrics

### Primary Metrics
- **MAE (Mean Absolute Error)** - Average absolute prediction error
- **RMSE (Root Mean Square Error)** - Penalizes larger errors more heavily

### Additional Analysis
- **Residual Analysis** - Examines prediction error patterns
- **Error Distribution** - Statistical properties of forecast errors
- **Model Comparison** - Side-by-side performance evaluation

---

## 🔬 Technical Implementation

### Data Pipeline
1. **Real-time Data Collection** from Yahoo Finance
2. **Data Preprocessing** with NaN handling and business day alignment
3. **Feature Engineering** including returns and technical indicators
4. **Train-Test Split** with time series considerations

### Model Training
- **Cross-validation** for hyperparameter optimization
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** for stable convergence
- **Model persistence** for deployment and reuse

### Visualization System
- **Interactive charts** with Plotly for exploration
- **Statistical plots** with Matplotlib for analysis
- **Dashboard integration** with Streamlit components

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing financial data
- **Facebook Prophet** team for their excellent forecasting library
- **TensorFlow/Keras** team for powerful deep learning tools
- **Streamlit** team for making web applications accessible
