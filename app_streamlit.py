from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

from plots import plot_stock_analysis

# -----------------------------
# APP SETUP
# -----------------------------
st.set_page_config(page_title="Nasdaq AI Hybrid Predictor", layout="wide")

LSTM_SUPPORTED_STOCKS = ["AAPL", "TSLA", "MSFT"]


@st.cache_data
def load_assets():
    df = pd.read_csv("output/nasdaq_stocks_list.csv")
    ticker_dict = dict(zip(df["Symbol"].str.strip(), df["Security Name"].str.strip()))
    scaler = joblib.load("output/models/lstm/scaler_lstm.pkl")
    return ticker_dict, scaler


ticker_dict, scaler = load_assets()


def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i : i + seq_length, 0])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)


# -----------------------------
# USER INTERFACE
# -----------------------------
st.title("📈 Nasdaq Stock Predictor")

st.sidebar.header("Stock Parameters")
st.sidebar.text("Select a stock, date range, and forecast horizon.")

selected_stock = st.sidebar.selectbox("Select a Nasdaq Stock:", options=ticker_dict)
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

# -----------------------------
# MAIN LOGIC
# -----------------------------
if st.sidebar.button("Get Data"):

    # -----------------------------
    # LOAD FULL DATA (MODELS USE THIS)
    # -----------------------------
    full_stock_data = yf.download(
        tickers=selected_stock,
        period="max",
        interval="1d",
        auto_adjust=False,
        multi_level_index=False,
    )

    full_stock_data.index = pd.to_datetime(full_stock_data.index)
    full_stock_data = full_stock_data.sort_index()

    # -----------------------------
    # FILTERED DATA (UI ONLY)
    # -----------------------------
    display_stock_data = full_stock_data.loc[start_date:end_date]

    # -----------------------------
    # COMPANY INFO
    # -----------------------------
    logo_url = f"https://eodhd.com/img/logos/US/{selected_stock}.png"
    st.markdown(
        f'<img src="{logo_url}" style="width:300px; border-radius:10px;">',
        unsafe_allow_html=True,
    )

    ticker_info = yf.Ticker(selected_stock).info
    st.header(ticker_info.get("longName", selected_stock))
    st.info(ticker_info.get("longBusinessSummary", ""))

    st.header("Ticker Data")
    st.write(display_stock_data)

    st.header(f"Interactive EDA for {selected_stock}")
    plot_stock_analysis(display_stock_data, selected_stock, 30)
