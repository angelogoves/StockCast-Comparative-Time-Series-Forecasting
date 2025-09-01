import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def plot_stock_analysis(stock_data, stock_name, ma_window=20):
    df = stock_data.copy()
    df.index = pd.to_datetime(df.index)

    # 1. Calculations
    df["Returns"] = df["Close"].pct_change()
    df["MA"] = df["Close"].rolling(window=ma_window).mean()
    df["STD"] = df["Close"].rolling(window=ma_window).std()
    df["Upper_Band"] = df["MA"] + (df["STD"] * 2)
    df["Lower_Band"] = df["MA"] - (df["STD"] * 2)

    # 2. Create 2x3 Subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Historical Price",
            "Price Distribution",
            "Price Frequency",
            "Moving Average",
            "Bollinger Bands",
            "Daily Returns",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.07,
    )

    # --- ROW 1 ---
    # Plot 1: Close Price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="white")),
        row=1,
        col=1,
    )

    # Plot 2: Boxplot
    fig.add_trace(
        go.Box(y=df["Close"], name="Price Dist", marker_color="cyan"), row=1, col=2
    )

    # Plot 3: Histogram
    fig.add_trace(
        go.Histogram(
            x=df["Close"],
            nbinsx=30,
            name="Frequency",
            marker_color="rgba(0, 255, 255, 0.6)",
        ),
        row=1,
        col=3,
    )

    # --- ROW 2 ---
    # Plot 4: Moving Average
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Price",
            opacity=0.3,
            line=dict(color="gray"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MA"], name=f"{ma_window}d MA", line=dict(color="orange")
        ),
        row=2,
        col=1,
    )

    # Plot 5: Bollinger Bands
    # Upper/Lower Bands with Shading
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Upper_Band"],
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Lower_Band"],
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(0, 255, 255, 0.1)",
            name="Volatility Zone",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], name="Price", line=dict(color="white", width=1)
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA"],
            name="Mid Band",
            line=dict(color="blue", dash="dash"),
        ),
        row=2,
        col=2,
    )

    # Plot 6: Daily Returns
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Returns"], name="Returns", line=dict(color="lime")
        ),
        row=2,
        col=3,
    )

    # 3. Final Styling
    fig.update_layout(
        # title_text=f"Interactive EDA for {stock_name}",
        template="plotly_dark",
        height=800,
        showlegend=False,
        hovermode="x unified",
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
