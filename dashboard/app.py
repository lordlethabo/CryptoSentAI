import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.pipeline import run_pipeline


st.set_page_config(
    page_title="CryptoSentAI",
    layout="wide"
)

st.title("CryptoSentAI Dashboard")

st.write(
"""
AI-powered crypto intelligence platform combining sentiment analysis,
machine learning forecasting, and trading strategy backtesting.
"""
)


st.sidebar.header("API Configuration")

api_key = st.sidebar.text_input("Binance API Key")

api_secret = st.sidebar.text_input(
    "Binance API Secret",
    type="password"
)

symbol = st.sidebar.text_input(
    "Trading Pair",
    value="BTCUSDT"
)


if st.sidebar.button("Run Analysis"):

    with st.spinner("Running CryptoSentAI pipeline..."):

        result = run_pipeline(api_key, api_secret, symbol)


    st.header("Market Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Price",
        f"{result['current_price']:.2f}"
    )

    col2.metric(
        "Predicted Price",
        f"{result['prediction']:.2f}"
    )

    col3.metric(
        "Trading Signal",
        result["signal"]
    )


    st.header("Sentiment Analysis")

    st.write(
        "Market Sentiment Score:",
        result["sentiment"]
    )


    st.header("Backtesting Results")

    backtest = result["backtest"]

    st.write(
        "Strategy Return (%)",
        round(backtest["strategy_return_percent"], 2)
    )

    st.write(
        "Buy & Hold Return (%)",
        round(backtest["buy_hold_return_percent"], 2)
    )


    st.header("Portfolio Performance")

    history = backtest["portfolio_history"]

    fig, ax = plt.subplots()

    ax.plot(history)

    ax.set_title("Portfolio Value Over Time")

    ax.set_xlabel("Trade Step")

    ax.set_ylabel("Portfolio Value")

    st.pyplot(fig)
