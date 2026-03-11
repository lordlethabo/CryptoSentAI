import streamlit as st
import pandas as pd

from src.pipeline import run_pipeline
from src.performance_tracker import calculate_performance


st.set_page_config(
    page_title="CryptoSentAI Dashboard",
    layout="wide"
)


st.title("CryptoSentAI Trading Dashboard")


# ------------------------------------------------
# Run AI pipeline
# ------------------------------------------------

if st.button("Run AI Analysis"):

    results = run_pipeline()

    st.subheader("Market Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Price",
        round(results["current_price"], 2)
    )

    col2.metric(
        "Predicted Price",
        round(results["prediction"], 2)
    )

    col3.metric(
        "Confidence Score",
        round(results["confidence"], 2)
    )

    st.subheader("Trading Signal")

    st.write(results["signal"])

    st.subheader("Sentiment Score")

    st.write(results["sentiment"])


# ------------------------------------------------
# Performance statistics
# ------------------------------------------------

st.subheader("Strategy Performance")

stats = calculate_performance()

if stats:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Trades",
        stats["total_trades"]
    )

    col2.metric(
        "Win Rate",
        round(stats["win_rate"] * 100, 2)
    )

    col3.metric(
        "Total Profit",
        round(stats["total_profit"], 2)
    )

else:

    st.write("No trade history yet.")


# ------------------------------------------------
# Backtest display
# ------------------------------------------------

st.subheader("Backtest Results")

try:

    df = pd.read_csv("trade_history.csv")

    st.dataframe(df)

except:

    st.write("Backtest results not available yet.")
