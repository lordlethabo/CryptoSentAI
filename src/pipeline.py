# CryptoSentAI Main Pipeline
# This script orchestrates the entire AI workflow

import pandas as pd
import sqlite3

from data_ingestion import fetch_crypto_prices
from sentiment_model import analyze_sentiment
from feature_engineering import build_features
from ml_models import train_models, make_predictions
from trading_strategy import generate_signal


def log_prediction(predicted_price, actual_price=None):

    conn = sqlite3.connect("data/sentiment_logs.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        date TEXT,
        predicted_price REAL,
        actual_price REAL
    )
    """)

    from datetime import datetime
    today = datetime.today().strftime("%Y-%m-%d")

    cursor.execute(
        "INSERT INTO predictions (date, predicted_price, actual_price) VALUES (?, ?, ?)",
        (today, predicted_price, actual_price)
    )

    conn.commit()
    conn.close()


def run_pipeline(api_key, api_secret, symbol="BTCUSDT"):

    print("Starting CryptoSentAI pipeline")
    print("--------------------------------")

    # STEP 1: Data Ingestion
    print("Fetching cryptocurrency market data...")
    df_price = fetch_crypto_prices(api_key, api_secret, symbol)

    if df_price is None or df_price.empty:
        print("Market data fetch failed.")
        return

    print("Market data loaded:", len(df_price), "rows")

    # STEP 2: Sentiment Analysis
    print("Running sentiment analysis...")

    sentiment_score = analyze_sentiment()

    print("Sentiment score:", sentiment_score)

    # STEP 3: Feature Engineering
    print("Building ML features...")

    df_features = build_features(df_price, sentiment_score)

    print("Feature dataset size:", df_features.shape)

    # STEP 4: Train Models
    print("Training machine learning models...")

    lr_model, rf_model = train_models(df_features)

    # STEP 5: Generate Predictions
    print("Generating price prediction...")

    prediction = make_predictions(lr_model, rf_model, df_features)

    current_price = df_features["close"].iloc[-1]

    print("Current BTC Price:", current_price)
    print("Predicted Next Price:", prediction)

    # STEP 6: Generate Trading Signal
    print("Generating trading signal...")

    signal = generate_signal(prediction, df_features)

    print("Trading Signal:", signal)

    # STEP 7: Store prediction
    print("Logging prediction to database...")

    log_prediction(prediction)

    print("--------------------------------")
    print("Pipeline completed successfully")

    return {
        "current_price": current_price,
        "prediction": prediction,
        "signal": signal,
        "sentiment": sentiment_score
    }


if __name__ == "__main__":

    print("CryptoSentAI Execution")

    api_key = input("Enter Binance API Key: ")
    api_secret = input("Enter Binance API Secret: ")

    result = run_pipeline(api_key, api_secret)

    print("\nFinal Output")
    print(result)
