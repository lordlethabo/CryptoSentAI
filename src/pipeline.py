import pandas as pd

from src.data_ingestion import fetch_crypto_prices
from src.sentiment_model import analyze_sentiment
from src.feature_engineering import build_features
from src.ml_models import train_models, make_predictions
from src.lstm_model import train_lstm, predict_lstm
from src.trading_strategy import generate_signal
from src.backtesting import backtest_strategy


def run_pipeline(api_key, api_secret, symbol="BTCUSDT"):

    print("Starting CryptoSentAI pipeline")

    print("Fetching market data...")
    df_price = fetch_crypto_prices(api_key, api_secret, symbol)

    if df_price is None or df_price.empty:
        raise ValueError("Failed to retrieve market data")

    print("Running sentiment analysis...")
    sentiment_score = analyze_sentiment()

    print("Building feature dataset...")
    df_features = build_features(df_price, sentiment_score)

    print("Training machine learning models...")
    lr_model, rf_model = train_models(df_features)

    print("Generating ML ensemble prediction...")
    ml_prediction = make_predictions(
        lr_model,
        rf_model,
        df_features
    )

    print("Training LSTM model...")
    lstm_model, lstm_scaler = train_lstm(df_price)

    print("Generating LSTM prediction...")
    lstm_prediction = predict_lstm(
        lstm_model,
        lstm_scaler,
        df_price
    )

    print("Combining predictions...")

    prediction = (
        ml_prediction +
        lstm_prediction
    ) / 2

    current_price = df_features["close"].iloc[-1]

    print("Generating trading signal...")

    signal = generate_signal(
        prediction,
        df_features
    )

    print("Running historical backtest...")

    backtest_results = backtest_strategy(
        df_features,
        lr_model,
        rf_model,
        generate_signal
    )

    print("Pipeline execution complete")

    return {

        "current_price": current_price,

        "prediction": prediction,

        "signal": signal,

        "sentiment": sentiment_score,

        "backtest": backtest_results
    }
