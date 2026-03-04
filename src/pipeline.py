import os
from dotenv import load_dotenv

from src.data_ingestion import fetch_crypto_prices
from src.sentiment_model import analyze_sentiment
from src.feature_engineering import build_features
from src.ml_models import train_models
from src.lstm_model import train_lstm, predict_lstm
from src.trading_strategy import generate_signal
from src.backtesting import backtest_strategy
from src.confidence import calculate_confidence


# Load environment variables
load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")


# Configuration
CONFIDENCE_THRESHOLD = 0.80


def run_pipeline(symbol="BTCUSDT"):

    print("Starting CryptoSentAI pipeline")
    print("--------------------------------")

    # Step 1: Fetch Market Data
    print("Fetching market data...")
    df_price = fetch_crypto_prices(
        BINANCE_API_KEY,
        BINANCE_API_SECRET,
        symbol
    )

    if df_price is None or df_price.empty:
        raise ValueError("Market data fetch failed")

    print("Market data rows:", len(df_price))


    # Step 2: Sentiment Analysis
    print("Running sentiment analysis...")
    sentiment_score = analyze_sentiment()

    print("Sentiment score:", sentiment_score)


    # Step 3: Feature Engineering
    print("Building ML features...")
    df_features = build_features(
        df_price,
        sentiment_score
    )

    print("Feature dataset size:", df_features.shape)


    # Step 4: Train ML Models
    print("Training ML models...")
    lr_model, rf_model = train_models(df_features)

    latest_features = df_features.iloc[-1:]

    X_latest = latest_features.drop(columns=["target"])


    # ML Predictions
    lr_prediction = lr_model.predict(X_latest)[0]
    rf_prediction = rf_model.predict(X_latest)[0]

    ml_prediction = (lr_prediction + rf_prediction) / 2

    print("ML prediction:", ml_prediction)


    # Step 5: Train LSTM Model
    print("Training LSTM model...")
    lstm_model, lstm_scaler = train_lstm(df_price)

    lstm_prediction = predict_lstm(
        lstm_model,
        lstm_scaler,
        df_price
    )

    print("LSTM prediction:", lstm_prediction)


    # Step 6: Ensemble Prediction
    prediction = (ml_prediction + lstm_prediction) / 2

    current_price = df_features["close"].iloc[-1]

    print("Current price:", current_price)
    print("Final prediction:", prediction)


    # Step 7: Confidence Score
    confidence = calculate_confidence(
        lr_prediction,
        rf_prediction,
        lstm_prediction,
        current_price,
        sentiment_score
    )

    print("Confidence score:", confidence)


    # Step 8: Trading Signal
    if confidence >= CONFIDENCE_THRESHOLD:

        signal = generate_signal(
            prediction,
            df_features
        )

        print("Trade allowed")

    else:

        signal = "HOLD"

        print("Confidence too low — no trade")

    print("Trading signal:", signal)


    # Step 9: Backtesting
    print("Running backtesting simulation...")

    backtest_results = backtest_strategy(
        df_features,
        lr_model,
        rf_model,
        generate_signal
    )

    print("Backtesting complete")


    print("--------------------------------")
    print("Pipeline finished")


    return {

        "current_price": current_price,

        "prediction": prediction,

        "confidence": confidence,

        "signal": signal,

        "sentiment": sentiment_score,

        "backtest": backtest_results
    }
