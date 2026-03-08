import os
import asyncio
from dotenv import load_dotenv

from src.data_ingestion import fetch_crypto_prices
from src.telegram_signals import fetch_signals
from src.sentiment_model import analyze_sentiment
from src.feature_engineering import build_features, get_feature_columns
from src.ml_models import train_models, predict_next_price
from src.lstm_model import train_lstm, predict_lstm
from src.confidence import calculate_confidence
from src.trading_strategy import generate_signal
from src.trade_execution import execute_trade
from src.backtesting import backtest_strategy


# -------------------------------------------------
# Load environment configuration
# -------------------------------------------------

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.80))

CHANNEL_1 = os.getenv("TELEGRAM_CHANNEL_1")
CHANNEL_2 = os.getenv("TELEGRAM_CHANNEL_2")

TELEGRAM_LIMIT = int(os.getenv("TELEGRAM_MESSAGE_LIMIT", 50))


# -------------------------------------------------
# Telegram collection
# -------------------------------------------------

async def collect_telegram_messages():

    messages = []

    try:

        if CHANNEL_1:

            msgs = await fetch_signals(CHANNEL_1, TELEGRAM_LIMIT)

            messages.extend(msgs)

        if CHANNEL_2:

            msgs = await fetch_signals(CHANNEL_2, TELEGRAM_LIMIT)

            messages.extend(msgs)

    except Exception as e:

        print("Telegram fetch error:", e)

    return messages


# -------------------------------------------------
# Main AI pipeline
# -------------------------------------------------

def run_pipeline(symbol=SYMBOL):

    print("\n----------------------------------")
    print("Starting CryptoSentAI pipeline")
    print("----------------------------------\n")

    # ----------------------------------
    # Step 1: Fetch market data
    # ----------------------------------

    print("Fetching market data...")

    df_price = fetch_crypto_prices(
        BINANCE_API_KEY,
        BINANCE_API_SECRET,
        symbol
    )

    if df_price is None or df_price.empty:

        raise ValueError("Market data fetch failed")

    print("Market rows:", len(df_price))


    # ----------------------------------
    # Step 2: Collect Telegram signals
    # ----------------------------------

    print("Fetching Telegram signals...")

    telegram_messages = asyncio.run(
        collect_telegram_messages()
    )

    print("Messages collected:", len(telegram_messages))


    # ----------------------------------
    # Step 3: Sentiment analysis
    # ----------------------------------

    print("Running sentiment analysis...")

    sentiment_score = analyze_sentiment(
        telegram_messages
    )

    print("Sentiment score:", sentiment_score)


    # ----------------------------------
    # Step 4: Feature engineering
    # ----------------------------------

    print("Building ML features...")

    df_features = build_features(
        df_price,
        sentiment_score
    )


    # ----------------------------------
    # Step 5: Train ML models
    # ----------------------------------

    print("Training ML models...")

    lr_model, rf_model = train_models(df_features)

    predictions = predict_next_price(
        lr_model,
        rf_model,
        df_features
    )

    lr_prediction = predictions["lr_prediction"]
    rf_prediction = predictions["rf_prediction"]
    ml_prediction = predictions["ensemble_prediction"]


    # ----------------------------------
    # Step 6: Train LSTM model
    # ----------------------------------

    print("Training LSTM model...")

    lstm_model, scaler = train_lstm(df_price)

    lstm_prediction = predict_lstm(
        lstm_model,
        scaler,
        df_price
    )


    # ----------------------------------
    # Step 7: Ensemble prediction
    # ----------------------------------

    prediction = (ml_prediction + lstm_prediction) / 2

    current_price = df_features["close"].iloc[-1]

    print("Current price:", current_price)
    print("Predicted price:", prediction)


    # ----------------------------------
    # Step 8: Confidence scoring
    # ----------------------------------

    confidence = calculate_confidence(
        lr_prediction,
        rf_prediction,
        lstm_prediction,
        current_price,
        sentiment_score
    )

    print("Confidence score:", confidence)


    # ----------------------------------
    # Step 9: Generate trading signal
    # ----------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        signal = generate_signal(
            prediction,
            df_features
        )

        print("Trading signal:", signal)

    else:

        signal = "HOLD"

        print("Signal blocked (low confidence)")


    # ----------------------------------
    # Step 10: Execute trade
    # ----------------------------------

    if signal != "HOLD":

        execute_trade(signal, current_price)


    # ----------------------------------
    # Step 11: Backtest strategy
    # ----------------------------------

    print("\nRunning backtest simulation...")

    backtest_results = backtest_strategy(
        df_features,
        lr_model,
        rf_model,
        generate_signal
    )

    print("Backtest completed")


    print("\n----------------------------------")
    print("Pipeline completed")
    print("----------------------------------\n")


    return {

        "current_price": current_price,

        "prediction": prediction,

        "confidence": confidence,

        "signal": signal,

        "sentiment": sentiment_score,

        "telegram_messages": len(telegram_messages),

        "backtest": backtest_results
    }
