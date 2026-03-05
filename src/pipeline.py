import os
import asyncio
from dotenv import load_dotenv

from src.data_ingestion import fetch_crypto_prices
from src.telegram_signals import fetch_signals
from src.sentiment_model import analyze_sentiment
from src.feature_engineering import build_features
from src.ml_models import train_models
from src.lstm_model import train_lstm, predict_lstm
from src.trading_strategy import generate_signal
from src.backtesting import backtest_strategy
from src.confidence import calculate_confidence


# ======================================
# LOAD ENVIRONMENT VARIABLES
# ======================================

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.80))

CHANNEL_1 = os.getenv("TELEGRAM_CHANNEL_1")
CHANNEL_2 = os.getenv("TELEGRAM_CHANNEL_2")

TELEGRAM_LIMIT = int(os.getenv("TELEGRAM_MESSAGE_LIMIT", 50))


# ======================================
# TELEGRAM MESSAGE COLLECTION
# ======================================

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


# ======================================
# MAIN PIPELINE
# ======================================

def run_pipeline(symbol=SYMBOL):

    print("\nStarting CryptoSentAI pipeline\n")

    # ----------------------------------
    # STEP 1: FETCH MARKET DATA
    # ----------------------------------

    print("Fetching market data...")

    df_price = fetch_crypto_prices(
        BINANCE_API_KEY,
        BINANCE_API_SECRET,
        symbol
    )

    if df_price is None or df_price.empty:
        raise ValueError("Market data fetch failed")

    print("Market data rows:", len(df_price))


    # ----------------------------------
    # STEP 2: COLLECT TELEGRAM SIGNALS
    # ----------------------------------

    print("Fetching Telegram signals...")

    telegram_messages = asyncio.run(
        collect_telegram_messages()
    )

    print("Messages collected:", len(telegram_messages))


    # ----------------------------------
    # STEP 3: SENTIMENT ANALYSIS
    # ----------------------------------

    print("Running sentiment analysis...")

    sentiment_score = analyze_sentiment(
        telegram_messages
    )

    print("Sentiment score:", sentiment_score)


    # ----------------------------------
    # STEP 4: FEATURE ENGINEERING
    # ----------------------------------

    print("Building features...")

    df_features = build_features(
        df_price,
        sentiment_score
    )


    # ----------------------------------
    # STEP 5: TRAIN ML MODELS
    # ----------------------------------

    print("Training ML models...")

    lr_model, rf_model = train_models(df_features)

    latest = df_features.iloc[-1:]

    X_latest = latest.drop(columns=["target"])

    lr_prediction = lr_model.predict(X_latest)[0]
    rf_prediction = rf_model.predict(X_latest)[0]

    ml_prediction = (lr_prediction + rf_prediction) / 2


    # ----------------------------------
    # STEP 6: LSTM FORECAST
    # ----------------------------------

    print("Training LSTM model...")

    lstm_model, scaler = train_lstm(df_price)

    lstm_prediction = predict_lstm(
        lstm_model,
        scaler,
        df_price
    )


    # ----------------------------------
    # STEP 7: ENSEMBLE PREDICTION
    # ----------------------------------

    prediction = (ml_prediction + lstm_prediction) / 2

    current_price = df_features["close"].iloc[-1]

    print("Current price:", current_price)
    print("Predicted price:", prediction)


    # ----------------------------------
    # STEP 8: CONFIDENCE CALCULATION
    # ----------------------------------

    confidence = calculate_confidence(
        lr_prediction,
        rf_prediction,
        lstm_prediction,
        current_price,
        sentiment_score
    )

    print("Confidence:", confidence)


    # ----------------------------------
    # STEP 9: GENERATE TRADING SIGNAL
    # ----------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        signal = generate_signal(
            prediction,
            df_features
        )

        print("Signal:", signal)

    else:

        signal = "HOLD"

        print("Confidence below threshold")


    # ----------------------------------
    # STEP 10: BACKTEST STRATEGY
    # ----------------------------------

    print("Running backtest...")

    backtest_results = backtest_strategy(
        df_features,
        lr_model,
        rf_model,
        generate_signal
    )


    print("\nPipeline finished\n")


    return {

        "current_price": current_price,

        "prediction": prediction,

        "confidence": confidence,

        "signal": signal,

        "sentiment": sentiment_score,

        "telegram_messages": len(telegram_messages),

        "backtest": backtest_results
    }
