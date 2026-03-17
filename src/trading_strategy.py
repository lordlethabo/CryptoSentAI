%%writefile src/trading_strategy.py

import os
from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Strategy configuration
# --------------------------------------------------

BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", 0.02))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", 0.02))

SENTIMENT_CONFIRMATION = float(os.getenv("SENTIMENT_CONFIRMATION", 0.2))

VOLATILITY_LIMIT = float(os.getenv("VOLATILITY_LIMIT", 0.06))

RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 70))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 30))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))


# --------------------------------------------------
# Trend detection
# --------------------------------------------------

def detect_trend(df):

    ema10 = df["ema_10"].iloc[-1]
    ema20 = df["ema_20"].iloc[-1]

    if ema10 > ema20:
        return "UPTREND"

    elif ema10 < ema20:
        return "DOWNTREND"

    else:
        return "SIDEWAYS"


# --------------------------------------------------
# Volatility check
# --------------------------------------------------

def high_volatility(df):

    volatility = df["volatility"].iloc[-1]

    return volatility > VOLATILITY_LIMIT


# --------------------------------------------------
# Signal generation
# --------------------------------------------------

def generate_signal(predicted_price, df):

    current_price = df["close"].iloc[-1]

    sentiment = df["sentiment"].iloc[-1]

    rsi = df["rsi"].iloc[-1]

    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]

    change_ratio = (predicted_price - current_price) / current_price

    trend = detect_trend(df)

    # avoid unstable markets
    if high_volatility(df):
        return "HOLD"

    # ------------------------------------------------
    # BUY conditions
    # ------------------------------------------------

    if (

        change_ratio > BUY_THRESHOLD and
        sentiment > SENTIMENT_CONFIRMATION and
        trend == "UPTREND" and
        rsi < RSI_OVERBOUGHT and
        macd > macd_signal

    ):

        return "BUY"


    # ------------------------------------------------
    # SELL conditions
    # ------------------------------------------------

    if (

        change_ratio < -SELL_THRESHOLD and
        sentiment < -SENTIMENT_CONFIRMATION and
        trend == "DOWNTREND" and
        rsi > RSI_OVERSOLD and
        macd < macd_signal

    ):

        return "SELL"


    return "HOLD"


# --------------------------------------------------
# Position sizing
# --------------------------------------------------

def calculate_position_size(capital):

    position = capital * RISK_PER_TRADE

    return position


# --------------------------------------------------
# Stop loss (ATR based)
# --------------------------------------------------

def stop_loss_price(entry_price, df):

    atr = df["atr"].iloc[-1]

    stop_loss = entry_price - (atr * 1.5)

    return stop_loss


# --------------------------------------------------
# Take profit
# --------------------------------------------------

def take_profit_price(entry_price, df):

    atr = df["atr"].iloc[-1]

    take_profit = entry_price + (atr * 3)

    return take_profit