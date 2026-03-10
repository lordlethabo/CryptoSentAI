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

    if volatility > VOLATILITY_LIMIT:

        return True

    return False


# --------------------------------------------------
# Signal generation
# --------------------------------------------------

def generate_signal(predicted_price, df):

    current_price = df["close"].iloc[-1]

    sentiment = df["sentiment"].iloc[-1]

    change_ratio = (predicted_price - current_price) / current_price

    trend = detect_trend(df)

    if high_volatility(df):

        return "HOLD"

    # BUY conditions
    if (

        change_ratio > BUY_THRESHOLD and

        sentiment > SENTIMENT_CONFIRMATION and

        trend == "UPTREND"

    ):

        return "BUY"

    # SELL conditions
    elif (

        change_ratio < -SELL_THRESHOLD and

        sentiment < -SENTIMENT_CONFIRMATION and

        trend == "DOWNTREND"

    ):

        return "SELL"

    else:

        return "HOLD"


# --------------------------------------------------
# Position sizing
# --------------------------------------------------

def calculate_position_size(capital, risk_per_trade=0.02):

    position = capital * risk_per_trade

    return position


# --------------------------------------------------
# Stop loss
# --------------------------------------------------

def stop_loss_trigger(entry_price, current_price, stop_loss_pct=0.03):

    loss = (entry_price - current_price) / entry_price

    if loss >= stop_loss_pct:

        return True

    return False


# --------------------------------------------------
# Take profit
# --------------------------------------------------

def take_profit_trigger(entry_price, current_price, take_profit_pct=0.05):

    gain = (current_price - entry_price) / entry_price

    if gain >= take_profit_pct:

        return True

    return False
