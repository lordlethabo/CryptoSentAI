import os
from dotenv import load_dotenv

load_dotenv()

# Thresholds for signal generation
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", 0.02))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", 0.02))


def generate_signal(predicted_price, df):
    """
    Generate BUY / SELL / HOLD signal
    based on predicted price movement.
    """

    current_price = df["close"].iloc[-1]

    change_ratio = (predicted_price - current_price) / current_price

    if change_ratio > BUY_THRESHOLD:

        return "BUY"

    elif change_ratio < -SELL_THRESHOLD:

        return "SELL"

    else:

        return "HOLD"


def calculate_position_size(capital, risk_per_trade=0.02):
    """
    Determine position size based on risk management.
    """

    position = capital * risk_per_trade

    return position


def stop_loss_trigger(entry_price, current_price, stop_loss_pct=0.03):
    """
    Determine if stop loss should trigger.
    """

    loss = (entry_price - current_price) / entry_price

    if loss >= stop_loss_pct:

        return True

    return False


def take_profit_trigger(entry_price, current_price, take_profit_pct=0.05):
    """
    Determine if take profit should trigger.
    """

    gain = (current_price - entry_price) / entry_price

    if gain >= take_profit_pct:

        return True

    return False
