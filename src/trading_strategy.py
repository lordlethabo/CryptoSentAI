def generate_signal(predicted_price, df,
                    buy_threshold=0.02,
                    sell_threshold=0.02):
    """
    Generate a trading signal based on predicted price movement.

    Parameters
    ----------
    predicted_price : float
        Model predicted price
    df : DataFrame
        Feature dataframe
    buy_threshold : float
        Minimum upward movement to trigger buy
    sell_threshold : float
        Minimum downward movement to trigger sell

    Returns
    -------
    str
        BUY, SELL, or HOLD
    """

    current_price = df["close"].iloc[-1]

    change = (predicted_price - current_price) / current_price

    if change > buy_threshold:
        return "BUY"

    elif change < -sell_threshold:
        return "SELL"

    else:
        return "HOLD"


def position_size(capital, risk_per_trade=0.02):
    """
    Calculate position size based on risk management.
    """

    return capital * risk_per_trade


def apply_stop_loss(entry_price, current_price, stop_loss_pct=0.03):
    """
    Stop loss check.
    """

    loss = (entry_price - current_price) / entry_price

    if loss > stop_loss_pct:
        return True

    return False


def apply_take_profit(entry_price, current_price, take_profit_pct=0.05):
    """
    Take profit condition.
    """

    gain = (current_price - entry_price) / entry_price

    if gain > take_profit_pct:
        return True

    return False
