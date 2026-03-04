
import pandas as pd


def backtest_strategy(df, lr_model, rf_model, signal_function, initial_capital=10000):

    """
    Runs historical trading simulation using the trained models.
    """

    capital = initial_capital
    crypto = 0

    portfolio_history = []
    signals = []

    for i in range(10, len(df) - 1):

        window = df.iloc[:i]

        X = window[["close", "sentiment", "price_change", "rolling_mean"]]
        y = window["target"]

        lr_model.fit(X, y)
        rf_model.fit(X, y)

        current_row = df.iloc[i:i+1]

        features = current_row[["close", "sentiment", "price_change", "rolling_mean"]]

        pred_lr = lr_model.predict(features.values)[0]
        pred_rf = rf_model.predict(features.values)[0]

        prediction = (pred_lr + pred_rf) / 2

        current_price = current_row["close"].values[0]

        signal = signal_function(prediction, window)

        signals.append(signal)

        if signal == "BUY" and capital > 0:

            crypto = capital / current_price
            capital = 0

        elif signal == "SELL" and crypto > 0:

            capital = crypto * current_price
            crypto = 0

        portfolio_value = capital + crypto * current_price

        portfolio_history.append(portfolio_value)

    final_price = df["close"].iloc[-1]

    final_portfolio_value = capital + crypto * final_price

    buy_hold_value = initial_capital * (final_price / df["close"].iloc[0])

    strategy_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100

    results = {
        "final_portfolio_value": final_portfolio_value,
        "buy_hold_value": buy_hold_value,
        "strategy_return_percent": strategy_return,
        "buy_hold_return_percent": buy_hold_return,
        "portfolio_history": portfolio_history,
        "signals": signals
    }

    return results
