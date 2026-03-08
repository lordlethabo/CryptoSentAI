import numpy as np
import pandas as pd

from src.feature_engineering import get_feature_columns


def backtest_strategy(df, lr_model, rf_model, signal_function,
                      initial_capital=10000):

    capital = initial_capital
    asset = 0

    trade_history = []
    portfolio_values = []

    features = get_feature_columns()

    for i in range(20, len(df) - 1):

        train = df.iloc[:i]

        X_train = train[features]
        y_train = train["target"]

        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        row = df.iloc[i:i+1]

        X_test = row[features]

        lr_pred = lr_model.predict(X_test)[0]
        rf_pred = rf_model.predict(X_test)[0]

        prediction = (lr_pred + rf_pred) / 2

        price = row["close"].values[0]

        signal = signal_function(prediction, train)

        if signal == "BUY" and capital > 0:

            asset = capital / price
            capital = 0

            trade_history.append({
                "type": "BUY",
                "price": price
            })

        elif signal == "SELL" and asset > 0:

            capital = asset * price
            asset = 0

            trade_history.append({
                "type": "SELL",
                "price": price
            })

        portfolio_value = capital + asset * price

        portfolio_values.append(portfolio_value)

    final_price = df["close"].iloc[-1]

    final_value = capital + asset * final_price

    buy_hold_value = initial_capital * (final_price / df["close"].iloc[0])

    strategy_return = (final_value - initial_capital) / initial_capital

    buy_hold_return = (buy_hold_value - initial_capital) / initial_capital

    stats = calculate_statistics(portfolio_values, trade_history)

    return {
        "initial_capital": initial_capital,
        "final_portfolio_value": final_value,
        "buy_hold_value": buy_hold_value,
        "strategy_return_percent": strategy_return * 100,
        "buy_hold_return_percent": buy_hold_return * 100,
        "trade_history": trade_history,
        "portfolio_history": portfolio_values,
        "statistics": stats
    }


def calculate_statistics(portfolio_values, trades):

    if len(portfolio_values) == 0:

        return {}

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    avg_return = np.mean(returns) if len(returns) > 0 else 0

    volatility = np.std(returns) if len(returns) > 0 else 0

    sharpe_ratio = avg_return / volatility if volatility > 0 else 0

    max_drawdown = calculate_max_drawdown(portfolio_values)

    wins = 0
    losses = 0

    for i in range(1, len(trades)):

        if trades[i]["type"] == "SELL":

            if trades[i]["price"] > trades[i-1]["price"]:
                wins += 1
            else:
                losses += 1

    total_trades = wins + losses

    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }


def calculate_max_drawdown(portfolio_values):

    peak = portfolio_values[0]

    max_dd = 0

    for value in portfolio_values:

        if value > peak:
            peak = value

        dd = (peak - value) / peak

        if dd > max_dd:
            max_dd = dd

    return max_dd
