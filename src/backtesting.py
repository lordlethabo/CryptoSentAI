%%writefile src/backtesting.py

import numpy as np
import pandas as pd

from src.feature_engineering import get_feature_columns


# --------------------------------------------------
# Backtesting engine
# --------------------------------------------------

def backtest_strategy(
    df,
    lr_model,
    rf_model,
    gb_model,
    signal_function,
    initial_capital=10000,
    fee=0.001,
    slippage=0.0005
):

    capital = initial_capital
    asset = 0

    trade_history = []
    portfolio_values = []

    features = get_feature_columns()

    window = 200

    for i in range(window, len(df) - 1):

        train = df.iloc[i-window:i]

        X_train = train[features]
        y_train = train["target"]

        # rolling retrain
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)

        row = df.iloc[i:i+1]

        X_test = row[features]

        lr_pred = lr_model.predict(X_test)[0]
        rf_pred = rf_model.predict(X_test)[0]
        gb_pred = gb_model.predict(X_test)[0]

        prediction = (

            (lr_pred * 0.2) +
            (rf_pred * 0.3) +
            (gb_pred * 0.5)

        )

        price = row["close"].values[0]

        signal = signal_function(prediction, train)

        # --------------------------------------------------
        # BUY
        # --------------------------------------------------

        if signal == "BUY" and capital > 0:

            buy_price = price * (1 + slippage)

            asset = (capital * (1 - fee)) / buy_price

            capital = 0

            trade_history.append({
                "type": "BUY",
                "price": buy_price
            })

        # --------------------------------------------------
        # SELL
        # --------------------------------------------------

        elif signal == "SELL" and asset > 0:

            sell_price = price * (1 - slippage)

            capital = asset * sell_price * (1 - fee)

            asset = 0

            trade_history.append({
                "type": "SELL",
                "price": sell_price
            })

        portfolio_value = capital + asset * price

        portfolio_values.append(portfolio_value)

    final_price = df["close"].iloc[-1]

    final_value = capital + asset * final_price

    buy_hold_value = initial_capital * (
        final_price / df["close"].iloc[0]
    )

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


# --------------------------------------------------
# Strategy statistics
# --------------------------------------------------

def calculate_statistics(portfolio_values, trades):

    if len(portfolio_values) == 0:

        return {}

    portfolio_values = np.array(portfolio_values)

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    avg_return = np.mean(returns) if len(returns) > 0 else 0

    volatility = np.std(returns) if len(returns) > 0 else 0

    sharpe_ratio = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0

    max_drawdown = calculate_max_drawdown(portfolio_values)

    wins = 0
    losses = 0

    for i in range(1, len(trades)):

        if trades[i]["type"] == "SELL":

            entry = trades[i-1]["price"]
            exit_price = trades[i]["price"]

            if exit_price > entry:
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


# --------------------------------------------------
# Max drawdown
# --------------------------------------------------

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