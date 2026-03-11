import pandas as pd
import os
from datetime import datetime


LOG_FILE = "trade_history.csv"


# --------------------------------------------------
# Initialize log file
# --------------------------------------------------

def initialize_log():

    if not os.path.exists(LOG_FILE):

        df = pd.DataFrame(columns=[

            "timestamp",
            "signal",
            "entry_price",
            "exit_price",
            "profit",
            "confidence"

        ])

        df.to_csv(LOG_FILE, index=False)


# --------------------------------------------------
# Record trade
# --------------------------------------------------

def record_trade(signal, entry_price, exit_price, confidence):

    profit = 0

    if exit_price is not None:

        profit = exit_price - entry_price

    row = {

        "timestamp": datetime.now(),

        "signal": signal,

        "entry_price": entry_price,

        "exit_price": exit_price,

        "profit": profit,

        "confidence": confidence

    }

    df = pd.read_csv(LOG_FILE)

    df = pd.concat([df, pd.DataFrame([row])])

    df.to_csv(LOG_FILE, index=False)


# --------------------------------------------------
# Calculate statistics
# --------------------------------------------------

def calculate_performance():

    if not os.path.exists(LOG_FILE):

        return {}

    df = pd.read_csv(LOG_FILE)

    total_trades = len(df)

    wins = len(df[df["profit"] > 0])

    losses = len(df[df["profit"] < 0])

    total_profit = df["profit"].sum()

    win_rate = wins / total_trades if total_trades > 0 else 0

    avg_profit = df["profit"].mean() if total_trades > 0 else 0

    return {

        "total_trades": total_trades,

        "wins": wins,

        "losses": losses,

        "win_rate": win_rate,

        "total_profit": total_profit,

        "avg_profit": avg_profit

    }
