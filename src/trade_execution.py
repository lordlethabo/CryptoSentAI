import os
from dotenv import load_dotenv
from binance.client import Client
import datetime

load_dotenv()


# --------------------------------------------------
# Environment configuration
# --------------------------------------------------

BROKER = os.getenv("BROKER", "paper")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")

TRADE_QUANTITY = float(os.getenv("TRADE_QUANTITY", 0.001))

QX_EMAIL = os.getenv("QX_EMAIL")
QX_PASSWORD = os.getenv("QX_PASSWORD")

TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", 10))


# --------------------------------------------------
# Binance Client
# --------------------------------------------------

binance_client = None

if BINANCE_API_KEY and BINANCE_API_SECRET:

    binance_client = Client(
        BINANCE_API_KEY,
        BINANCE_API_SECRET
    )


# --------------------------------------------------
# Paper trading portfolio
# --------------------------------------------------

portfolio = {

    "cash": 10000,

    "asset": 0
}


# --------------------------------------------------
# Trade logger
# --------------------------------------------------

def log_trade(signal, price):

    timestamp = datetime.datetime.now()

    log = {

        "time": timestamp,

        "signal": signal,

        "price": price

    }

    print("Trade Log:", log)

    return log


# --------------------------------------------------
# Paper trading execution
# --------------------------------------------------

def execute_paper_trade(signal, price):

    global portfolio

    if signal == "BUY" and portfolio["cash"] > 0:

        portfolio["asset"] = portfolio["cash"] / price

        portfolio["cash"] = 0

        print("Paper BUY executed")

    elif signal == "SELL" and portfolio["asset"] > 0:

        portfolio["cash"] = portfolio["asset"] * price

        portfolio["asset"] = 0

        print("Paper SELL executed")

    else:

        print("Paper HOLD")

    log_trade(signal, price)

    return portfolio


# --------------------------------------------------
# Binance execution
# --------------------------------------------------

def execute_binance_trade(signal):

    if not binance_client:

        print("Binance client not configured")

        return None

    if signal == "BUY":

        order = binance_client.order_market_buy(

            symbol=TRADE_SYMBOL,

            quantity=TRADE_QUANTITY

        )

        print("Binance BUY executed")

        return order

    elif signal == "SELL":

        order = binance_client.order_market_sell(

            symbol=TRADE_SYMBOL,

            quantity=TRADE_QUANTITY

        )

        print("Binance SELL executed")

        return order

    else:

        print("No Binance trade executed")

        return None


# --------------------------------------------------
# Broker QX execution
# --------------------------------------------------

def execute_qx_trade(signal):

    """
    Placeholder adapter for QX broker.
    Real implementation may require websocket
    or browser automation depending on API.
    """

    if signal == "BUY":

        print("QX CALL trade executed")

        print("Trade amount:", TRADE_AMOUNT)

    elif signal == "SELL":

        print("QX PUT trade executed")

        print("Trade amount:", TRADE_AMOUNT)

    else:

        print("QX HOLD")

    return {

        "broker": "qx",

        "signal": signal,

        "amount": TRADE_AMOUNT

    }


# --------------------------------------------------
# Main execution router
# --------------------------------------------------

def execute_trade(signal, price):

    if BROKER == "paper":

        return execute_paper_trade(signal, price)

    elif BROKER == "binance":

        return execute_binance_trade(signal)

    elif BROKER == "qx":

        return execute_qx_trade(signal)

    else:

        print("Unknown broker:", BROKER)

        return None
