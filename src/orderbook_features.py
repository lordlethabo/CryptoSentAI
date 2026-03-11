import os
from dotenv import load_dotenv
from binance.client import Client
import numpy as np

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def fetch_orderbook(symbol=SYMBOL, depth=50):
    """
    Retrieve order book data from Binance.
    """

    book = client.get_order_book(symbol=symbol, limit=depth)

    bids = book["bids"]
    asks = book["asks"]

    bid_volume = sum(float(b[1]) for b in bids)
    ask_volume = sum(float(a[1]) for a in asks)

    return bid_volume, ask_volume


def orderbook_imbalance(symbol=SYMBOL):
    """
    Calculate order book imbalance.
    """

    bid_vol, ask_vol = fetch_orderbook(symbol)

    total = bid_vol + ask_vol

    if total == 0:
        return 0

    imbalance = (bid_vol - ask_vol) / total

    return imbalance
