import os
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from pycoingecko import CoinGeckoAPI

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

LOOKBACK_DAYS = int(os.getenv("PRICE_LOOKBACK_DAYS", 60))


def fetch_from_binance(symbol):

    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1DAY,
        f"{LOOKBACK_DAYS} day ago UTC"
    )

    df = pd.DataFrame(klines)

    df = df.iloc[:, 0:6]

    df.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    numeric_cols = ["open", "high", "low", "close", "volume"]

    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp")

    df.reset_index(drop=True, inplace=True)

    return df


def fetch_from_coingecko(symbol):

    cg = CoinGeckoAPI()

    coin = symbol.replace("USDT", "").lower()

    data = cg.get_coin_market_chart_by_id(
        coin,
        "usd",
        LOOKBACK_DAYS
    )

    prices = data["prices"]

    df = pd.DataFrame(prices, columns=["timestamp", "close"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 0

    df = df[[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]]

    return df


def fetch_crypto_prices(api_key, api_secret, symbol):

    try:

        print("Fetching data from Binance")

        df = fetch_from_binance(symbol)

        return df

    except Exception as e:

        print("Binance failed:", e)

        print("Using CoinGecko fallback")

        return fetch_from_coingecko(symbol)
