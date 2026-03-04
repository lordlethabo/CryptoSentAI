from binance.client import Client
import pandas as pd


def fetch_crypto_prices(api_key, api_secret, symbol="BTCUSDT", days=60):
    """
    Fetch historical cryptocurrency prices from Binance.

    Parameters
    ----------
    api_key : str
        Binance API key
    api_secret : str
        Binance API secret
    symbol : str
        Trading pair (default BTCUSDT)
    days : int
        Number of historical days to fetch

    Returns
    -------
    DataFrame
        Clean dataframe containing timestamp and close price
    """

    try:

        client = Client(api_key, api_secret)

        klines = client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1DAY,
            f"{days} day ago UTC"
        )

        df = pd.DataFrame(klines, columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        df = df.sort_values("timestamp")

        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:

        print("Error fetching Binance data:", e)

        return None


def validate_market_data(df):
    """
    Ensures the dataframe has required fields for ML pipeline.
    """

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return True
