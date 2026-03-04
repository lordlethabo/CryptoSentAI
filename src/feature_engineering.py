
import pandas as pd
import numpy as np


def build_features(df, sentiment_score):
    """
    Build machine learning features from raw crypto market data.

    Parameters
    ----------
    df : DataFrame
        Market data from Binance ingestion
    sentiment_score : float
        Aggregated sentiment score from NLP models

    Returns
    -------
    DataFrame
        Feature engineered dataset ready for ML models
    """

    df = df.copy()

    # Add sentiment feature
    df["sentiment"] = sentiment_score

    # Price change percentage
    df["price_change"] = df["close"].pct_change()

    # Rolling mean (short-term trend)
    df["rolling_mean"] = df["close"].rolling(window=3).mean()

    # Rolling standard deviation (volatility)
    df["volatility"] = df["close"].rolling(window=5).std()

    # Momentum indicator
    df["momentum"] = df["close"] - df["close"].shift(3)

    # Simple moving averages
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()

    # Exponential moving average
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    # Relative Strength Index (RSI)
    df["rsi"] = compute_rsi(df["close"])

    # Target variable (next price)
    df["target"] = df["close"].shift(-1)

    df = df.dropna()

    df.reset_index(drop=True, inplace=True)

    return df


def compute_rsi(series, period=14):
    """
    Compute Relative Strength Index (RSI).
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def normalize_features(df):
    """
    Normalize features for ML models.
    """

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    feature_cols = [
        "close",
        "sentiment",
        "price_change",
        "rolling_mean",
        "volatility",
        "momentum",
        "sma_5",
        "sma_10",
        "ema_10",
        "rsi"
    ]

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler
