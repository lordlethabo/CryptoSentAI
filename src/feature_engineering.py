import pandas as pd
import numpy as np


# -----------------------------------------------------
# RSI
# -----------------------------------------------------

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# -----------------------------------------------------
# MACD
# -----------------------------------------------------

def calculate_macd(series):

    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    return macd, signal


# -----------------------------------------------------
# Bollinger Bands
# -----------------------------------------------------

def calculate_bollinger(series, window=20):

    sma = series.rolling(window).mean()

    std = series.rolling(window).std()

    upper = sma + (2 * std)

    lower = sma - (2 * std)

    return upper, lower


# -----------------------------------------------------
# VWAP
# -----------------------------------------------------

def calculate_vwap(df):

    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    return vwap


# -----------------------------------------------------
# Feature builder
# -----------------------------------------------------

def build_features(df, sentiment_score):

    df = df.copy()

    # sentiment feature
    df["sentiment"] = sentiment_score

    # price change
    df["price_change"] = df["close"].pct_change()

    # momentum
    df["momentum"] = df["close"] - df["close"].shift(3)

    # volatility
    df["volatility"] = df["close"].rolling(5).std()

    # moving averages
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()

    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()

    # trend strength
    df["trend_strength"] = (df["ema_10"] - df["ema_20"]) / df["close"]

    # RSI
    df["rsi"] = calculate_rsi(df["close"])

    # MACD
    macd, macd_signal = calculate_macd(df["close"])

    df["macd"] = macd
    df["macd_signal"] = macd_signal

    # Bollinger Bands
    upper, lower = calculate_bollinger(df["close"])

    df["bollinger_upper"] = upper
    df["bollinger_lower"] = lower

    # VWAP
    df["vwap"] = calculate_vwap(df)

    # volume indicators
    df["volume_change"] = df["volume"].pct_change()

    df["volume_ma"] = df["volume"].rolling(5).mean()

    # target
    df["target"] = df["close"].shift(-1)

    df = df.dropna()

    df.reset_index(drop=True, inplace=True)

    return df


# -----------------------------------------------------
# Feature list used by ML models
# -----------------------------------------------------

def get_feature_columns():

    return [

        "close",

        "sentiment",

        "price_change",

        "momentum",

        "volatility",

        "trend_strength",

        "sma_5",
        "sma_10",

        "ema_10",
        "ema_20",

        "rsi",

        "macd",
        "macd_signal",

        "bollinger_upper",
        "bollinger_lower",

        "vwap",

        "volume_change",
        "volume_ma"
    ]
