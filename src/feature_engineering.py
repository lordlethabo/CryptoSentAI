import pandas as pd


def build_features(df, sentiment_score):
    """
    Build machine learning features from market data
    and sentiment signal.
    """

    df = df.copy()

    # Add sentiment as a feature
    df["sentiment"] = sentiment_score

    # Price returns
    df["price_change"] = df["close"].pct_change()

    # Rolling statistics
    df["rolling_mean_3"] = df["close"].rolling(window=3).mean()
    df["rolling_mean_7"] = df["close"].rolling(window=7).mean()

    # Volatility
    df["volatility"] = df["close"].rolling(window=5).std()

    # Momentum
    df["momentum"] = df["close"] - df["close"].shift(3)

    # Moving averages
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()

    # Exponential moving average
    df["ema_10"] = df["close"].ewm(span=10).mean()

    # Relative Strength Index
    df["rsi"] = calculate_rsi(df["close"], period=14)

    # Target variable (next step price)
    df["target"] = df["close"].shift(-1)

    df = df.dropna()

    df.reset_index(drop=True, inplace=True)

    return df


def calculate_rsi(series, period=14):
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


def get_feature_columns():
    """
    List of ML features used for training.
    """

    return [
        "close",
        "sentiment",
        "price_change",
        "rolling_mean_3",
        "rolling_mean_7",
        "volatility",
        "momentum",
        "sma_5",
        "sma_10",
        "ema_10",
        "rsi"
    ]
