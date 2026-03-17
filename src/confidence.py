%%writefile src/confidence.py

import numpy as np


# ----------------------------------------------------
# Normalize value between 0 and 1
# ----------------------------------------------------

def normalize(value, min_val, max_val):

    if max_val - min_val == 0:
        return 0.5

    return (value - min_val) / (max_val - min_val)


# ----------------------------------------------------
# Calculate prediction strength
# ----------------------------------------------------

def prediction_strength(current_price, predicted_price):

    change = abs(predicted_price - current_price)

    strength = change / current_price

    return strength


# ----------------------------------------------------
# Volatility penalty
# ----------------------------------------------------

def volatility_penalty(df):

    if "returns" not in df.columns:
        return 0.0

    volatility = df["returns"].std()

    return min(volatility, 0.05)


# ----------------------------------------------------
# Main confidence calculation
# ----------------------------------------------------

def calculate_confidence(

        current_price,
        ensemble_prediction,
        sentiment_score,
        lstm_prediction=None,
        df=None
):

    # Prediction strength
    strength = prediction_strength(current_price, ensemble_prediction)

    # Normalize sentiment
    sentiment_norm = (sentiment_score + 1) / 2

    # LSTM contribution
    lstm_weight = 0.2
    lstm_component = 0

    if lstm_prediction is not None:

        lstm_strength = prediction_strength(current_price, lstm_prediction)

        lstm_component = lstm_strength * lstm_weight

    # Volatility penalty
    penalty = 0

    if df is not None:

        penalty = volatility_penalty(df)

    # Combine signals
    confidence = (

        (strength * 0.5) +
        (sentiment_norm * 0.3) +
        lstm_component -
        penalty

    )

    # Clamp between 0 and 1
    confidence = max(0, min(confidence, 1))

    return confidence