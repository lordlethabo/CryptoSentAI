import numpy as np


def normalize(value, min_val, max_val):
    """
    Normalize a value between 0 and 1.
    """

    if max_val - min_val == 0:
        return 0

    return (value - min_val) / (max_val - min_val)


def model_agreement_score(predictions):
    """
    Measures how closely models agree with each other.
    Lower standard deviation = higher agreement.
    """

    mean_pred = np.mean(predictions)

    if mean_pred == 0:
        return 0

    std_dev = np.std(predictions)

    agreement = 1 - (std_dev / abs(mean_pred))

    agreement = max(0, min(agreement, 1))

    return agreement


def price_movement_strength(prediction, current_price):
    """
    Measures strength of predicted move.
    Larger moves = stronger signals.
    """

    move = abs(prediction - current_price) / current_price

    move_score = min(move * 5, 1)

    return move_score


def sentiment_strength(sentiment):
    """
    Convert sentiment score into confidence component.
    """

    sentiment_score = abs(sentiment)

    sentiment_score = min(sentiment_score, 1)

    return sentiment_score


def calculate_confidence(
        lr_prediction,
        rf_prediction,
        lstm_prediction,
        current_price,
        sentiment_score
):
    """
    Calculate final AI confidence score.
    """

    predictions = np.array([
        lr_prediction,
        rf_prediction,
        lstm_prediction
    ])

    avg_prediction = np.mean(predictions)

    agreement = model_agreement_score(predictions)

    movement = price_movement_strength(
        avg_prediction,
        current_price
    )

    sentiment = sentiment_strength(
        sentiment_score
    )

    confidence = (
        (agreement * 0.4) +
        (movement * 0.4) +
        (sentiment * 0.2)
    )

    confidence = max(0, min(confidence, 1))

    return confidence
