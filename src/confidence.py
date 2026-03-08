import numpy as np


def model_agreement_score(predictions):
    """
    Measure agreement between multiple model predictions.
    Lower variance means higher agreement.
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
    Measure how significant the predicted move is.
    Small moves are usually noise.
    """

    movement = abs(prediction - current_price) / current_price

    score = min(movement * 5, 1)

    return score


def sentiment_strength(sentiment_score):
    """
    Convert sentiment score into a confidence contribution.
    """

    sentiment_value = abs(sentiment_score)

    sentiment_value = max(0, min(sentiment_value, 1))

    return sentiment_value


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

    sentiment = sentiment_strength(sentiment_score)

    confidence = (
        (agreement * 0.4) +
        (movement * 0.4) +
        (sentiment * 0.2)
    )

    confidence = max(0, min(confidence, 1))

    return confidence
