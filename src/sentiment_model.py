import numpy as np
from transformers import pipeline

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")


def score_message(text):
    """
    Score a single message using transformer sentiment analysis.
    Returns a value between -1 and +1.
    """

    try:

        result = sentiment_pipeline(text)[0]

        label = result["label"]
        score = result["score"]

        if label.upper() == "POSITIVE":
            return score
        else:
            return -score

    except Exception:

        return 0


def keyword_sentiment_boost(text):
    """
    Adds crypto-specific sentiment weighting
    """

    bullish_keywords = [
        "buy",
        "long",
        "pump",
        "breakout",
        "bull",
        "moon"
    ]

    bearish_keywords = [
        "sell",
        "short",
        "dump",
        "bear",
        "crash"
    ]

    score = 0

    for word in bullish_keywords:
        if word in text:
            score += 0.1

    for word in bearish_keywords:
        if word in text:
            score -= 0.1

    return score


def analyze_sentiment(messages):
    """
    Analyze sentiment across a list of Telegram messages.
    """

    if not messages:
        return 0

    scores = []

    for text in messages:

        transformer_score = score_message(text)

        keyword_score = keyword_sentiment_boost(text)

        final_score = transformer_score + keyword_score

        scores.append(final_score)

    sentiment = np.mean(scores)

    sentiment = max(min(sentiment, 1), -1)

    return float(sentiment)
