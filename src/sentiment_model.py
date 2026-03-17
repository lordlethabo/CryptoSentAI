%%writefile src/sentiment_model.py

import numpy as np
import re

from transformers import pipeline


# -----------------------------------------------------
# Load financial sentiment model
# -----------------------------------------------------

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


# -----------------------------------------------------
# Text cleaner
# -----------------------------------------------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = text.strip()

    return text


# -----------------------------------------------------
# Keyword sentiment boost
# -----------------------------------------------------

def keyword_sentiment_boost(text):

    bullish_keywords = [
        "buy",
        "long",
        "bull",
        "breakout",
        "pump",
        "moon",
        "support"
    ]

    bearish_keywords = [
        "sell",
        "short",
        "bear",
        "dump",
        "crash",
        "resistance"
    ]

    score = 0

    for word in bullish_keywords:

        if word in text:

            score += 0.08

    for word in bearish_keywords:

        if word in text:

            score -= 0.08

    return score


# -----------------------------------------------------
# Batch sentiment scoring
# -----------------------------------------------------

def batch_score_messages(messages):

    cleaned = [clean_text(m)[:512] for m in messages]

    try:

        results = sentiment_pipeline(cleaned)

    except Exception:

        return [0] * len(messages)

    scores = []

    for r in results:

        label = r["label"].upper()

        score = r["score"]

        if label == "POSITIVE":
            scores.append(score)

        elif label == "NEGATIVE":
            scores.append(-score)

        else:
            scores.append(0)

    return scores


# -----------------------------------------------------
# Main sentiment analysis
# -----------------------------------------------------

def analyze_sentiment(messages):

    if not messages:
        return 0

    # Remove duplicates
    messages = list(set(messages))

    transformer_scores = batch_score_messages(messages)

    final_scores = []

    for text, transformer_score in zip(messages, transformer_scores):

        keyword_score = keyword_sentiment_boost(text)

        score = transformer_score + keyword_score

        final_scores.append(score)

    sentiment = np.mean(final_scores)

    sentiment = max(min(sentiment, 1), -1)

    return float(sentiment)