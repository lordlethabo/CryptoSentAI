from transformers import pipeline
import numpy as np


# Load sentiment models once
distilbert_model = pipeline("sentiment-analysis")

finbert_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


def get_sample_texts():
    """
    Temporary sample texts.
    These will later be replaced by Telegram/Twitter ingestion.
    """

    texts = [

        "Bitcoin market looks bullish today",

        "Crypto investors are worried about regulation",

        "Institutional adoption of Bitcoin is increasing",

        "Market volatility is creating uncertainty",

        "Traders expect BTC to break resistance",

        "Crypto market crash fears return",

        "Ethereum and Bitcoin adoption continues to grow",

        "Investors are optimistic about blockchain technology",

        "Bearish pressure is building in the crypto market",

        "Many analysts believe BTC will rally soon"
    ]

    return texts


def score_text(text):
    """
    Compute sentiment score using two models.
    """

    result1 = distilbert_model(text)[0]
    result2 = finbert_model(text)[0]

    score1 = result1["score"] if result1["label"] == "POSITIVE" else -result1["score"]

    score2 = result2["score"] if result2["label"] == "positive" else -result2["score"]

    return (score1 + score2) / 2


def analyze_sentiment():
    """
    Run sentiment analysis on multiple texts and return
    a single aggregated sentiment score.
    """

    texts = get_sample_texts()

    scores = []

    for text in texts:

        try:

            score = score_text(text)

            scores.append(score)

        except Exception:

            continue

    if len(scores) == 0:

        return 0

    sentiment_score = np.mean(scores)

    return sentiment_score
