%%writefile src/__init__.py
"""
CryptoSentAI source package
"""

# Expose key modules for easy import
from .data_ingestion import fetch_crypto_prices
from .feature_engineering import build_features
from .sentiment_model import analyze_sentiment
from .ml_models import train_models, predict_next_price