%%writefile src/__init__.py

"""
CryptoSentAI package initializer
"""

from .data_ingestion import fetch_crypto_prices
from .feature_engineering import build_features
from .ml_models import train_models, predict_next_price
from .lstm_model import train_lstm, predict_lstm
from .sentiment_model import analyze_sentiment
from .confidence import calculate_confidence
from .trading_strategy import generate_signal
from .backtesting import backtest_strategy
from .pipeline import run_pipeline