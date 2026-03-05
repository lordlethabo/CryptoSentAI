import os
from dotenv import load_dotenv
from src.pipeline import run_pipeline


def print_banner():

    print("\n==============================")
    print("      CryptoSentAI Engine     ")
    print("==============================\n")


def print_results(result):

    print("\n----- PIPELINE RESULTS -----\n")

    print("Current Price:", result["current_price"])
    print("Predicted Price:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Trading Signal:", result["signal"])
    print("Sentiment Score:", result["sentiment"])

    print("\n----- BACKTEST RESULTS -----\n")

    backtest = result["backtest"]

    print("Final Portfolio Value:", backtest.get("final_portfolio_value"))
    print("Strategy Return (%):", backtest.get("strategy_return_percent"))
    print("Buy & Hold Return (%):", backtest.get("buy_hold_return_percent"))

    print("\n-----------------------------\n")


def main():

    load_dotenv()

    symbol = os.getenv("BINANCE_SYMBOL", "BTCUSDT")

    print_banner()

    try:

        result = run_pipeline(symbol)

        print_results(result)

    except Exception as e:

        print("Pipeline execution failed")
        print("Error:", e)


if __name__ == "__main__":

    main()
