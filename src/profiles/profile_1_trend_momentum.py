"""
profile_1_trend_momentum.py
Profile 1: trend-following with a momentum trigger.
Entry requires an established uptrend (price above a rising 50-day SMA)
plus a fresh momentum signal (RSI crossing above 50, or MACD crossing
above its signal line). Stop is set a fixed ATR multiple below price.
"""

import pandas as pd
from core.signals import get_price_history, add_indicators
from core.risk import add_position_size

PROFILE_1_CONFIG = {
    "trend_lookback": 10,
    "rsi_threshold": 50,
    "atr_stop_multiplier": 2.0,
}


def add_entry_signal(df: pd.DataFrame, config: dict = PROFILE_1_CONFIG) -> pd.DataFrame:
    df = df.copy()

    sma_rising = df["sma50"] > df["sma50"].shift(config["trend_lookback"])
    price_above_sma = df["Close"] > df["sma50"]
    trend_ok = price_above_sma & sma_rising

    rsi_cross_up = (df["rsi14"] >= config["rsi_threshold"]) & (df["rsi14"].shift(1) < config["rsi_threshold"])
    macd_cross_up = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    momentum_trigger = rsi_cross_up | macd_cross_up

    df["entry_signal"] = trend_ok & momentum_trigger
    df["stop_price"] = df["Close"] - config["atr_stop_multiplier"] * df["atr14"]

    return df


if __name__ == "__main__":
    ticker = "ASML.AS"
    prices = get_price_history(ticker)
    result = add_indicators(prices)
    result = add_entry_signal(result)
    result = add_position_size(result, portfolio_value=20000)

    signals_found = result[result["entry_signal"]]
    print(f"Entry signals found for {ticker} in the last year:")
    print(signals_found[["Close", "sma50", "rsi14", "macd", "macd_signal", "stop_price", "position_shares", "position_cost"]])