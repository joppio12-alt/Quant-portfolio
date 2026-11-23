"""
signals.py
Computes the indicators the strategy rules are built on: trend (SMA),
trend strength (ADX), momentum (RSI, MACD, relative strength), and
volatility (ATR).
"""

import pandas as pd
import yfinance as yf


def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = yf.Ticker(ticker).history(period=period)
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["sma50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = true_range.rolling(window=14).mean()

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_wilder = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_wilder
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_wilder

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return df


def add_relative_strength(df: pd.DataFrame, benchmark: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = df.copy()
    stock_return = df["Close"].pct_change(window)
    benchmark_close = benchmark["Close"].reindex(df.index, method="ffill")
    benchmark_return = benchmark_close.pct_change(window)
    df["rel_strength_60d"] = (stock_return - benchmark_return) * 100
    return df


if __name__ == "__main__":
    ticker = "ASML.AS"
    benchmark_ticker = "^AEX"

    print(f"Pulling price history for {ticker} and benchmark {benchmark_ticker}...")
    prices = get_price_history(ticker)
    benchmark_prices = get_price_history(benchmark_ticker)

    print("Computing indicators...")
    result = add_indicators(prices)
    result = add_relative_strength(result, benchmark_prices)

    print(result[["Close", "sma50", "rsi14", "adx14", "rel_strength_60d"]].tail(10))