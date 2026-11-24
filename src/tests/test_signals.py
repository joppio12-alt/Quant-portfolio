import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from core.signals import add_indicators, add_relative_strength


def make_synthetic_prices(closes):
    """Build a minimal OHLC dataframe from a list of closing prices."""
    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 1 for c in closes],
        "Low": [c - 1 for c in closes],
        "Close": closes,
    })
    return df


def test_sma_on_linear_series():
    closes = list(range(1, 101))  # 1, 2, 3, ..., 100
    df = add_indicators(make_synthetic_prices(closes))
    # For a straight line, the 50-day average at day 50 is the average of 1..50
    assert df["sma50"].iloc[49] == pytest_approx(sum(range(1, 51)) / 50)


def test_rsi_pure_uptrend_is_100():
    closes = list(range(1, 31))  # strictly increasing, no down-days at all
    df = add_indicators(make_synthetic_prices(closes))
    assert df["rsi14"].iloc[-1] == 100


def test_rsi_pure_downtrend_is_0():
    closes = list(range(30, 0, -1))  # strictly decreasing, no up-days at all
    df = add_indicators(make_synthetic_prices(closes))
    assert df["rsi14"].iloc[-1] == 0


def test_atr_constant_range():
    # Every day has the exact same high-low range of 2
    closes = [100] * 30
    df = make_synthetic_prices(closes)
    df = add_indicators(df)
    # ATR should settle on exactly that constant range
    assert round(df["atr14"].iloc[-1], 2) == 2.0


def test_adx_stays_within_bounds():
    closes = [100 + (i % 5) for i in range(60)]  # choppy, no real trend
    df = add_indicators(make_synthetic_prices(closes))
    adx_values = df["adx14"].dropna()
    assert (adx_values >= 0).all() and (adx_values <= 100).all()


def test_relative_strength_detects_outperformance():
    stock = make_synthetic_prices([100 + i * 2 for i in range(70)])  # fast riser
    benchmark = make_synthetic_prices([100 + i for i in range(70)])  # slow riser
    df = add_relative_strength(stock, benchmark, window=60)
    # The stock rose faster than the benchmark, so relative strength should be positive
    assert df["rel_strength_60d"].iloc[-1] > 0


def pytest_approx(value, tol=0.01):
    class Approx:
        def __eq__(self, other):
            return abs(other - value) < tol
    return Approx()