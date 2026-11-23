"""
risk.py
Portfolio-level risk policy: converts a stop price into a position size.
Shared across all profiles -- how much to buy once a profile has decided
what to buy is a portfolio rule, not a strategy opinion.
"""

import pandas as pd

DEFAULT_RISK_PCT = 0.01


def add_position_size(df: pd.DataFrame, portfolio_value: float, risk_pct: float = DEFAULT_RISK_PCT) -> pd.DataFrame:
    df = df.copy()
    risk_amount = portfolio_value * risk_pct
    stop_distance = df["Close"] - df["stop_price"]
    df["position_shares"] = (risk_amount / stop_distance).where(stop_distance > 0).apply(
        lambda x: int(x) if pd.notna(x) else 0
    )
    df["position_cost"] = df["position_shares"] * df["Close"]
    return df