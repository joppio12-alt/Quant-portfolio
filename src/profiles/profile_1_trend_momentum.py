"""
profile_1_trend_momentum.py
Profile 1: trend-following with a momentum trigger.
Entry requires an established uptrend (price above a rising 50-day SMA)
plus a fresh momentum signal (RSI crossing above 50, or MACD crossing
above its signal line). Stop is set a fixed ATR multiple below price.

Also defines Profile 1's exit rules: a trailing stop (ratchets up with
the highest close since entry), a trend-break exit (price falls back
below the 50-day SMA), and a time-stop for positions that are neither
working nor stopped out after ~6 weeks. These exit rules, like the
entry rules, are specific to this profile -- a different profile would
define its own.
"""

import datetime

import pandas as pd
from core.signals import get_price_history, add_indicators
from core.risk import add_position_size

PROFILE_1_CONFIG = {
    "trend_lookback": 10,
    "rsi_threshold": 50,
    "atr_stop_multiplier": 2.0,
    "time_stop_trading_days": 30,  # ~6 weeks -- starting assumption, pending backtest validation
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


def trading_days_held(entry_date: str) -> int:
    entry = datetime.date.fromisoformat(entry_date)
    calendar_days = (datetime.date.today() - entry).days
    return int(calendar_days * 5 / 7)  # rough trading-day approximation, ignores holidays


def check_exit(position: dict, today_row, config: dict = PROFILE_1_CONFIG) -> dict:
    """
    Checks one open position against today's price data.
    today_row: latest row from add_indicators() output for this ticker
    (needs Close, sma50, atr14).

    Exit priority: trailing stop first (hard protective floor), then
    trend break (the thesis itself has failed), then time-stop (a
    stagnant position that's neither working nor stopped out).
    """
    close = today_row["Close"]
    sma50 = today_row["sma50"]

    highest_close = max(position.get("highest_close", position["entry_price"]), close)
    atr_at_entry = position.get("atr_at_entry", today_row["atr14"])
    trailing_stop = highest_close - config["atr_stop_multiplier"] * atr_at_entry

    if close <= trailing_stop:
        return {"exit": True, "reason": "trailing_stop", "exit_price": close, "highest_close": highest_close}

    if close < sma50:
        return {"exit": True, "reason": "trend_break", "exit_price": close, "highest_close": highest_close}

    days_held = trading_days_held(position["entry_date"])
    if days_held >= config["time_stop_trading_days"]:
        unrealized_pnl = (close - position["entry_price"]) * position["shares"]
        if unrealized_pnl <= 0:
            return {"exit": True, "reason": "time_stop", "exit_price": close, "highest_close": highest_close}

    return {"exit": False, "reason": None, "exit_price": None, "highest_close": highest_close}


if __name__ == "__main__":
    ticker = "ASML.AS"
    prices = get_price_history(ticker)
    result = add_indicators(prices)
    result = add_entry_signal(result)
    result = add_position_size(result, portfolio_value=20000)

    signals_found = result[result["entry_signal"]]
    print(f"Entry signals found for {ticker} in the last year:")
    print(signals_found[["Close", "sma50", "rsi14", "macd", "macd_signal", "stop_price", "position_shares", "position_cost"]])

    if not signals_found.empty:
        first_signal_date = signals_found.index[0]
        first_signal_row = signals_found.iloc[0]
        fake_position = {
            "entry_date": first_signal_date.date().isoformat(),
            "entry_price": first_signal_row["Close"],
            "shares": int(first_signal_row["position_shares"]),
            "highest_close": first_signal_row["Close"],
            "atr_at_entry": first_signal_row["atr14"],
        }
        latest_row = result.iloc[-1]
        exit_check = check_exit(fake_position, latest_row)
        print(f"\nExit check for a position opened on {first_signal_date.date()}: {exit_check}")