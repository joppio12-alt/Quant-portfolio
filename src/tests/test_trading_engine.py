"""
test_trading_engine.py
A consolidated regression suite covering every decision-layer piece of
the trading engine: position sizing, entry signals, exit conditions,
portfolio state, shortlist selection (caps + cash reserve), the
confirm-and-record / confirm-and-execute human-in-the-loop steps, and
rotation. Uses fabricated, deterministic data throughout -- no network
calls, no dependence on what the market happens to be doing today --
so this gives the same result every run and is meant to be the first
thing to run if something in the engine seems broken.

Run with: pytest tests/test_trading_engine.py -v
"""

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.risk import add_position_size
from core.portfolio import (
    add_position, remove_position, position_count, sector_counts,
    sector_capital, total_invested, held_tickers,
)
from profiles.profile_1_trend_momentum import add_entry_signal, check_exit
from core.daily_runner import (
    select_shortlist, confirm_and_record, check_open_positions,
    confirm_and_execute_exits, score_candidates, evaluate_rotation,
    confirm_and_execute_rotation,
)


# ---------------------------------------------------------------------
# Position sizing (core.risk)
# ---------------------------------------------------------------------

def test_position_sizing():
    df = pd.DataFrame({"Close": [100.0], "stop_price": [90.0]})
    result = add_position_size(df, portfolio_value=20000.0, risk_pct=0.01)
    # risk_amount = 200, stop_distance = 10 -> 20 shares
    assert result["position_shares"].iloc[0] == 20
    assert result["position_cost"].iloc[0] == 2000.0


def test_position_sizing_zero_stop_distance():
    # A degenerate case: stop at or above Close should size to 0, not crash
    df = pd.DataFrame({"Close": [100.0], "stop_price": [100.0]})
    result = add_position_size(df, portfolio_value=20000.0, risk_pct=0.01)
    assert result["position_shares"].iloc[0] == 0


# ---------------------------------------------------------------------
# Entry signal (profiles.profile_1_trend_momentum)
# ---------------------------------------------------------------------

def test_entry_signal_fires_on_trend_plus_momentum():
    index = pd.bdate_range("2026-01-01", periods=11)
    df = pd.DataFrame({
        "Close": [100] * 10 + [105],
        "sma50": np.linspace(95, 100, 11),   # rising -> trend_ok
        "rsi14": [40] * 10 + [55],            # crosses above 50 -> momentum
        "macd": [0.5] * 10 + [1.0],
        "macd_signal": [1.5] * 10 + [2.0],    # no macd cross, RSI alone should trigger
        "atr14": [3.0] * 11,
    }, index=index)

    result = add_entry_signal(df)
    last = result.iloc[-1]
    assert bool(last["entry_signal"]) is True
    assert abs(last["stop_price"] - (105 - 2 * 3.0)) < 1e-9


def test_entry_signal_does_not_fire_without_momentum():
    index = pd.bdate_range("2026-01-01", periods=11)
    df = pd.DataFrame({
        "Close": [100] * 10 + [105],
        "sma50": np.linspace(95, 100, 11),
        "rsi14": [40] * 11,                   # stays below 50, no cross
        "macd": [0.5] * 11,                   # stays below signal, no cross
        "macd_signal": [1.5] * 11,
        "atr14": [3.0] * 11,
    }, index=index)

    result = add_entry_signal(df)
    assert bool(result.iloc[-1]["entry_signal"]) is False


# ---------------------------------------------------------------------
# Exit conditions (profiles.profile_1_trend_momentum.check_exit)
# ---------------------------------------------------------------------

def test_check_exit_trailing_stop():
    position = {"entry_price": 100.0, "entry_date": "2026-01-01", "shares": 1,
                "highest_close": 100.0, "atr_at_entry": 2.0}
    today_row = pd.Series({"Close": 90.0, "sma50": 95.0, "atr14": 2.0})
    result = check_exit(position, today_row)
    assert result["exit"] is True
    assert result["reason"] == "trailing_stop"


def test_check_exit_trend_break():
    position = {"entry_price": 100.0, "entry_date": "2026-01-01", "shares": 1,
                "highest_close": 100.0, "atr_at_entry": 2.0}
    # trailing_stop = 100 - 2*2 = 96; Close=97 is above that, so the stop doesn't fire
    # Close=97 < sma50=100, so trend_break should fire instead
    today_row = pd.Series({"Close": 97.0, "sma50": 100.0, "atr14": 2.0})
    result = check_exit(position, today_row)
    assert result["exit"] is True
    assert result["reason"] == "trend_break"


def test_check_exit_time_stop_on_stagnant_position():
    old_date = (datetime.date.today() - datetime.timedelta(days=45)).isoformat()
    position = {"entry_price": 100.0, "entry_date": old_date, "shares": 1,
                "highest_close": 100.0, "atr_at_entry": 2.0}
    # Close == entry price (flat, not profitable) and well past the 30-trading-day mark
    today_row = pd.Series({"Close": 100.0, "sma50": 100.0, "atr14": 2.0})
    result = check_exit(position, today_row)
    assert result["exit"] is True
    assert result["reason"] == "time_stop"


def test_check_exit_time_stop_spares_a_profitable_stagnant_position():
    old_date = (datetime.date.today() - datetime.timedelta(days=45)).isoformat()
    position = {"entry_price": 100.0, "entry_date": old_date, "shares": 1,
                "highest_close": 105.0, "atr_at_entry": 2.0}
    # Past the time-stop window, but still ahead -- should be spared
    today_row = pd.Series({"Close": 103.0, "sma50": 100.0, "atr14": 2.0})
    result = check_exit(position, today_row)
    assert result["exit"] is False


def test_check_exit_no_exit_on_healthy_recent_position():
    recent_date = datetime.date.today().isoformat()
    position = {"entry_price": 100.0, "entry_date": recent_date, "shares": 1,
                "highest_close": 100.0, "atr_at_entry": 2.0}
    today_row = pd.Series({"Close": 110.0, "sma50": 100.0, "atr14": 2.0})
    result = check_exit(position, today_row)
    assert result["exit"] is False


# ---------------------------------------------------------------------
# Portfolio state (core.portfolio)
# ---------------------------------------------------------------------

def test_portfolio_state_add_and_remove_position():
    state = {"cash": 10000.0, "positions": []}
    state = add_position(state, ticker="X", sector="Technology", entry_date="2026-01-01",
                          entry_price=50.0, shares=10, stop_price=45.0,
                          track="global", atr_at_entry=1.5)

    assert position_count(state) == 1
    assert state["cash"] == 9500.0
    assert sector_counts(state) == {"Technology": 1}
    assert sector_capital(state) == {"Technology": 500.0}
    assert total_invested(state) == 500.0
    assert held_tickers(state) == {"X"}

    state = remove_position(state, "X", exit_price=55.0)
    assert position_count(state) == 0
    assert state["cash"] == 9500.0 + 550.0


# ---------------------------------------------------------------------
# Shortlist selection: position cap, sector cap, cash reserve (core.daily_runner)
# ---------------------------------------------------------------------

def test_select_shortlist_respects_sector_cap_and_cash_reserve():
    fake_portfolio = {"cash": 25000.0, "positions": []}
    fake_candidates = [
        {"ticker": "A", "sector": "Technology", "cost": 5000, "score": 2.0},
        {"ticker": "B", "sector": "Technology", "cost": 5000, "score": 1.8},  # blocked: sector cap
        {"ticker": "C", "sector": "Healthcare", "cost": 5000, "score": 1.5},
        {"ticker": "D", "sector": "Energy", "cost": 5000, "score": 1.2},
        {"ticker": "E", "sector": "Utilities", "cost": 5000, "score": 1.0},
        {"ticker": "F", "sector": "Financials", "cost": 5000, "score": 0.8},  # blocked: cash reserve
    ]
    shortlist, blocked = select_shortlist(fake_candidates, fake_portfolio, portfolio_value=25000.0)

    assert {c["ticker"] for c in shortlist} == {"A", "C", "D", "E"}
    assert {c["ticker"] for c in blocked} == {"B", "F"}
    assert sum(c["cost"] for c in shortlist) == 20000.0  # leaves 5000 >= the 2500 reserve floor


# ---------------------------------------------------------------------
# Confirm-and-record / confirm-and-execute-exits (human-in-the-loop steps)
# ---------------------------------------------------------------------

@patch("builtins.input", return_value="all")
def test_confirm_and_record_writes_approved_entry(mock_input):
    state = {"cash": 10000.0, "positions": []}
    shortlist = [
        {"ticker": "Y", "sector": "Energy", "close": 20.0, "stop_price": 18.0,
         "shares": 10, "cost": 200.0, "track": "global", "atr14": 1.0},
    ]
    state = confirm_and_record(shortlist, state, scenario="test_engine")
    assert held_tickers(state) == {"Y"}
    assert state["cash"] == 9800.0


@patch("builtins.input", return_value="none")
def test_confirm_and_record_respects_a_decline(mock_input):
    state = {"cash": 10000.0, "positions": []}
    shortlist = [
        {"ticker": "Y", "sector": "Energy", "close": 20.0, "stop_price": 18.0,
         "shares": 10, "cost": 200.0, "track": "global", "atr14": 1.0},
    ]
    state = confirm_and_record(shortlist, state, scenario="test_engine")
    assert held_tickers(state) == set()
    assert state["cash"] == 10000.0


@patch("builtins.input", return_value="all")
def test_confirm_and_execute_exits(mock_input):
    state = {"cash": 10000.0, "positions": [
        {"ticker": "Z", "sector": "Energy", "entry_date": "2026-01-01", "entry_price": 50.0,
         "shares": 5, "cost": 250.0, "stop_price": 45.0, "track": "global",
         "atr_at_entry": 1.0, "highest_close": 50.0},
    ]}
    exit_candidates = [{"ticker": "Z", "reason": "trend_break", "exit_price": 48.0,
                         "shares": 5, "proceeds": 240.0}]
    state = confirm_and_execute_exits(exit_candidates, state, scenario="test_engine")
    assert held_tickers(state) == set()
    assert state["cash"] == 10240.0


# ---------------------------------------------------------------------
# check_open_positions wiring (confirms the loop + add_indicators + check_exit
# integration, not just check_exit in isolation)
# ---------------------------------------------------------------------

def test_check_open_positions_detects_trend_break():
    index = pd.bdate_range("2026-01-01", periods=60)
    closes = pd.Series([100.0] * 59 + [97.0], index=index)
    prices = pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": 1_000_000,
    })

    state = {"cash": 10000.0, "positions": [
        {"ticker": "SYN", "sector": "Technology", "entry_date": "2026-01-01",
         "entry_price": 100.0, "shares": 1, "cost": 100.0, "stop_price": 95.0,
         "track": "global", "atr_at_entry": 2.0, "highest_close": 100.0},
    ]}
    price_data = {"SYN": prices}

    exit_candidates = check_open_positions(state, price_data)
    assert len(exit_candidates) == 1
    assert exit_candidates[0]["ticker"] == "SYN"
    assert exit_candidates[0]["reason"] == "trend_break"


# ---------------------------------------------------------------------
# Rotation: eligibility + margin threshold + swap execution
# (score_held_position is mocked -- its internals are already covered by
# the signals/entry-signal tests above; this isolates evaluate_rotation's
# own logic: who's eligible, who's weakest, does the margin clear the bar)
# ---------------------------------------------------------------------

@patch("core.daily_runner.score_held_position", return_value=(-0.5, 48.0))
@patch("builtins.input", return_value="y")
def test_rotation_fires_when_margin_clears_threshold(mock_input, mock_score_held):
    strong_candidate = {"ticker": "STRONG", "sector": "Technology", "track": "global",
                         "close": 50.0, "stop_price": 45.0, "shares": 10, "cost": 500.0,
                         "atr14": 1.5, "score": 1.5}  # margin vs mocked -0.5 is 2.0, clears 1.0

    old_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    state = {"cash": 10000.0, "positions": [
        {"ticker": "WEAK", "sector": "Technology", "entry_date": old_date, "entry_price": 30.0,
         "shares": 10, "cost": 300.0, "stop_price": 25.0, "track": "global",
         "atr_at_entry": 1.0, "highest_close": 30.0},
    ]}

    rotation = evaluate_rotation([strong_candidate], state, price_data={}, benchmarks={},
                                  scoring_params=(0, 1, 0, 1))
    assert rotation is not None
    assert rotation["buy"]["ticker"] == "STRONG"
    assert rotation["sell"]["ticker"] == "WEAK"

    state = confirm_and_execute_rotation(rotation, state, scenario="test_engine")
    assert held_tickers(state) == {"STRONG"}
    assert state["cash"] == 9980.0  # 10000 + 480 (sell proceeds) - 500 (buy cost)


@patch("core.daily_runner.score_held_position", return_value=(1.0, 48.0))
def test_rotation_does_not_fire_when_margin_is_too_small(mock_score_held):
    marginal_candidate = {"ticker": "MARGINAL", "sector": "Technology", "track": "global",
                           "close": 50.0, "stop_price": 45.0, "shares": 10, "cost": 500.0,
                           "atr14": 1.5, "score": 1.2}  # margin only 0.2, below the 1.0 bar

    old_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    state = {"cash": 10000.0, "positions": [
        {"ticker": "HELD", "sector": "Technology", "entry_date": old_date, "entry_price": 30.0,
         "shares": 10, "cost": 300.0, "stop_price": 25.0, "track": "global",
         "atr_at_entry": 1.0, "highest_close": 30.0},
    ]}

    rotation = evaluate_rotation([marginal_candidate], state, price_data={}, benchmarks={},
                                  scoring_params=(0, 1, 0, 1))
    assert rotation is None


def test_rotation_ignores_positions_held_less_than_minimum():
    strong_candidate = {"ticker": "STRONG", "sector": "Technology", "track": "global",
                         "close": 50.0, "stop_price": 45.0, "shares": 10, "cost": 500.0,
                         "atr14": 1.5, "score": 5.0}  # huge margin, but held position is too new

    recent_date = datetime.date.today().isoformat()
    state = {"cash": 10000.0, "positions": [
        {"ticker": "NEW", "sector": "Technology", "entry_date": recent_date, "entry_price": 30.0,
         "shares": 10, "cost": 300.0, "stop_price": 25.0, "track": "global",
         "atr_at_entry": 1.0, "highest_close": 30.0},
    ]}

    rotation = evaluate_rotation([strong_candidate], state, price_data={}, benchmarks={},
                                  scoring_params=(0, 1, 0, 1))
    assert rotation is None  # NEW hasn't cleared the 14-day minimum holding period


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])