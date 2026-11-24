"""
portfolio.py
Tracks the paper-trading portfolio state: cash, open positions, and
how much capital / how many positions sit in each sector. Supports
multiple independent scenarios via separate state files -- "zero" is
the clean-start one we're building first; a "live" one seeded with
real holdings can be added the same way later.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

STARTING_CASH = 25000.0


def portfolio_file(scenario: str = "zero") -> Path:
    return DATA_DIR / f"portfolio_{scenario}.json"


def load_portfolio(scenario: str = "zero") -> dict:
    path = portfolio_file(scenario)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": []}


def save_portfolio(state: dict, scenario: str = "zero"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(portfolio_file(scenario), "w") as f:
        json.dump(state, f, indent=2)


def position_count(state: dict) -> int:
    return len(state["positions"])


def sector_counts(state: dict) -> dict:
    counts = {}
    for p in state["positions"]:
        counts[p["sector"]] = counts.get(p["sector"], 0) + 1
    return counts


def sector_capital(state: dict) -> dict:
    capital = {}
    for p in state["positions"]:
        capital[p["sector"]] = capital.get(p["sector"], 0.0) + p["cost"]
    return capital


def total_invested(state: dict) -> float:
    return sum(p["cost"] for p in state["positions"])


def held_tickers(state: dict) -> set:
    return {p["ticker"] for p in state["positions"]}


def add_position(state: dict, ticker: str, sector: str, entry_date: str,
                  entry_price: float, shares: int, stop_price: float, track: str,
                  atr_at_entry: float) -> dict:
    cost = shares * entry_price
    state["positions"].append({
        "ticker": ticker,
        "sector": sector,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "shares": shares,
        "cost": cost,
        "stop_price": stop_price,
        "track": track,
        "atr_at_entry": atr_at_entry,
        "highest_close": entry_price,
    })
    state["cash"] -= cost
    return state


def update_highest_close(state: dict, ticker: str, highest_close: float) -> dict:
    for p in state["positions"]:
        if p["ticker"] == ticker:
            p["highest_close"] = highest_close
    return state


def remove_position(state: dict, ticker: str, exit_price: float) -> dict:
    position = next((p for p in state["positions"] if p["ticker"] == ticker), None)
    if position is None:
        raise ValueError(f"No open position for {ticker}")
    proceeds = position["shares"] * exit_price
    state["cash"] += proceeds
    state["positions"] = [p for p in state["positions"] if p["ticker"] != ticker]
    return state


if __name__ == "__main__":
    state = load_portfolio("zero")
    print("Loaded portfolio state:")
    print(f"  Cash: {state['cash']}")
    print(f"  Open positions: {position_count(state)}")
    print(f"  Sector counts: {sector_counts(state)}")
    save_portfolio(state, "zero")
    print(f"\nSaved to {portfolio_file('zero')}")