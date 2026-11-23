import json
import glob
import time
from pathlib import Path

import yfinance as yf

from core.signals import add_indicators
from core.risk import add_position_size
from profiles.profile_1_trend_momentum import add_entry_signal

PORTFOLIO_VALUE = 25000  # placeholder starting value -- adjust to your actual paper balance
CHUNK_SIZE = 100
LOOKBACK = "6mo"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_latest_universe() -> dict:
    files = sorted(DATA_DIR.glob("universe_*.json"))
    with open(files[-1]) as f:
        return json.load(f)


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def batch_download(tickers: list) -> dict:
    data = {}
    for group in chunked(tickers, CHUNK_SIZE):
        raw = yf.download(
            tickers=group,
            period=LOOKBACK,
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
        for ticker in group:
            try:
                df = raw[ticker].dropna(how="all")
                if not df.empty:
                    data[ticker] = df
            except KeyError:
                continue
    return data


def scan_for_entries(price_data: dict, portfolio_value: float) -> list:
    candidates = []
    for ticker, prices in price_data.items():
        if len(prices) < 60:
            continue
        try:
            df = add_indicators(prices)
            df = add_entry_signal(df)
            df = add_position_size(df, portfolio_value)
            today = df.iloc[-1]
            if today["entry_signal"]:
                candidates.append({
                    "ticker": ticker,
                    "close": round(today["Close"], 2),
                    "stop_price": round(today["stop_price"], 2),
                    "shares": int(today["position_shares"]),
                    "cost": round(today["position_cost"], 2),
                })
        except Exception as e:
            print(f"  Skipped {ticker}: {e}")
    return candidates


if __name__ == "__main__":
    universe = load_latest_universe()
    all_tickers = universe["europe"] + universe["global"]

    start = time.time()
    print(f"Downloading price history for {len(all_tickers)} tickers in batches of {CHUNK_SIZE}...")
    price_data = batch_download(all_tickers)
    print(f"Downloaded {len(price_data)} tickers in {time.time() - start:.1f}s")

    print("Scanning for entry signals...")
    results = scan_for_entries(price_data, PORTFOLIO_VALUE)

    print(f"\nFound {len(results)} entry candidate(s) today:\n")
    for c in results:
        print(f"  {c['ticker']}: close={c['close']}, stop={c['stop_price']}, "
              f"buy {c['shares']} shares (~{c['cost']})")