"""
universe.py
Builds and validates the trading universe for both tracks (Europe + Global),
and saves a dated snapshot for reproducible backtesting.
"""

import json
import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

# Track A: Europe (EURO STOXX 50 + AEX), Yahoo Finance ticker format.
# The last 5 are AEX members not fully confirmed for 2026 -- validation
# below will drop any that don't actually resolve.
EUROPE_TICKERS = [
    "ASML.AS", "MC.PA", "TTE.PA", "SAP.DE", "SIE.DE", "SAN.PA", "OR.PA",
    "SU.PA", "AI.PA", "ALV.DE", "AIR.PA", "RMS.PA", "IBE.MC", "DTE.DE",
    "DG.PA", "BNP.PA", "MBG.DE", "SAN.MC", "SAF.PA", "EL.PA", "CS.PA",
    "BAYN.DE", "IFX.DE", "PRX.AS", "ABI.BR", "ENEL.MI", "MUV2.DE",
    "INGA.AS", "ADYEN.AS", "DHL.DE", "BBVA.MC", "BAS.DE", "RI.PA",
    "ISP.MI", "ITX.MC", "KER.PA", "UCG.MI", "STLAM.MI", "CRH",
    "NDA-FI.HE", "BMW.DE", "BN.PA", "FLUT", "DB1.DE", "ENI.MI",
    "AD.AS", "ADS.DE", "VOW3.DE", "NOKIA.HE", "VNA.DE",
    "SHELL.AS", "UNA.AS", "REN.AS", "ASM.AS", "ABN.AS", "MT.AS", "NN.AS",
    "HEIA.AS", "DSFIR.AS", "PHIA.AS", "WKL.AS", "UMG.AS", "KPN.AS",
    "BESI.AS", "ASRNL.AS",
    "AGN.AS", "AKZA.AS", "EXO.AS", "IMCD.AS", "RAND.AS",
]

# Manual additions to the Global track beyond S&P 500 + Nasdaq-100
# (e.g. very recent IPOs not yet in either index).
# Note: a stock with only days/weeks of price history won't have enough
# data yet for the 50-day trend filter or ATR(14) stop to compute --
# it'll sit inactive in the universe until enough trading days accumulate.
MANUAL_GLOBAL_ADDITIONS = [
    # "ANT",  # Anthropic -- uncomment to include
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_sp500_tickers() -> list:
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0]
    tickers = sp500["Symbol"].tolist()
    # Yahoo Finance uses '-' instead of '.' for share classes (e.g. BRK.B -> BRK-B)
    return [t.replace(".", "-") for t in tickers]


def fetch_nasdaq100_tickers() -> list:
    """Scrape the current Nasdaq-100 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        for col in table.columns:
            if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                return table[col].astype(str).str.replace(".", "-", regex=False).tolist()
    print("Could not auto-detect the ticker column. Tables found:")
    for i, table in enumerate(tables):
        print(f"  Table {i}: columns = {list(table.columns)}, rows = {len(table)}")
    raise ValueError("Could not find Nasdaq-100 constituents table")

def validate_tickers(tickers: list) -> tuple:
    """Try pulling a bit of data for each ticker; split into valid/invalid."""
    valid, invalid = [], []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d")
            (valid if not hist.empty else invalid).append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid


def build_universe_snapshot():
    print("Fetching S&P 500 list from Wikipedia...")
    sp500 = fetch_sp500_tickers()

    print("Fetching Nasdaq-100 list from Wikipedia...")
    nasdaq100 = fetch_nasdaq100_tickers()

    global_tickers = sorted(set(sp500) | set(nasdaq100) | set(MANUAL_GLOBAL_ADDITIONS))
    print(f"Combined Global universe: {len(global_tickers)} unique tickers")

    print(f"Validating {len(EUROPE_TICKERS)} Europe tickers...")
    europe_valid, europe_invalid = validate_tickers(EUROPE_TICKERS)

    print(f"Validating {len(global_tickers)} Global tickers (this will take a few minutes)...")
    global_valid, global_invalid = validate_tickers(global_tickers)

    snapshot = {
        "date": datetime.date.today().isoformat(),
        "europe": sorted(europe_valid),
        "global": sorted(global_valid),
    }

    filename = f"data/universe_{snapshot['date']}.json"
    with open(filename, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSaved snapshot to {filename}")
    print(f"Europe: {len(europe_valid)} valid, {len(europe_invalid)} invalid -> {europe_invalid}")
    print(f"Global: {len(global_valid)} valid, {len(global_invalid)} invalid -> {global_invalid}")


if __name__ == "__main__":
    build_universe_snapshot()